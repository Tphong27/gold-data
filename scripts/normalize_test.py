# scripts/normalize_test.py
# Normalize gold_live_new.csv -> gold_clean_new.csv
# Output columns match Google Sheet template (11 cols)
# Unit: VND/lượng
# Includes USD/VND fx and VN vs World (TG) comparisons

import os
import re
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

IN_CSV = "data/gold_live_new.csv"
OUT_CLEAN = "data/gold_clean_new.csv"
FX_CACHE_JSON = "data/fx_cache.json"
TZ_VN = "Asia/Ho_Chi_Minh"

TEXT_COLS = ["Ngày", "Thời điểm cập nhật giá mới", "Mã vàng", "Loại vàng", "Currency"]
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell", "Số lần update"]

# Unit conversion constants
GRAMS_PER_OZ_TROY = 31.1034768
GRAMS_PER_LUONG = 37.5
OZ_TO_LUONG = GRAMS_PER_LUONG / GRAMS_PER_OZ_TROY  # ~1.205653

# FX API (mid-market, open endpoint; no key)
FX_API_URL = os.getenv("FX_API_URL", "https://open.er-api.com/v6/latest/USD")
FX_TIMEOUT_SEC = int(os.getenv("FX_TIMEOUT_SEC", "20"))


def to_number(x):
    """Robust numeric parsing for strings like 1.234.567 or 1,234.56 or ' 171300000 ' """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "")
    if s == "":
        return np.nan

    s = re.sub(r"[^0-9,.\-]", "", s)

    if "," in s and "." in s:
        # pick decimal separator by last occurrence
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # 1.234,56 -> 1234.56
        else:
            s = s.replace(",", "")  # 1,234.56 -> 1234.56
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) in (1, 2):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        # if last part is not 1-2 digits => thousand separators
        if len(parts[-1]) not in (1, 2):
            s = s.replace(".", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def ensure_text(df: pd.DataFrame):
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()


def parse_timestamp_vn(df: pd.DataFrame):
    """
    Input has:
      - Ngày (YYYY-MM-DD)
      - Thời điểm cập nhật giá mới (HH:MM:SS)
    """
    df["timestamp_vn"] = pd.to_datetime(
        df["Ngày"].astype(str) + " " + df["Thời điểm cập nhật giá mới"].astype(str),
        errors="coerce",
    )
    # localize to VN tz
    if getattr(df["timestamp_vn"].dt, "tz", None) is None:
        df["timestamp_vn"] = df["timestamp_vn"].dt.tz_localize(
            TZ_VN, nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        df["timestamp_vn"] = df["timestamp_vn"].dt.tz_convert(TZ_VN)


def ensure_numeric(df: pd.DataFrame):
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].apply(to_number)


def _load_fx_cache():
    if not os.path.exists(FX_CACHE_JSON):
        return None
    try:
        with open(FX_CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_fx_cache(rate: float):
    os.makedirs(os.path.dirname(FX_CACHE_JSON) or ".", exist_ok=True)
    payload = {
        "usd_vnd": rate,
        "source_url": FX_API_URL,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(FX_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fetch_usd_vnd_midmarket() -> float:
    """
    Fetch USD->VND mid-market FX.
    Fallback:
      - Use cache if API fails
      - Or use env FX_VND_PER_USD if you set it
    """
    env_fx = os.getenv("FX_VND_PER_USD")
    if env_fx:
        try:
            return float(env_fx)
        except Exception:
            pass

    try:
        r = requests.get(FX_API_URL, timeout=FX_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()

        rate = None
        conv = data.get("conversion_rates") if isinstance(data, dict) else None
        if isinstance(conv, dict) and "VND" in conv:
            rate = float(conv["VND"])
        if rate is None:
            rates = data.get("rates") if isinstance(data, dict) else None
            if isinstance(rates, dict) and "VND" in rates:
                rate = float(rates["VND"])

        if rate is None:
            raise ValueError("FX response missing VND rate")

        _save_fx_cache(rate)
        return rate

    except Exception as e:
        cache = _load_fx_cache()
        if cache and "usd_vnd" in cache:
            print(f"⚠️ FX API failed ({e}). Using cached USD/VND={cache['usd_vnd']}")
            return float(cache["usd_vnd"])
        raise RuntimeError(
            f"FX API failed and no cache available. "
            f"Set env FX_VND_PER_USD or fix FX_API_URL. Error: {e}"
        )


def convert_xauusd_to_vnd_luong(df: pd.DataFrame, fx_usd_vnd: float) -> pd.DataFrame:
    """
    Convert rows where Mã vàng == 'XAUUSD' from USD/oz -> VND/lượng.
    Output: Giá mua/Giá bán for XAUUSD become VND/lượng
    """
    df = df.copy()

    mask = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")

    # force numeric columns (avoid pandas lossy assignment)
    df["Giá mua"] = pd.to_numeric(df["Giá mua"], errors="coerce").astype("float64")
    df["Giá bán"] = pd.to_numeric(df["Giá bán"], errors="coerce").astype("float64")

    # store fx column for all rows (so sheet can show it)
    df["fx_usd_vnd"] = float(fx_usd_vnd)

    if mask.any():
        buy_usd = df.loc[mask, "Giá mua"].to_numpy(dtype="float64")
        sell_usd = df.loc[mask, "Giá bán"].to_numpy(dtype="float64")

        df.loc[mask, "Giá mua"] = buy_usd * fx_usd_vnd * OZ_TO_LUONG
        df.loc[mask, "Giá bán"] = sell_usd * fx_usd_vnd * OZ_TO_LUONG

        # after normalization, treat currency as VND
        if "Currency" in df.columns:
            df.loc[mask, "Currency"] = "VND"

    return df


def dedup(df: pd.DataFrame):
    """
    Dedup by snapshot timestamp + code
    """
    df = df.copy()
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df)
    return df, before, after


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output columns (match image):
      Ngày
      Thời điểm cập nhật giá mới
      Mã vàng
      Loại vàng
      Giá mua
      Giá bán
      Tỷ giá USD/VND
      Chênh lệch giá VN – TG
      Tỷ lệ chênh lệch (%)
      Chênh lệch giá mua
      Chênh lệch giá bán
    All prices are VND/lượng (including TG after XAUUSD conversion)
    """
    df = df.copy()

    # sort for diffs
    df = df.sort_values(["Mã vàng", "timestamp_vn"])
    df["diff_buy"] = df.groupby("Mã vàng")["Giá mua"].diff()
    df["diff_sell"] = df.groupby("Mã vàng")["Giá bán"].diff()

    # world price by timestamp from XAUUSD (already VND/lượng)
    world = (
        df[df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")]
        [["timestamp_vn", "Giá bán", "Giá mua"]]
        .rename(columns={"Giá bán": "world_sell_vnd", "Giá mua": "world_buy_vnd"})
        .drop_duplicates(subset=["timestamp_vn"], keep="last")
    )

    df = df.merge(world, on="timestamp_vn", how="left")

    is_world = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")

    # Chênh lệch giá VN – TG = giá bán VN - giá bán TG
    df["diff_vn_tg"] = np.where(
        (~is_world) & df["world_sell_vnd"].notna(),
        df["Giá bán"] - df["world_sell_vnd"],
        np.nan,
    )

    # Tỷ lệ chênh lệch (%) = (Chênh lệch / Giá bán TG) * 100
    df["diff_pct"] = np.where(
        (~is_world) & df["world_sell_vnd"].notna() & (df["world_sell_vnd"] != 0),
        (df["diff_vn_tg"] / df["world_sell_vnd"]) * 100.0,
        np.nan,
    )

    out = pd.DataFrame(
        {
            "Ngày": df["Ngày"],
            "Thời điểm cập nhật giá mới": df["Thời điểm cập nhật giá mới"],
            "Mã vàng": df["Mã vàng"],
            "Loại vàng": df["Loại vàng"],
            "Giá mua": df["Giá mua"],
            "Giá bán": df["Giá bán"],
            "Tỷ giá USD/VND": df["fx_usd_vnd"],
            "Chênh lệch giá VN – TG": df["diff_vn_tg"],
            "Tỷ lệ chênh lệch (%)": df["diff_pct"],
            "Chênh lệch giá mua": df["diff_buy"],
            "Chênh lệch giá bán": df["diff_sell"],
        }
    )

    # final sort nicer for sheet
    out["_ts"] = pd.to_datetime(
        out["Ngày"].astype(str) + " " + out["Thời điểm cập nhật giá mới"].astype(str),
        errors="coerce",
    )
    out = out.sort_values(["_ts", "Mã vàng"]).drop(columns=["_ts"])

    return out


def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    df = pd.read_csv(
        IN_CSV,
        dtype={"Mã vàng": "string", "Loại vàng": "string", "Currency": "string"},
    )

    print("Loaded:", df.shape)
    print("Columns:", list(df.columns))

    ensure_text(df)
    parse_timestamp_vn(df)
    ensure_numeric(df)

    fx_usd_vnd = fetch_usd_vnd_midmarket()
    print(f"FX USD/VND used (mid-market): {fx_usd_vnd}")

    df = convert_xauusd_to_vnd_luong(df, fx_usd_vnd)

    df, before, after = dedup(df)

    out = build_output(df)

    os.makedirs(os.path.dirname(OUT_CLEAN) or ".", exist_ok=True)
    out.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")

    print(f"✅ Saved: {OUT_CLEAN}")
    print(f"dedup: {before} -> {after}")
    null_ts = pd.to_datetime(
        out["Ngày"].astype(str) + " " + out["Thời điểm cập nhật giá mới"].astype(str),
        errors="coerce",
    ).isna().sum()
    print("null timestamp:", int(null_ts))


if __name__ == "__main__":
    main()
