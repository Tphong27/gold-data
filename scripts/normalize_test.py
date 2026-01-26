# scripts/normalize_test.py
# Output gold_clean.csv with columns like Google Sheet template

import os
import re
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

LIVE_CSV = "data/gold_live.csv"
OUT_CLEAN = "data/gold_clean.csv"
FX_CACHE_JSON = "data/fx_cache.json"
TZ_VN = "Asia/Ho_Chi_Minh"

TEXT_COLS = ["Ngày", "Thời điểm", "Mã vàng", "Loại vàng", "Currency"]
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell", "Số lần update"]

# Unit conversion constants
GRAMS_PER_OZ_TROY = 31.1034768
GRAMS_PER_LUONG = 37.5
OZ_TO_LUONG = GRAMS_PER_LUONG / GRAMS_PER_OZ_TROY  # ~1.205653

# FX API (mid-market, open endpoint; no key)
FX_API_URL = os.getenv("FX_API_URL", "https://open.er-api.com/v6/latest/USD")
FX_TIMEOUT_SEC = int(os.getenv("FX_TIMEOUT_SEC", "20"))


def to_number(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(" ", "")
    if s == "":
        return pd.NA

    s = re.sub(r"[^0-9,.\-]", "", s)

    if "," in s and "." in s:
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
        if len(parts[-1]) in (1, 2):
            pass
        else:
            s = s.replace(".", "")

    try:
        return float(s)
    except Exception:
        return pd.NA


def ensure_text(df: pd.DataFrame):
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()


def parse_timestamp_vn(df: pd.DataFrame):
    df["timestamp_vn"] = pd.to_datetime(df["Ngày"] + " " + df["Thời điểm"], errors="coerce")
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
        "saved_at_utc": datetime.now(timezone.utc).isoformat()
    }
    with open(FX_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fetch_usd_vnd_midmarket() -> float:
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
    Also store:
      - buy_usd_oz (USD/oz)
      - sell_usd_oz (USD/oz)
      - fx_usd_vnd (USD/VND)
    """
    df = df.copy()

    mask = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")
    df["fx_usd_vnd"] = np.nan
    df.loc[mask, "fx_usd_vnd"] = float(fx_usd_vnd)

    # preserve original USD/oz
    df["buy_usd_oz"] = np.nan
    df["sell_usd_oz"] = np.nan
    if mask.any():
        df.loc[mask, "buy_usd_oz"] = pd.to_numeric(df.loc[mask, "Giá mua"], errors="coerce").to_numpy(dtype="float64")
        df.loc[mask, "sell_usd_oz"] = pd.to_numeric(df.loc[mask, "Giá bán"], errors="coerce").to_numpy(dtype="float64")

    # force numeric
    df["Giá mua"] = pd.to_numeric(df["Giá mua"], errors="coerce").astype("float64")
    df["Giá bán"] = pd.to_numeric(df["Giá bán"], errors="coerce").astype("float64")

    # convert for XAUUSD rows
    if mask.any():
        buy_arr = pd.to_numeric(df.loc[mask, "buy_usd_oz"], errors="coerce").to_numpy(dtype="float64")
        sell_arr = pd.to_numeric(df.loc[mask, "sell_usd_oz"], errors="coerce").to_numpy(dtype="float64")

        df.loc[mask, "Giá mua"] = buy_arr * fx_usd_vnd * OZ_TO_LUONG
        df.loc[mask, "Giá bán"] = sell_arr * fx_usd_vnd * OZ_TO_LUONG
        df.loc[mask, "Currency"] = "VND"

    return df


def dedup(df: pd.DataFrame):
    df = df.copy()
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df)
    return df, before, after


def build_output_columns(df: pd.DataFrame, fx_usd_vnd: float) -> pd.DataFrame:
    """
    Build final columns like the sheet:
    Ngày | Thời điểm cập nhật giá mới | Mã vàng | Loại vàng | Giá mua | Giá bán |
    Tỷ giá USD/VND | Chênh lệch giá VN – TG | Tỷ lệ chênh lệch (%) |
    Chêch lệch giá mua | Chênh lệch giá bán | Giá tiền đô | unit_standard
    """
    df = df.copy()

    # Ensure base
    df["fx_usd_vnd"] = df.get("fx_usd_vnd")
    if df["fx_usd_vnd"].isna().all():
        df["fx_usd_vnd"] = float(fx_usd_vnd)

    # delta buy/sell by code
    df = df.sort_values(["Mã vàng", "timestamp_vn"])
    df["diff_buy"] = df.groupby("Mã vàng")["Giá mua"].diff()
    df["diff_sell"] = df.groupby("Mã vàng")["Giá bán"].diff()

    # World price (TG) per timestamp from XAUUSD (already VND/luong)
    world = (
        df[df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")]
        [["timestamp_vn", "Giá bán", "Giá mua"]]
        .rename(columns={"Giá bán": "world_sell_vnd", "Giá mua": "world_buy_vnd"})
        .drop_duplicates(subset=["timestamp_vn"], keep="last")
    )

    df = df.merge(world, on="timestamp_vn", how="left")

    # Chênh lệch VN – TG: use SELL comparison
    # Only for VN codes (not XAUUSD)
    is_world = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")

    df["diff_vn_world_sell"] = np.where(
        (~is_world) & df["world_sell_vnd"].notna(),
        df["Giá bán"] - df["world_sell_vnd"],
        np.nan
    )

    df["diff_pct"] = np.where(
        (~is_world) & df["world_sell_vnd"].notna() & (df["world_sell_vnd"] != 0),
        (df["diff_vn_world_sell"] / df["world_sell_vnd"]) * 100.0,
        np.nan
    )

    # unit_standard
    df["unit_standard"] = "VND/lượng"

    # "Giá tiền đô" -> keep raw USD/oz buy (for world row), else blank
    df["gia_tien_do"] = df.get("buy_usd_oz")
    df.loc[~is_world, "gia_tien_do"] = np.nan

    # Build final dataframe with sheet headers
    out = pd.DataFrame({
        "Ngày": df["Ngày"],
        "Thời điểm cập nhật giá mới": df["Thời điểm"],
        "Mã vàng": df["Mã vàng"],
        "Loại vàng": df["Loại vàng"],
        "Giá mua": df["Giá mua"],
        "Giá bán": df["Giá bán"],
        "Tỷ giá USD/VND": df["fx_usd_vnd"],
        "Chênh lệch giá VN – TG (giá bán VN - giá bán TG)": df["diff_vn_world_sell"],
        "Tỷ lệ chênh lệch (%)": df["diff_pct"],
        "Chêch lệch giá mua": df["diff_buy"],
        "Chênh lệch giá bán": df["diff_sell"],
        "Giá tiền đô": df["gia_tien_do"],
        "unit_standard": df["unit_standard"],
    })

    return out


def main():
    if not os.path.exists(LIVE_CSV):
        raise FileNotFoundError(f"Missing input file: {LIVE_CSV}")

    df = pd.read_csv(
        LIVE_CSV,
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

    # Dedup and sort
    df, before, after = dedup(df)
    df = df.sort_values(["timestamp_vn", "Mã vàng"])

    # Build final output columns per sheet
    out = build_output_columns(df, fx_usd_vnd)

    os.makedirs(os.path.dirname(OUT_CLEAN) or ".", exist_ok=True)
    out.to_csv(OUT_CLEAN, index=False)

    print(f"✅ Saved: {OUT_CLEAN}")
    print(f"dedup: {before} -> {after}")
    print("null timestamp:", int(pd.to_datetime(out["Ngày"].astype(str) + " " + out["Thời điểm cập nhật giá mới"].astype(str), errors="coerce").isna().sum()))


if __name__ == "__main__":
    main()
