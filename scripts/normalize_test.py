# scripts/normalize_test.py
# Normalize + clean gold_live.csv -> gold_clean.csv
# - Rename day-change columns
# - Add unit column "Lượng"
# - Drop update columns
# - Convert XAUUSD USD/oz -> VND/lượng using real-time mid-market FX

import os
import re
import json
from datetime import datetime, timezone

import pandas as pd
import requests

LIVE_CSV = "data/gold_live.csv"
OUT_CLEAN = "data/gold_clean.csv"
FX_CACHE_JSON = "data/fx_cache.json"
TZ_VN = "Asia/Ho_Chi_Minh"

TEXT_COLS = ["Ngày", "Thời điểm", "Mã vàng", "Loại vàng", "Currency"]
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell"]

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

        # open.er-api.com schema: conversion_rates.VND
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
    We'll output standardized VND prices in columns:
      - Giá mua (VND/lượng)
      - Giá bán (VND/lượng)
    Also keep original USD columns in:
      - Giá mua (USD/oz) gốc -> buy_usd_oz
      - Giá bán (USD/oz) gốc -> sell_usd_oz
    """
    df = df.copy()

    # Add unit column required by user
    df["Lượng"] = "lượng"

    # Identify world gold rows
    mask = df["Mã vàng"].astype(str).eq("XAUUSD")
    if not mask.any():
        df["fx_usd_vnd"] = pd.NA
        return df

    df["fx_usd_vnd"] = pd.NA
    df.loc[mask, "fx_usd_vnd"] = fx_usd_vnd

    # Preserve original USD/oz
    df["buy_usd_oz"] = pd.NA
    df["sell_usd_oz"] = pd.NA
    df.loc[mask, "buy_usd_oz"] = df.loc[mask, "Giá mua"]
    df.loc[mask, "sell_usd_oz"] = df.loc[mask, "Giá bán"]

    # Convert to VND/luong
    df.loc[mask, "Giá mua"] = df.loc[mask, "buy_usd_oz"] * fx_usd_vnd * OZ_TO_LUONG
    df.loc[mask, "Giá bán"] = df.loc[mask, "sell_usd_oz"] * fx_usd_vnd * OZ_TO_LUONG

    # Currency after normalization
    df.loc[mask, "Currency"] = "VND"

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "Day change buy": "Chênh lệch giá mua (Hôm qua→Nay)",
        "Day change sell": "Chênh lệch giá bán (Hôm qua→Nay)",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = []
    for c in ["Số lần update", "Thời điểm cập nhật dữ liệu"]:
        if c in df.columns:
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Giá mua" in df.columns and "Giá bán" in df.columns:
        df["spread"] = df["Giá bán"] - df["Giá mua"]
        df["mid"] = (df["Giá mua"] + df["Giá bán"]) / 2

        df = df.sort_values(["Mã vàng", "timestamp_vn"])
        df["delta_buy"] = df.groupby("Mã vàng")["Giá mua"].diff()
        df["delta_sell"] = df.groupby("Mã vàng")["Giá bán"].diff()
    else:
        df["spread"] = pd.NA
        df["mid"] = pd.NA
        df["delta_buy"] = pd.NA
        df["delta_sell"] = pd.NA
    return df


def dedup(df: pd.DataFrame):
    df = df.copy()
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df)
    return df, before, after


def main():
    if not os.path.exists(LIVE_CSV):
        raise FileNotFoundError(f"Missing input file: {LIVE_CSV}")

    df = pd.read_csv(
        LIVE_CSV,
        dtype={"Mã vàng": "string", "Loại vàng": "string", "Currency": "string"},
    )

    print("Loaded:", df.shape)
    print("Columns:", list(df.columns))

    # 1) clean text columns
    ensure_text(df)

    # 2) parse VN timestamp
    parse_timestamp_vn(df)

    # 3) convert numerics
    ensure_numeric(df)

    # 4) fetch FX mid-market + convert XAUUSD USD/oz -> VND/lượng
    fx_usd_vnd = fetch_usd_vnd_midmarket()
    print(f"FX USD/VND used (mid-market): {fx_usd_vnd}")

    df = convert_xauusd_to_vnd_luong(df, fx_usd_vnd)

    # 5) drop unwanted columns
    df = drop_unwanted_columns(df)

    # 6) rename columns
    df = rename_columns(df)

    # 7) add features
    df = add_features(df)

    # 8) dedup & sort
    df, before, after = dedup(df)
    df = df.sort_values(["Mã vàng", "timestamp_vn"])

    # 9) save
    os.makedirs(os.path.dirname(OUT_CLEAN) or ".", exist_ok=True)
    df.to_csv(OUT_CLEAN, index=False)

    print(f"✅ Saved: {OUT_CLEAN}")
    print(f"dedup: {before} -> {after}")
    print("null timestamp:", int(df["timestamp_vn"].isna().sum()))
    print("time range:", df["timestamp_vn"].min(), "->", df["timestamp_vn"].max())


if __name__ == "__main__":
    main()
