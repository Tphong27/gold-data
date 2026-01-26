# scripts/normalize_test.py
# Normalize + clean gold_live.csv -> gold_clean.csv (with FX + XAUUSD USD/oz -> VND/luong)
# Run:  python scripts/normalize_test.py

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
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell", "Số lần update"]

# Unit conversion: USD/oz -> USD/luong
GRAMS_PER_OZ_TROY = 31.1034768
GRAMS_PER_LUONG = 37.5
OZ_TO_LUONG = GRAMS_PER_LUONG / GRAMS_PER_OZ_TROY  # ~1.205653

# FX API (mid-market)
FX_API_URL = os.getenv("FX_API_URL", "https://open.er-api.com/v6/latest/USD")  # open access :contentReference[oaicite:1]{index=1}
FX_TIMEOUT_SEC = int(os.getenv("FX_TIMEOUT_SEC", "20"))


def to_number(x):
    """
    Convert strings like:
      - 17,130,000
      - 17.130.000
      - 4985.1
      - 4,985.10
      - 1.234,56
    into float safely.
    """
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
            s = s.replace(".", "").replace(",", ".")  # decimal comma
        else:
            s = s.replace(",", "")  # thousands comma
    elif "." in s:
        parts = s.split(".")
        if len(parts[-1]) in (1, 2):
            pass  # decimal dot
        else:
            s = s.replace(".", "")  # thousands dot

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

    if "Số lần update" in df.columns:
        # keep as integer-like when possible
        df["Số lần update"] = df["Số lần update"].astype("Int64")


def _load_fx_cache():
    if not os.path.exists(FX_CACHE_JSON):
        return None
    try:
        with open(FX_CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_fx_cache(rate: float, source_url: str):
    os.makedirs(os.path.dirname(FX_CACHE_JSON) or ".", exist_ok=True)
    payload = {
        "usd_vnd": rate,
        "source_url": source_url,
        "saved_at_utc": datetime.now(timezone.utc).isoformat()
    }
    with open(FX_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fetch_usd_vnd_midmarket() -> float:
    """
    Fetch USD->VND mid-market rate from FX_API_URL.
    Default: open.er-api.com open endpoint. :contentReference[oaicite:2]{index=2}

    Fallback order:
      1) ENV FX_VND_PER_USD (if set)
      2) Cache file data/fx_cache.json
      3) Raise error
    """
    env_fx = os.getenv("FX_VND_PER_USD")
    if env_fx:
        try:
            rate = float(env_fx)
            return rate
        except Exception:
            pass

    # Try API
    try:
        r = requests.get(FX_API_URL, timeout=FX_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()

        # open.er-api.com schema: conversion_rates.VND
        rate = None
        if isinstance(data, dict):
            conv = data.get("conversion_rates")
            if isinstance(conv, dict) and "VND" in conv:
                rate = float(conv["VND"])

            # some APIs might use "rates"
            if rate is None:
                rates = data.get("rates")
                if isinstance(rates, dict) and "VND" in rates:
                    rate = float(rates["VND"])

        if rate is None:
            raise ValueError("FX response missing VND rate")

        _save_fx_cache(rate, FX_API_URL)
        return rate

    except Exception as e:
        # Cache fallback
        cache = _load_fx_cache()
        if cache and "usd_vnd" in cache:
            print(f"⚠️ FX API failed ({e}). Using cached USD/VND={cache['usd_vnd']}")
            return float(cache["usd_vnd"])

        raise RuntimeError(
            f"FX API failed and no cache available. "
            f"Set env FX_VND_PER_USD to a number, or fix FX_API_URL. Error: {e}"
        )


def add_fx_and_vnd_luong_columns(df: pd.DataFrame, fx_usd_vnd: float) -> pd.DataFrame:
    """
    Add normalized columns in VND/luong for all codes.
    - For VND-based rows: copy buy/sell to buy_vnd_luong/sell_vnd_luong
    - For XAUUSD (USD/oz): convert to VND/luong using fx and OZ_TO_LUONG
    """
    df = df.copy()

    # Defaults: assume already VND/luong
    df["currency_norm"] = "VND"
    df["unit_norm"] = "luong"
    df["fx_usd_vnd"] = pd.NA

    df["buy_vnd_luong"] = df.get("Giá mua", pd.NA)
    df["sell_vnd_luong"] = df.get("Giá bán", pd.NA)

    mask = df["Mã vàng"].astype(str).eq("XAUUSD")
    if mask.any():
        df.loc[mask, "fx_usd_vnd"] = fx_usd_vnd

        # USD/oz -> VND/luong
        df.loc[mask, "buy_vnd_luong"] = df.loc[mask, "Giá mua"] * fx_usd_vnd * OZ_TO_LUONG
        df.loc[mask, "sell_vnd_luong"] = df.loc[mask, "Giá bán"] * fx_usd_vnd * OZ_TO_LUONG

    df["mid_vnd_luong"] = (df["buy_vnd_luong"] + df["sell_vnd_luong"]) / 2
    df["spread_vnd_luong"] = df["sell_vnd_luong"] - df["buy_vnd_luong"]

    return df


def add_features(df: pd.DataFrame):
    # Original features on raw buy/sell
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

    # Normalized deltas on mid_vnd_luong (useful for DSS)
    if "mid_vnd_luong" in df.columns:
        df = df.sort_values(["Mã vàng", "timestamp_vn"])
        df["delta_mid_vnd_luong"] = df.groupby("Mã vàng")["mid_vnd_luong"].diff()
    else:
        df["delta_mid_vnd_luong"] = pd.NA

    return df


def dedup(df: pd.DataFrame):
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df)
    return df, before, after


def sanity_checks(df: pd.DataFrame):
    if "Mã vàng" in df.columns:
        mask = df["Mã vàng"].eq("XAUUSD")
        if mask.any():
            cols = [c for c in [
                "timestamp_vn", "Giá mua", "Giá bán", "fx_usd_vnd",
                "buy_vnd_luong", "sell_vnd_luong"
            ] if c in df.columns]
            print("XAUUSD sample (tail):")
            print(df.loc[mask, cols].tail(3).to_string(index=False))


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

    # 4) fetch FX (mid-market) + convert XAUUSD USD/oz -> VND/luong
    fx_usd_vnd = fetch_usd_vnd_midmarket()
    print(f"FX USD/VND used (mid-market): {fx_usd_vnd}")

    df = add_fx_and_vnd_luong_columns(df, fx_usd_vnd)

    # 5) features + deltas
    df = add_features(df)

    # 6) dedup & sort
    df, before, after = dedup(df)
    df = df.sort_values(["Mã vàng", "timestamp_vn"])

    # 7) save (keep a clean column order)
    preferred_cols = [
        "timestamp_vn", "Ngày", "Thời điểm",
        "Mã vàng", "Loại vàng",
        "Currency", "currency_norm", "unit_norm", "fx_usd_vnd",
        "Giá mua", "Giá bán", "spread", "mid",
        "buy_vnd_luong", "sell_vnd_luong", "spread_vnd_luong", "mid_vnd_luong",
        "delta_buy", "delta_sell", "delta_mid_vnd_luong",
        "Day change buy", "Day change sell", "Số lần update",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    os.makedirs(os.path.dirname(OUT_CLEAN) or ".", exist_ok=True)
    df.to_csv(OUT_CLEAN, index=False)

    print(f"✅ Saved: {OUT_CLEAN}")
    print(f"dedup: {before} -> {after}")
    print("null timestamp:", int(df["timestamp_vn"].isna().sum()))
    print("time range:", df["timestamp_vn"].min(), "->", df["timestamp_vn"].max())

    sanity_checks(df)


if __name__ == "__main__":
    main()
