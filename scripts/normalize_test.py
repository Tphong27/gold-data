# scripts/normalize_test.py
import os
import re
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# Cập nhật đường dẫn file theo yêu cầu mới
LIVE_CSV = "data/gold_live_new.csv"
OUT_CLEAN = "data/gold_clean_new.csv"
FX_CACHE_JSON = "data/fx_cache.json"
TZ_VN = "Asia/Ho_Chi_Minh"

# Cập nhật TEXT_COLS khớp với headers mới của gold_live_new.csv
TEXT_COLS = ["Ngày", "Thời điểm cập nhật giá mới", "Mã vàng", "Loại vàng", "Currency"]
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell", "Số lần update"]

# Hệ số chuyển đổi
GRAMS_PER_OZ_TROY = 31.1034768
GRAMS_PER_LUONG = 37.5
OZ_TO_LUONG = GRAMS_PER_LUONG / GRAMS_PER_OZ_TROY 

FX_API_URL = os.getenv("FX_API_URL", "https://open.er-api.com/v6/latest/USD")
FX_TIMEOUT_SEC = int(os.getenv("FX_TIMEOUT_SEC", "20"))

def to_number(x):
    if pd.isna(x): return pd.NA
    s = str(x).strip().replace(" ", "")
    if s == "": return pd.NA
    s = re.sub(r"[^0-9,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."): s = s.replace(".", "").replace(",", ".")
        else: s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) in (1, 2): s = s.replace(".", "").replace(",", ".")
        else: s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        if len(parts[-1]) in (1, 2): pass
        else: s = s.replace(".", "")
    try: return float(s)
    except Exception: return pd.NA

def ensure_text(df: pd.DataFrame):
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

def parse_timestamp_vn(df: pd.DataFrame):
    # Sử dụng cột "Thời điểm cập nhật giá mới" từ file live_new
    df["timestamp_vn"] = pd.to_datetime(df["Ngày"] + " " + df["Thời điểm cập nhật giá mới"], errors="coerce")
    if getattr(df["timestamp_vn"].dt, "tz", None) is None:
        df["timestamp_vn"] = df["timestamp_vn"].dt.tz_localize(TZ_VN, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df["timestamp_vn"] = df["timestamp_vn"].dt.tz_convert(TZ_VN)

def ensure_numeric(df: pd.DataFrame):
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].apply(to_number)

def fetch_usd_vnd_midmarket() -> float:
    env_fx = os.getenv("FX_VND_PER_USD")
    if env_fx:
        try: return float(env_fx)
        except: pass
    try:
        r = requests.get(FX_API_URL, timeout=FX_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()
        rate = data.get("conversion_rates", {}).get("VND") or data.get("rates", {}).get("VND")
        if rate: return float(rate)
        raise ValueError("No rate found")
    except Exception as e:
        if os.path.exists(FX_CACHE_JSON):
            with open(FX_CACHE_JSON, "r") as f: return json.load(f)["usd_vnd"]
        raise e

def convert_xauusd_to_vnd_luong(df: pd.DataFrame, fx_usd_vnd: float) -> pd.DataFrame:
    df = df.copy()
    mask = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")
    df["fx_usd_vnd"] = np.nan
    df.loc[mask, "fx_usd_vnd"] = float(fx_usd_vnd)
    df["buy_usd_oz"] = np.nan
    if mask.any():
        df.loc[mask, "buy_usd_oz"] = pd.to_numeric(df.loc[mask, "Giá mua"], errors="coerce")
    df["Giá mua"] = pd.to_numeric(df["Giá mua"], errors="coerce").astype("float64")
    df["Giá bán"] = pd.to_numeric(df["Giá bán"], errors="coerce").astype("float64")
    if mask.any():
        df.loc[mask, "Giá mua"] = df.loc[mask, "buy_usd_oz"] * fx_usd_vnd * OZ_TO_LUONG
        df.loc[mask, "Giá bán"] = 0 # Thường XAUUSD không có giá bán trong raw của bạn
        df.loc[mask, "Currency"] = "VND"
    return df

def main():
    if not os.path.exists(LIVE_CSV):
        raise FileNotFoundError(f"Missing: {LIVE_CSV}")

    df = pd.read_csv(LIVE_CSV, dtype={"Mã vàng": "string", "Loại vàng": "string"})
    ensure_text(df)
    parse_timestamp_vn(df)
    ensure_numeric(df)

    fx_usd_vnd = fetch_usd_vnd_midmarket()
    df = convert_xauusd_to_vnd_luong(df, fx_usd_vnd)

    # Dedup & Sort
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    df = df.sort_values(["timestamp_vn", "Mã vàng"])

    # Tính toán chênh lệch
    df["diff_buy"] = df.groupby("Mã vàng")["Giá mua"].diff()
    df["diff_sell"] = df.groupby("Mã vàng")["Giá bán"].diff()

    # Lấy giá thế giới làm mốc
    world = df[df["Mã vàng"].str.upper() == "XAUUSD"][["timestamp_vn", "Giá mua"]].rename(columns={"Giá mua": "world_price_vnd"})
    df = df.merge(world, on="timestamp_vn", how="left")

    is_world = df["Mã vàng"].str.upper() == "XAUUSD"
    df["diff_vn_tg"] = np.where(~is_world & df["world_price_vnd"].notna(), df["Giá bán"] - df["world_price_vnd"], np.nan)
    df["diff_pct"] = np.where(df["diff_vn_tg"].notna() & (df["world_price_vnd"] > 0), (df["diff_vn_tg"] / df["world_price_vnd"]) * 100, np.nan)

    # Build Output theo đúng thứ tự trong ảnh bạn gửi
    out = pd.DataFrame({
        "Ngày": df["Ngày"],
        "Thời điểm cập nhật giá mới": df["Thời điểm cập nhật giá mới"],
        "Mã vàng": df["Mã vàng"],
        "Loại vàng": df["Loại vàng"],
        "Giá mua": df["Giá mua"],
        "Giá bán": df["Giá bán"],
        "Tỷ giá USD/VND": fx_usd_vnd,
        "Chênh lệch giá VN – TG": df["diff_vn_tg"],
        "Tỷ lệ chênh lệch (%)": df["diff_pct"],
        "Chênh lệch giá mua": df["diff_buy"],
        "Chênh lệch giá bán": df["diff_sell"]
    })

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {OUT_CLEAN}")

if __name__ == "__main__":
    main()
