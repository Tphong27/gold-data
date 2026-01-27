# scripts/normalize_test.py
# -----------------------------------------------------------------------------
# MỤC TIÊU
# - Đọc dữ liệu bảng "thô" đã fetch từ Google Sheets: data/gold_live_new.csv
# - Chuẩn hoá + tính thêm các cột để xuất ra: data/gold_clean_new.csv
#
# OUTPUT (11 cột theo template trong Google Sheet):
#   1) Ngày
#   2) Thời điểm cập nhật giá mới
#   3) Mã vàng
#   4) Loại vàng
#   5) Giá mua (VND/lượng)
#   6) Giá bán (VND/lượng)
#   7) Tỷ giá USD/VND
#   8) Chênh lệch giá VN – TG
#   9) Tỷ lệ chênh lệch (%)
#   10) Chênh lệch giá mua (VND/lượng)   (diff theo thời gian cho từng mã)
#   11) Chênh lệch giá bán (VND/lượng)   (diff theo thời gian cho từng mã)
#
# QUY ƯỚC QUAN TRỌNG
# - Dữ liệu XAUUSD từ API là USD/oz (troy ounce).
# - Ta convert XAUUSD sang VND/lượng bằng:
#       (USD/oz) * (USD->VND fx) * (oz_to_luong)
# - Sau khi convert, toàn bộ "Giá mua/Giá bán" đều cùng đơn vị: VND/lượng
#
# LƯU Ý
# - Tỷ lệ chênh lệch (%) có thể trống nếu world_sell_vnd = 0 hoặc không có dữ liệu TG
#   vì code đang tránh chia cho 0 (điều kiện world_sell_vnd != 0).
# -----------------------------------------------------------------------------

import os
import re
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ========== INPUT / OUTPUT ==========
IN_CSV = "data/gold_live_new.csv"          # dữ liệu bảng lấy từ sheet GOLD_PRICE
OUT_CLEAN = "data/gold_clean_new.csv"      # file clean output theo template
FX_CACHE_JSON = "data/fx_cache.json"       # cache tỷ giá để fallback khi FX API lỗi
TZ_VN = "Asia/Ho_Chi_Minh"

# ========== COLUMN SETS ==========
# TEXT_COLS: các cột nên strip/trim (xóa khoảng trắng đầu/cuối)
TEXT_COLS = ["Ngày", "Thời điểm cập nhật giá mới", "Mã vàng", "Loại vàng", "Currency"]

# NUM_COLS: các cột cần parse về dạng số (float)
# (trong gold_live_new.csv có thể là string, có dấu phẩy, dấu chấm...)
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell", "Số lần update"]

# ========== UNIT CONVERSION: oz -> lượng ==========
# 1 troy ounce = 31.1034768 g
GRAMS_PER_OZ_TROY = 31.1034768

# 1 lượng (VN) = 37.5 g
GRAMS_PER_LUONG = 37.5

# hệ số đổi từ oz -> lượng:
#   (37.5 g / 31.1034768 g) ~ 1.205653 lượng/oz
OZ_TO_LUONG = GRAMS_PER_LUONG / GRAMS_PER_OZ_TROY  # ~1.205653

# ========== FX API ==========
# API lấy tỷ giá USD->VND theo mid-market (không cần API key)
FX_API_URL = os.getenv("FX_API_URL", "https://open.er-api.com/v6/latest/USD")
FX_TIMEOUT_SEC = int(os.getenv("FX_TIMEOUT_SEC", "20"))


def to_number(x):
    """
    Parse chuỗi về số float theo kiểu "robust":
    - Có thể gặp format: 1.234.567 hoặc 1,234.56 hoặc ' 171300000 '
    - Xử lý dấu phân tách nghìn/dấu thập phân dựa trên ngữ cảnh
    - Nếu không parse được -> np.nan
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().replace(" ", "")
    if s == "":
        return np.nan

    # giữ lại chỉ số/-,/,.
    s = re.sub(r"[^0-9,.\-]", "", s)

    # Nếu vừa có ',' vừa có '.' -> quyết định dấu thập phân là dấu xuất hiện sau cùng
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # 1.234,56 -> 1234.56
            s = s.replace(".", "").replace(",", ".")
        else:
            # 1,234.56 -> 1234.56
            s = s.replace(",", "")

    # Chỉ có ',' -> có thể là decimal hoặc thousand separator
    elif "," in s:
        parts = s.split(",")
        # nếu phần sau dấu ',' dài 1-2 -> coi là decimal
        if len(parts[-1]) in (1, 2):
            s = s.replace(".", "").replace(",", ".")
        else:
            # ngược lại coi ',' là thousand sep
            s = s.replace(",", "")

    # Chỉ có '.' -> có thể là decimal hoặc thousand separator
    elif "." in s:
        parts = s.split(".")
        # nếu phần sau dấu '.' không phải 1-2 -> coi '.' là thousand sep
        if len(parts[-1]) not in (1, 2):
            s = s.replace(".", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def ensure_text(df: pd.DataFrame):
    """
    Strip các cột text để tránh lỗi dedup/join do dư khoảng trắng.
    """
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()


def parse_timestamp_vn(df: pd.DataFrame):
    """
    Tạo cột timestamp chuẩn (timezone VN) để:
    - dedup theo snapshot (timestamp + code)
    - sort theo thời gian để tính chênh lệch (diff)
    Input:
      - Ngày (YYYY-MM-DD)
      - Thời điểm cập nhật giá mới (HH:MM:SS)
    Output:
      - timestamp_vn (datetime64 tz-aware)
    """
    df["timestamp_vn"] = pd.to_datetime(
        df["Ngày"].astype(str) + " " + df["Thời điểm cập nhật giá mới"].astype(str),
        errors="coerce",
    )

    # nếu pandas chưa có tz -> localize sang Asia/Ho_Chi_Minh
    if getattr(df["timestamp_vn"].dt, "tz", None) is None:
        df["timestamp_vn"] = df["timestamp_vn"].dt.tz_localize(
            TZ_VN, nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        # nếu đã có tz thì convert về VN
        df["timestamp_vn"] = df["timestamp_vn"].dt.tz_convert(TZ_VN)


def ensure_numeric(df: pd.DataFrame):
    """
    Convert các cột số sang float bằng to_number().
    """
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].apply(to_number)


def _load_fx_cache():
    """
    Load cache tỷ giá từ file data/fx_cache.json.
    Cache dùng khi FX API bị lỗi hoặc bị rate limit.
    """
    if not os.path.exists(FX_CACHE_JSON):
        return None
    try:
        with open(FX_CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_fx_cache(rate: float):
    """
    Save tỷ giá USD/VND + metadata vào cache.
    """
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
    Lấy tỷ giá USD -> VND (mid-market).

    Fallback thứ tự:
    1) Nếu có env FX_VND_PER_USD -> dùng luôn (ổn định cho CI)
    2) Gọi FX API (open.er-api.com)
    3) Nếu FX API fail -> dùng cache fx_cache.json (nếu có)
    4) Nếu không có cache -> raise lỗi
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

        # Schema thường gặp: conversion_rates.VND
        conv = data.get("conversion_rates") if isinstance(data, dict) else None
        if isinstance(conv, dict) and "VND" in conv:
            rate = float(conv["VND"])

        # fallback schema: rates.VND
        if rate is None:
            rates = data.get("rates") if isinstance(data, dict) else None
            if isinstance(rates, dict) and "VND" in rates:
                rate = float(rates["VND"])

        if rate is None:
            raise ValueError("FX response missing VND rate")

        # save cache để dùng lại cho lần sau
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
    Convert các dòng có Mã vàng == 'XAUUSD' từ USD/oz -> VND/lượng.

    Sau bước này:
    - Giá mua / Giá bán của XAUUSD sẽ trở thành VND/lượng (giống các mã VN)
    - df có thêm cột fx_usd_vnd (áp dụng cho tất cả dòng để sheet hiển thị)
    - Currency của XAUUSD được set về VND (vì giá đã quy đổi)
    """
    df = df.copy()

    # mask xác định dòng "giá thế giới"
    mask = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")

    # Ép kiểu số trước để tránh lỗi pandas khi gán (lossy setitem)
    df["Giá mua"] = pd.to_numeric(df["Giá mua"], errors="coerce").astype("float64")
    df["Giá bán"] = pd.to_numeric(df["Giá bán"], errors="coerce").astype("float64")

    # Ghi tỷ giá cho tất cả dòng (để output có cột Tỷ giá USD/VND)
    df["fx_usd_vnd"] = float(fx_usd_vnd)

    # Nếu có XAUUSD thì convert
    if mask.any():
        buy_usd = df.loc[mask, "Giá mua"].to_numpy(dtype="float64")
        sell_usd = df.loc[mask, "Giá bán"].to_numpy(dtype="float64")

        # USD/oz -> VND/lượng
        df.loc[mask, "Giá mua"] = buy_usd * fx_usd_vnd * OZ_TO_LUONG
        df.loc[mask, "Giá bán"] = sell_usd * fx_usd_vnd * OZ_TO_LUONG

        # Sau khi quy đổi thì Currency của dòng XAUUSD coi như VND
        if "Currency" in df.columns:
            df.loc[mask, "Currency"] = "VND"

    return df


def dedup(df: pd.DataFrame):
    """
    Dedup dữ liệu theo snapshot:
      key = (timestamp_vn | Mã vàng)

    Lý do:
    - Mỗi lần workflow chạy có thể append thêm dữ liệu giống nhau
    - Dedupe giúp dữ liệu không bị lặp
    """
    df = df.copy()
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df)
    return df, before, after


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xây output theo template 11 cột.

    Các bước chính:
    1) Tính chênh lệch mua/bán theo thời gian cho từng Mã vàng (diff)
    2) Lấy giá TG (world) theo timestamp từ mã XAUUSD (đã là VND/lượng)
    3) Join giá TG vào toàn bộ df theo timestamp_vn
    4) Tính:
       - Chênh lệch giá VN – TG = Giá bán VN - Giá bán TG
       - Tỷ lệ chênh lệch (%) = (Chênh lệch / Giá bán TG) * 100
    5) Làm tròn 2 chữ số
    6) Sort cho dễ nhìn
    """
    df = df.copy()

    # --- (1) Tính diff mua/bán theo từng mã ---
    # sort theo mã + thời gian để diff đúng
    df = df.sort_values(["Mã vàng", "timestamp_vn"])
    df["diff_buy"] = df.groupby("Mã vàng")["Giá mua"].diff()
    df["diff_sell"] = df.groupby("Mã vàng")["Giá bán"].diff()

    # --- (2) Tạo bảng "world" lấy từ XAUUSD theo timestamp ---
    # Note: XAUUSD đã được convert sang VND/lượng ở bước trước
    world = (
        df[df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")][["timestamp_vn", "Giá bán", "Giá mua"]]
        .rename(columns={"Giá bán": "world_sell_vnd", "Giá mua": "world_buy_vnd"})
        .drop_duplicates(subset=["timestamp_vn"], keep="last")
    )

    # --- (3) Join world vào toàn bộ df theo timestamp_vn ---
    df = df.merge(world, on="timestamp_vn", how="left")

    is_world = df["Mã vàng"].astype("string").str.upper().eq("XAUUSD")

    # --- (4) Chênh lệch VN - TG ---
    # diff_vn_tg chỉ áp dụng cho mã VN (không phải XAUUSD) và khi có world price
    df["diff_vn_tg"] = np.where(
        (~is_world) & df["world_sell_vnd"].notna(),
        df["Giá bán"] - df["world_sell_vnd"],
        np.nan,
    )

    # --- (5) Tỷ lệ chênh lệch (%) ---
    # tránh chia cho 0: world_sell_vnd != 0
    df["diff_pct"] = np.where(
        (~is_world) & df["world_sell_vnd"].notna() & (df["world_sell_vnd"] != 0),
        (df["diff_vn_tg"] / df["world_sell_vnd"]) * 100.0,
        np.nan,
    )

    # --- (6) Build DataFrame output đúng thứ tự cột template ---
    out = pd.DataFrame(
        {
            "Ngày": df["Ngày"],
            "Thời điểm cập nhật giá mới": df["Thời điểm cập nhật giá mới"],
            "Mã vàng": df["Mã vàng"],
            "Loại vàng": df["Loại vàng"],
            "Giá mua (VND/lượng)": df["Giá mua"],
            "Giá bán (VND/lượng)": df["Giá bán"],
            "Tỷ giá USD/VND": df["fx_usd_vnd"],
            "Chênh lệch giá VN – TG": df["diff_vn_tg"],
            "Tỷ lệ chênh lệch (%)": df["diff_pct"],
            "Chênh lệch giá mua (VND/lượng)": df["diff_buy"],
            "Chênh lệch giá bán (VND/lượng)": df["diff_sell"],
        }
    )

    # --- (7) Làm tròn 2 chữ số thập phân theo yêu cầu ---
    round_cols = [
        "Giá mua (VND/lượng)",
        "Giá bán (VND/lượng)",
        "Chênh lệch giá VN – TG",
        "Chênh lệch giá mua (VND/lượng)",
        "Chênh lệch giá bán (VND/lượng)",
        "Tỷ lệ chênh lệch (%)",
    ]
    for c in round_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    # --- (8) Sort cuối theo thời gian + mã cho dễ xem ---
    out["_ts"] = pd.to_datetime(
        out["Ngày"].astype(str) + " " + out["Thời điểm cập nhật giá mới"].astype(str),
        errors="coerce",
    )
    out = out.sort_values(["_ts", "Mã vàng"]).drop(columns=["_ts"])

    return out


def main():
    """
    Luồng xử lý tổng:
    1) Read raw table CSV
    2) Clean text columns
    3) Parse timestamp VN
    4) Parse numeric
    5) Fetch FX USD/VND + convert XAUUSD USD/oz -> VND/lượng
    6) Dedup snapshot
    7) Build output 11 cột
    8) Save gold_clean_new.csv
    """
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    # dtype string để text columns không bị pandas chuyển sang dạng khác
    df = pd.read_csv(
        IN_CSV,
        dtype={"Mã vàng": "string", "Loại vàng": "string", "Currency": "string"},
    )

    print("Loaded:", df.shape)
    print("Columns:", list(df.columns))

    # 1) strip text columns
    ensure_text(df)

    # 2) parse timestamp (tz-aware)
    parse_timestamp_vn(df)

    # 3) numeric conversion
    ensure_numeric(df)

    # 4) FX + convert world gold rows
    fx_usd_vnd = fetch_usd_vnd_midmarket()
    print(f"FX USD/VND used (mid-market): {fx_usd_vnd}")

    df = convert_xauusd_to_vnd_luong(df, fx_usd_vnd)

    # 5) dedup
    df, before, after = dedup(df)

    # 6) build final output
    out = build_output(df)

    # 7) save
    os.makedirs(os.path.dirname(OUT_CLEAN) or ".", exist_ok=True)
    out.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")

    print(f"✅ Saved: {OUT_CLEAN}")
    print(f"dedup: {before} -> {after}")

    # thống kê timestamp lỗi (nếu có)
    null_ts = pd.to_datetime(
        out["Ngày"].astype(str) + " " + out["Thời điểm cập nhật giá mới"].astype(str),
        errors="coerce",
    ).isna().sum()
    print("null timestamp:", int(null_ts))


if __name__ == "__main__":
    main()
