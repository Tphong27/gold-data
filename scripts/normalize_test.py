# scripts/normalize_test.py
# Normalize + clean gold_live.csv -> gold_clean.csv
# Run:  python scripts/normalize_test.py

import os
import re
import pandas as pd

LIVE_CSV = "data/gold_live.csv"
OUT_CLEAN = "data/gold_clean.csv"
TZ_VN = "Asia/Ho_Chi_Minh"

TEXT_COLS = ["Ngày", "Thời điểm", "Mã vàng", "Loại vàng", "Currency"]
NUM_COLS = ["Giá mua", "Giá bán", "Day change buy", "Day change sell", "Số lần update"]


def to_number(x):
    """
    Convert strings like:
      - 17,130,000
      - 17.130.000
      - 4985.1
      - 4,985.10
      - 1.234,56
    into float safely (without breaking decimals like 4985.1 -> 49851).
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip().replace(" ", "")
    if s == "":
        return pd.NA

    # keep digits and basic separators
    s = re.sub(r"[^0-9,.\-]", "", s)

    if "," in s and "." in s:
        # decide decimal separator by the last occurring symbol
        if s.rfind(",") > s.rfind("."):
            # 1.234,56  -> 1234.56
            s = s.replace(".", "").replace(",", ".")
        else:
            # 1,234.56  -> 1234.56
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) in (1, 2):
            # 123,4 or 123,45 -> decimal comma
            s = s.replace(".", "").replace(",", ".")
        else:
            # 17,130,000 -> thousands commas
            s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        if len(parts[-1]) in (1, 2):
            # 4985.1 or 4985.12 -> decimal dot
            pass
        else:
            # 17.130.000 -> thousands dots
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
    # build naive datetime first
    df["timestamp_vn"] = pd.to_datetime(df["Ngày"] + " " + df["Thời điểm"], errors="coerce")

    # localize/convert to VN timezone
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

    # optional: keep updates as integer-like
    if "Số lần update" in df.columns:
        df["Số lần update"] = df["Số lần update"].astype("Int64")


def add_features(df: pd.DataFrame):
    # spread & mid (only meaningful when both buy/sell exist)
    if "Giá mua" in df.columns and "Giá bán" in df.columns:
        df["spread"] = df["Giá bán"] - df["Giá mua"]
        df["mid"] = (df["Giá mua"] + df["Giá bán"]) / 2

        # deltas by gold code
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
    # dedup by (timestamp + code)
    df["__key"] = df["timestamp_vn"].astype(str) + "|" + df["Mã vàng"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df)
    return df, before, after


def sanity_checks(df: pd.DataFrame):
    # show XAUUSD sample to ensure decimals aren't broken
    if "Mã vàng" in df.columns:
        mask = df["Mã vàng"].eq("XAUUSD")
        if mask.any():
            cols = [c for c in ["timestamp_vn", "Giá mua", "Giá bán", "Currency"] if c in df.columns]
            print("XAUUSD sample (tail):")
            print(df.loc[mask, cols].tail(5).to_string(index=False))


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

    # 4) basic features + deltas
    df = add_features(df)

    # 5) dedup & sort
    df, before, after = dedup(df)
    df = df.sort_values(["Mã vàng", "timestamp_vn"])

    # 6) save
    os.makedirs(os.path.dirname(OUT_CLEAN) or ".", exist_ok=True)
    df.to_csv(OUT_CLEAN, index=False)

    # logs
    print(f"✅ Saved: {OUT_CLEAN}")
    print(f"dedup: {before} -> {after}")
    print("null timestamp:", int(df["timestamp_vn"].isna().sum()))
    print("time range:", df["timestamp_vn"].min(), "->", df["timestamp_vn"].max())

    sanity_checks(df)


if __name__ == "__main__":
    main()
