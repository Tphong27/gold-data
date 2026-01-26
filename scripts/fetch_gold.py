# scripts/fetch_gold.py
# Fetch gold data from Google Sheets (XLSX export) and save:
# - data/gold_live_new.csv (table)
# - data/gold_live_raw_new.log (timestamp + JSON, 1 line / snapshot, dedup by timestamp)

import os
import re
import pandas as pd

SHEET_ID = "12IidFzGCo4yzUN77SqUTiUsF4qLp7RtAMSUR35IhCKs"
INPUT_XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"

OUT_TABLE_NEW = "data/gold_live_new.csv"
OUT_RAW_LOG = "data/gold_live_raw_new.log"

SHEET_PRICE = "GOLD_PRICE"
SHEET_RAW = "RAW_DATA"

# Headers expected in GOLD_PRICE
HEADERS_TABLE = [
    "Ngày",
    "Thời điểm cập nhật giá mới",
    "Thời điểm cập nhật dữ liệu",
    "Mã vàng",
    "Loại vàng",
    "Giá mua",
    "Giá bán",
    "Day change buy",
    "Day change sell",
    "Currency",
    "Số lần update",
]


def fetch_from_sheets(xlsx_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read GOLD_PRICE and RAW_DATA from Google Sheets export XLSX.
    Requirements: the sheet must be shared as "Anyone with the link can view".
    """
    try:
        df_price = pd.read_excel(xlsx_url, sheet_name=SHEET_PRICE)
        # Read RAW_DATA as string to avoid datetime/JSON coercion
        df_raw = pd.read_excel(xlsx_url, sheet_name=SHEET_RAW, dtype=str)
        return df_price, df_raw
    except Exception as e:
        raise RuntimeError(
            "Không thể đọc Google Sheet. Hãy đảm bảo sheet đã bật "
            "'Bất kỳ ai có liên kết đều có thể xem'. "
            f"Lỗi: {e}"
        )


def _ensure_headers_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure GOLD_PRICE has all required columns and in correct order.
    Missing columns will be created as empty.
    """
    df = df.copy()
    for col in HEADERS_TABLE:
        if col not in df.columns:
            df[col] = ""
    return df[HEADERS_TABLE]


def save_dedup_table(df_new: pd.DataFrame) -> None:
    """
    Save GOLD_PRICE to OUT_TABLE_NEW and deduplicate by:
    Ngày | Thời điểm cập nhật giá mới | Mã vàng
    """
    os.makedirs(os.path.dirname(OUT_TABLE_NEW), exist_ok=True)

    df_new = _ensure_headers_table(df_new)

    if (not os.path.exists(OUT_TABLE_NEW)) or (os.path.getsize(OUT_TABLE_NEW) == 0):
        df_new.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
        print(f"✅ Created table: {OUT_TABLE_NEW} rows={len(df_new)}")
        return

    try:
        df_old = pd.read_csv(OUT_TABLE_NEW, dtype=str, keep_default_na=False)
    except Exception as e:
        df_new.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
        print(f"⚠️ Recreated table due to read error ({e}): {OUT_TABLE_NEW} rows={len(df_new)}")
        return

    df_old = _ensure_headers_table(df_old)
    df_new = df_new.astype(str)

    df_all = pd.concat([df_old, df_new], ignore_index=True)

    key_col = "Thời điểm cập nhật giá mới"
    df_all["__key"] = (
        df_all["Ngày"].astype(str).str.strip() + "|" +
        df_all[key_col].astype(str).str.strip() + "|" +
        df_all["Mã vàng"].astype(str).str.strip()
    )
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df_all)

    df_all = df_all.sort_values(["Ngày", key_col, "Mã vàng"])
    df_all.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
    print(f"✅ Updated table: {OUT_TABLE_NEW} rows={after} (dedup {before}->{after})")


_TS_JSON_RE = re.compile(r"^\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s*(\{.*)$")


def _split_raw_line(ts_or_line: str, maybe_json: str | None) -> tuple[str | None, str | None]:
    """
    Normalize RAW_DATA record into (timestamp_str, json_str).
    Supports:
      - 2 columns: (timestamp, json)
      - 1 column: "dd/MM/yyyy HH:mm:ss{...}" or "dd/MM/yyyy HH:mm:ss {...}"
    """
    if ts_or_line is None:
        return None, None

    a = str(ts_or_line).strip()
    b = None if maybe_json is None else str(maybe_json).strip()

    if a.lower() == "nan" or a == "":
        return None, None

    # If we already have JSON in column B
    if b and b.lower() != "nan":
        # Ensure b starts at first '{'
        if "{" in b:
            b = b[b.find("{"):]
        return a, b

    # Otherwise parse from single combined line in column A
    m = _TS_JSON_RE.match(a)
    if m:
        ts = m.group(1).strip()
        js = m.group(2).strip()
        return ts, js

    # As a fallback: if it's just timestamp without json -> ignore
    if a.startswith("{") and a.endswith("}"):
        # no timestamp found
        return None, a

    return None, None


def append_dedup_raw_log(df_raw: pd.DataFrame) -> None:
    """
    Append RAW_DATA to OUT_RAW_LOG in format:
      dd/MM/yyyy HH:mm:ss <json>
    Dedup by timestamp to avoid file growing too fast.
    """
    os.makedirs(os.path.dirname(OUT_RAW_LOG), exist_ok=True)

    # Build set of existing timestamps
    existing_ts: set[str] = set()
    if os.path.exists(OUT_RAW_LOG) and os.path.getsize(OUT_RAW_LOG) > 0:
        try:
            with open(OUT_RAW_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # timestamp is first 19 chars: dd/MM/yyyy HH:mm:ss
                    ts = line[:19].strip()
                    if ts:
                        existing_ts.add(ts)
        except Exception as e:
            # If file can't be read, do not block pipeline—just continue appending
            print(f"⚠️ Could not read existing raw log for dedup ({e}). Will append anyway.")
            existing_ts = set()

    wrote = 0
    skipped = 0

    with open(OUT_RAW_LOG, "a", encoding="utf-8") as f:
        for _, row in df_raw.iterrows():
            col_a = row.iloc[0] if len(row) > 0 else None
            col_b = row.iloc[1] if len(row) > 1 else None

            ts, js = _split_raw_line(col_a, col_b)
            if not ts or not js:
                continue

            # normalize json to start with '{'
            if "{" in js:
                js = js[js.find("{"):]
            js = js.strip()

            # dedup by timestamp
            if ts in existing_ts:
                skipped += 1
                continue

            f.write(f"{ts} {js}\n")
            existing_ts.add(ts)
            wrote += 1

    print(f"✅ Raw log updated: {OUT_RAW_LOG} wrote={wrote} skipped_existing={skipped}")


def main():
    print("🔄 Đang tải dữ liệu từ Google Sheet (XLSX export)...")
    df_price, df_raw = fetch_from_sheets(INPUT_XLSX_URL)

    # 1) RAW log: timestamp + JSON (dedup by timestamp)
    append_dedup_raw_log(df_raw)

    # 2) Table: GOLD_PRICE -> CSV (dedup by Ngày|Thời điểm cập nhật giá mới|Mã vàng)
    save_dedup_table(df_price)


if __name__ == "__main__":
    main()
