# scripts/fetch_gold.py
# -----------------------------------------------------------------------------
# MỤC TIÊU
# 1) Lấy dữ liệu vàng từ Google Sheets bằng cách tải file XLSX export (public link)
# 2) Xuất ra 2 file trong repo:
#    - data/gold_live_new.csv        : dữ liệu dạng bảng (sheet GOLD_PRICE)
#    - data/gold_live_raw_new.log    : log raw JSON (sheet RAW_DATA), 1 dòng / snapshot
#
# LƯU Ý QUAN TRỌNG
# - Google Sheet phải bật "Anyone with the link can view" (ai có link đều xem được)
# - Không dùng Google API/OAuth, chỉ tải XLSX public.
# - RAW log sẽ dedup theo TIMESTAMP để log không bị phình.
# - TABLE csv sẽ dedup theo (Ngày | Thời điểm cập nhật giá mới | Mã vàng)
# -----------------------------------------------------------------------------

import os
import re
import pandas as pd

# ========== CONFIG: Google Sheet ==========
SHEET_ID = "12IidFzGCo4yzUN77SqUTiUsF4qLp7RtAMSUR35IhCKs"

# Export XLSX trực tiếp từ Google Sheets (yêu cầu sheet public view)
INPUT_XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"

# ========== OUTPUT FILES ==========
OUT_TABLE_NEW = "data/gold_live_new.csv"       # bảng dữ liệu
OUT_RAW_LOG = "data/gold_live_raw_new.log"     # raw JSON log (1 dòng / snapshot)

# ========== SHEET NAMES ==========
SHEET_PRICE = "GOLD_PRICE"   # sheet chứa bảng giá dạng table
SHEET_RAW = "RAW_DATA"       # sheet chứa raw json log

# ========== HEADERS EXPECTED IN GOLD_PRICE ==========
# Script sẽ đảm bảo đủ các cột này, thiếu cột nào sẽ tạo cột rỗng.
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
    Đọc 2 sheet từ file XLSX export của Google Sheets:
      - GOLD_PRICE: dữ liệu dạng bảng
      - RAW_DATA  : dữ liệu raw (ép string để tránh pandas tự parse)
    """
    try:
        # GOLD_PRICE: đọc bình thường
        df_price = pd.read_excel(xlsx_url, sheet_name=SHEET_PRICE)

        # RAW_DATA: ép dtype=str để JSON/timestamp không bị pandas biến đổi
        df_raw = pd.read_excel(xlsx_url, sheet_name=SHEET_RAW, dtype=str)

        return df_price, df_raw

    except Exception as e:
        # Thường lỗi do:
        # - Sheet chưa public
        # - Sai tên sheet
        # - Link bị chặn/quá quyền
        raise RuntimeError(
            "Không thể đọc Google Sheet. Hãy đảm bảo sheet đã bật "
            "'Bất kỳ ai có liên kết đều có thể xem'. "
            f"Lỗi: {e}"
        )


def _ensure_headers_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo df có đầy đủ cột theo HEADERS_TABLE và đúng thứ tự.
    Nếu thiếu cột -> tạo cột rỗng.
    """
    df = df.copy()
    for col in HEADERS_TABLE:
        if col not in df.columns:
            df[col] = ""
    # Trả về đúng thứ tự cột để output consistent
    return df[HEADERS_TABLE]


def save_dedup_table(df_new: pd.DataFrame) -> None:
    """
    Lưu sheet GOLD_PRICE ra file OUT_TABLE_NEW (CSV) theo hướng "append + dedup".

    Dedup key = (Ngày | Thời điểm cập nhật giá mới | Mã vàng)
    -> đảm bảo mỗi snapshot cho 1 mã vàng không bị lặp.

    Luồng:
    - Nếu file chưa tồn tại -> tạo mới
    - Nếu file đã tồn tại -> đọc file cũ, concat + dedup + sort rồi ghi lại
    """
    os.makedirs(os.path.dirname(OUT_TABLE_NEW), exist_ok=True)

    # đảm bảo df_new đủ cột và thứ tự
    df_new = _ensure_headers_table(df_new)

    # Nếu file chưa tồn tại hoặc rỗng -> ghi luôn
    if (not os.path.exists(OUT_TABLE_NEW)) or (os.path.getsize(OUT_TABLE_NEW) == 0):
        df_new.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
        print(f"✅ Created table: {OUT_TABLE_NEW} rows={len(df_new)}")
        return

    # Nếu file đã có -> đọc để append/dedup
    try:
        # dtype=str để tránh pandas tự parse số/ngày làm thay đổi format
        df_old = pd.read_csv(OUT_TABLE_NEW, dtype=str, keep_default_na=False)
    except Exception as e:
        # Nếu file cũ bị lỗi đọc -> tạo lại bằng df_new
        df_new.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
        print(f"⚠️ Recreated table due to read error ({e}): {OUT_TABLE_NEW} rows={len(df_new)}")
        return

    # đảm bảo df_old cũng đúng header
    df_old = _ensure_headers_table(df_old)

    # ép df_new về string để key ghép không bị NaN/float
    df_new = df_new.astype(str)

    # concat cũ + mới
    df_all = pd.concat([df_old, df_new], ignore_index=True)

    # Dedup key: Ngày | Thời điểm cập nhật giá mới | Mã vàng
    key_col = "Thời điểm cập nhật giá mới"
    df_all["__key"] = (
        df_all["Ngày"].astype(str).str.strip() + "|" +
        df_all[key_col].astype(str).str.strip() + "|" +
        df_all["Mã vàng"].astype(str).str.strip()
    )

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df_all)

    # sort cho đẹp + ổn định
    df_all = df_all.sort_values(["Ngày", key_col, "Mã vàng"])

    # ghi lại file
    df_all.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
    print(f"✅ Updated table: {OUT_TABLE_NEW} rows={after} (dedup {before}->{after})")


# Regex để tách dạng "dd/MM/yyyy HH:mm:ss {json...}" hoặc "dd/MM/yyyy HH:mm:ss{json...}"
_TS_JSON_RE = re.compile(r"^\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s*(\{.*)$")


def _split_raw_line(ts_or_line: str, maybe_json: str | None) -> tuple[str | None, str | None]:
    """
    Chuẩn hoá 1 record raw thành (timestamp_str, json_str)

    Hỗ trợ 2 kiểu dữ liệu sheet RAW_DATA:
    1) 2 cột:
       - colA: "dd/MM/yyyy HH:mm:ss"
       - colB: "{...json...}"
    2) 1 cột gộp:
       - colA: "dd/MM/yyyy HH:mm:ss {...json...}"
       - hoặc "dd/MM/yyyy HH:mm:ss{...json...}"

    Trả về:
      - (ts, js) nếu parse được
      - (None, None) nếu không hợp lệ
    """
    if ts_or_line is None:
        return None, None

    a = str(ts_or_line).strip()
    b = None if maybe_json is None else str(maybe_json).strip()

    # bỏ qua dòng rỗng/NaN
    if a.lower() == "nan" or a == "":
        return None, None

    # Case 1: có JSON ở cột B
    if b and b.lower() != "nan":
        # đảm bảo json bắt đầu từ '{'
        if "{" in b:
            b = b[b.find("{"):]
        return a, b

    # Case 2: JSON dính liền trong cột A -> dùng regex tách
    m = _TS_JSON_RE.match(a)
    if m:
        ts = m.group(1).strip()
        js = m.group(2).strip()
        return ts, js

    # fallback: nếu chỉ có json mà không có timestamp -> bỏ qua (tuỳ bạn muốn giữ hay không)
    if a.startswith("{") and a.endswith("}"):
        return None, a

    return None, None


def append_dedup_raw_log(df_raw: pd.DataFrame) -> None:
    """
    Append RAW_DATA vào OUT_RAW_LOG theo format:
      dd/MM/yyyy HH:mm:ss <json>

    Dedup theo timestamp (19 ký tự đầu) để tránh file log phình khi workflow chạy lặp.
    """
    os.makedirs(os.path.dirname(OUT_RAW_LOG), exist_ok=True)

    # Tập timestamp đã tồn tại trong log (để dedup)
    existing_ts: set[str] = set()

    # Nếu log đã tồn tại -> đọc toàn bộ timestamp đầu dòng vào set
    if os.path.exists(OUT_RAW_LOG) and os.path.getsize(OUT_RAW_LOG) > 0:
        try:
            with open(OUT_RAW_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # timestamp ở đầu dòng: "dd/MM/yyyy HH:mm:ss" => 19 ký tự
                    ts = line[:19].strip()
                    if ts:
                        existing_ts.add(ts)
        except Exception as e:
            # Nếu đọc file log lỗi thì vẫn cho pipeline chạy, chỉ không dedup được file cũ
            print(f"⚠️ Could not read existing raw log for dedup ({e}). Will append anyway.")
            existing_ts = set()

    wrote = 0
    skipped = 0

    # Append mode
    with open(OUT_RAW_LOG, "a", encoding="utf-8") as f:
        for _, row in df_raw.iterrows():
            # Lấy 2 cột đầu của sheet RAW_DATA (nếu có)
            col_a = row.iloc[0] if len(row) > 0 else None
            col_b = row.iloc[1] if len(row) > 1 else None

            ts, js = _split_raw_line(col_a, col_b)
            if not ts or not js:
                # bỏ qua record không parse được
                continue

            # đảm bảo json bắt đầu từ '{'
            if "{" in js:
                js = js[js.find("{"):]
            js = js.strip()

            # dedup theo timestamp
            if ts in existing_ts:
                skipped += 1
                continue

            # ghi 1 dòng snapshot
            f.write(f"{ts} {js}\n")
            existing_ts.add(ts)
            wrote += 1

    print(f"✅ Raw log updated: {OUT_RAW_LOG} wrote={wrote} skipped_existing={skipped}")


def main():
    """
    Luồng tổng:
    1) Tải XLSX từ Google Sheets
    2) Update raw log (dedup timestamp)
    3) Update table csv (dedup snapshot+code)
    """
    print("🔄 Đang tải dữ liệu từ Google Sheet (XLSX export)...")
    df_price, df_raw = fetch_from_sheets(INPUT_XLSX_URL)

    # 1) RAW log: timestamp + JSON (dedup by timestamp)
    append_dedup_raw_log(df_raw)

    # 2) Table: GOLD_PRICE -> CSV (dedup by Ngày|Thời điểm cập nhật giá mới|Mã vàng)
    save_dedup_table(df_price)


if __name__ == "__main__":
    main()
