import os
import re
import pandas as pd

SHEET_ID = "12IidFzGCo4yzUN77SqUTiUsF4qLp7RtAMSUR35IhCKs"
INPUT_XLSX = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"

OUT_TABLE_NEW = "data/gold_live_new.csv"
OUT_RAW_NEW = "data/gold_live_raw_new.csv"

SHEET_TABLE = "GOLD_PRICE"
SHEET_RAW = "RAW_DATA"

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


def fetch_from_sheets(url: str):
    """Đọc dữ liệu từ Google Sheets qua link export XLSX"""
    try:
        df_price = pd.read_excel(url, sheet_name=SHEET_TABLE)
        df_raw = pd.read_excel(url, sheet_name=SHEET_RAW)
        return df_price, df_raw
    except Exception as e:
        raise RuntimeError(
            "Không thể đọc Google Sheet. Hãy đảm bảo sheet đã share public (Anyone with the link can view). "
            f"Lỗi: {e}"
        )


def normalize_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột GOLD_PRICE:
    - strip khoảng trắng
    - sửa trường hợp copy/paste bị dính chữ: 'Day change buyDay change sellCurrency'
      (đôi khi sheet export bị lỗi header)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Fix một số trường hợp header bị dính (nếu có)
    joined = "".join(df.columns)
    if "Day change buyDay change sellCurrency" in joined:
        # Trường hợp xấu: export ra 1 cột dính 3 tên -> rất hiếm.
        # Nếu gặp, bạn cần chỉnh lại sheet header. Ở đây chỉ cảnh báo.
        print("⚠️ Cảnh báo: Header có dấu hiệu bị dính 'Day change buyDay change sellCurrency'. Hãy kiểm tra lại sheet.")

    # Fix trường hợp tên cột thiếu khoảng trắng kiểu "Day change buyDay change sell"
    # (nếu excel đọc ra đúng 2 cột thì không cần)
    rename_map = {}
    for c in df.columns:
        if c.replace(" ", "") == "Daychangebuy":
            rename_map[c] = "Day change buy"
        if c.replace(" ", "") == "Daychangesell":
            rename_map[c] = "Day change sell"
        if c.replace(" ", "") == "Solanupdate":
            rename_map[c] = "Số lần update"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def align_table_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ép GOLD_PRICE về đúng schema/đúng thứ tự cột"""
    df = df.copy()

    # đảm bảo đủ cột
    for col in HEADERS_TABLE:
        if col not in df.columns:
            df[col] = pd.NA

    # chỉ lấy đúng cột cần theo thứ tự
    df = df[HEADERS_TABLE].copy()

    # strip text cho các cột text
    for c in ["Ngày", "Thời điểm cập nhật giá mới", "Thời điểm cập nhật dữ liệu", "Mã vàng", "Loại vàng", "Currency"]:
        df[c] = df[c].astype("string").str.strip()

    return df


def _read_existing_lines(path: str) -> set[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.rstrip("\n") for line in f if line.strip())


RAW_TS_RE = re.compile(r"^\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s*(\{.*\})\s*$")


def normalize_raw_line(line: str) -> str | None:
    """
    RAW mẫu: 25/01/2026 14:46:44{"success":true,...}
    -> chuẩn hóa thành: 25/01/2026 14:46:44 {...json...}
    """
    if not line:
        return None
    s = str(line).strip()
    if not s or s.lower() == "nan":
        return None

    m = RAW_TS_RE.match(s)
    if m:
        ts, js = m.group(1), m.group(2)
        return f"{ts} {js}"

    # fallback: nếu không match regex, vẫn ghi nguyên dòng để khỏi mất dữ liệu
    return s


def process_raw_to_json_log(df_raw: pd.DataFrame):
    """
    RAW_DATA của bạn là 1 cột chứa cả datetime + json.
    Ghi ra OUT_RAW_NEW dạng 1 dòng / snapshot, có dedup để không ghi trùng vô hạn.
    """
    os.makedirs(os.path.dirname(OUT_RAW_NEW), exist_ok=True)

    existing = _read_existing_lines(OUT_RAW_NEW)
    new_lines = []

    # Lấy từng row, ưu tiên cell đầu tiên (cột A)
    for _, row in df_raw.iterrows():
        raw_cell = row.iloc[0] if len(row) > 0 else None
        line = normalize_raw_line(raw_cell)
        if not line or line in existing:
            continue
        new_lines.append(line)

    if not new_lines:
        print("ℹ️ RAW log: không có dòng mới để ghi.")
        return

    with open(OUT_RAW_NEW, "a", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    print(f"✅ RAW log: đã ghi thêm {len(new_lines)} dòng vào {OUT_RAW_NEW}")


def save_dedup_table(df_new: pd.DataFrame):
    """
    Lưu GOLD_PRICE dạng bảng và khử trùng theo:
    Ngày + Thời điểm cập nhật giá mới + Mã vàng
    """
    os.makedirs(os.path.dirname(OUT_TABLE_NEW), exist_ok=True)

    df_new = align_table_schema(df_new)

    if (not os.path.exists(OUT_TABLE_NEW)) or (os.path.getsize(OUT_TABLE_NEW) == 0):
        df_new.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
        print(f"✅ Đã tạo bảng: {OUT_TABLE_NEW} rows={len(df_new)}")
        return

    df_old = pd.read_csv(OUT_TABLE_NEW, encoding="utf-8-sig", dtype="string")
    df_all = pd.concat([df_old, df_new.astype("string")], ignore_index=True)

    key_col = "Thời điểm cập nhật giá mới"
    df_all["__key"] = (
        df_all["Ngày"].astype(str) + "|" +
        df_all[key_col].astype(str) + "|" +
        df_all["Mã vàng"].astype(str)
    )

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])
    after = len(df_all)

    df_all = df_all.sort_values(["Ngày", key_col, "Mã vàng"])
    df_all.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")

    print(f"✅ Đã cập nhật bảng: {OUT_TABLE_NEW} rows={after} (dedup {before}->{after})")


if __name__ == "__main__":
    print("🔄 Đang tải dữ liệu từ Google Sheet...")
    price_data, raw_data = fetch_from_sheets(INPUT_XLSX)

    # chuẩn hóa header GOLD_PRICE
    price_data = normalize_table_columns(price_data)

    # ghi raw + bảng
    process_raw_to_json_log(raw_data)
    save_dedup_table(price_data)
