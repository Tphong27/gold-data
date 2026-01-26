import os
import json
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# Cấu hình nguồn dữ liệu từ link Google Sheet bạn cung cấp
SHEET_ID = "12lidFzGCo4yzUN77SqUTiUsF4qLp7RtAMSUR35lhCKs"
INPUT_SHEET = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"

# File đầu ra mới
OUT_TABLE_NEW = "data/gold_live_new.csv"      
OUT_RAW_NEW = "data/gold_live_raw_new.csv"    

TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Các đầu mục chính xác theo sheet GOLD_PRICE của bạn
HEADERS_TABLE = [
    "Ngày", "Thời điểm cập nhật giá mới", "Thời điểm cập nhật dữ liệu", 
    "Mã vàng", "Loại vàng", "Giá mua", "Giá bán", 
    "Day change buy", "Day change sell", "Currency", "Số lần update"
]

def fetch_from_sheets(url):
    """Đọc dữ liệu từ Google Sheets qua link export XLSX"""
    try:
        # Đọc sheet GOLD_PRICE
        df_price = pd.read_excel(url, sheet_name='GOLD_PRICE')
        # Đọc sheet RAW_DATA
        df_raw = pd.read_excel(url, sheet_name='RAW_DATA')
        return df_price, df_raw
    except Exception as e:
        raise RuntimeError(f"Không thể đọc Google Sheet. Hãy đảm bảo sheet đã bật 'Bất kỳ ai có liên kết đều có thể xem'. Lỗi: {e}")

def process_raw_to_json_log(df_raw: pd.DataFrame):
    """Lưu dữ liệu RAW_DATA (datetime + JSON) vào file log"""
    os.makedirs(os.path.dirname(OUT_RAW_NEW), exist_ok=True)
    
    with open(OUT_RAW_NEW, "a", encoding="utf-8") as f:
        for _, row in df_raw.iterrows():
            # Đọc dòng đầu tiên của mỗi hàng (giả định nội dung nằm ở cột A)
            line = str(row.iloc[0]).strip()
            if line and line != "nan":
                f.write(f"{line}\n")
            
    print(f"✅ Đã ghi dữ liệu raw vào: {OUT_RAW_NEW}")

def save_dedup_table(df_new: pd.DataFrame):
    """Lưu dữ liệu bảng và khử trùng"""
    os.makedirs(os.path.dirname(OUT_TABLE_NEW), exist_ok=True)

    # Đảm bảo đủ cột và đúng thứ tự
    for col in HEADERS_TABLE:
        if col not in df_new.columns:
            df_new[col] = ""
    df_new = df_new[HEADERS_TABLE]

    if (not os.path.exists(OUT_TABLE_NEW)) or (os.path.getsize(OUT_TABLE_NEW) == 0):
        df_new.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
        print(f"✅ Đã tạo bảng: {OUT_TABLE_NEW}")
        return

    df_old = pd.read_csv(OUT_TABLE_NEW)
    df_all = pd.concat([df_old, df_new], ignore_index=True)

    # Khóa khử trùng: Ngày + Thời điểm cập nhật giá mới + Mã vàng
    key_col = "Thời điểm cập nhật giá mới"
    df_all["__key"] = (
        df_all["Ngày"].astype(str) + "|" +
        df_all[key_col].astype(str) + "|" +
        df_all["Mã vàng"].astype(str)
    )
    df_all = df_all.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])

    df_all.sort_values(["Ngày", key_col, "Mã vàng"], inplace=True)
    df_all.to_csv(OUT_TABLE_NEW, index=False, encoding="utf-8-sig")
    print(f"✅ Đã cập nhật bảng: {OUT_TABLE_NEW}")

if __name__ == "__main__":
    print(f"🔄 Đang tải dữ liệu từ Google Sheet...")
    price_data, raw_data = fetch_from_sheets(INPUT_SHEET)
    
    process_raw_to_json_log(raw_data)
    save_dedup_table(price_data)
