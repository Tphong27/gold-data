import os
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

API_URL = "https://www.vang.today/api/prices"
OUT_FILE = "data/gold_live.csv"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

HEADERS = [
    "Ngày", "Thời điểm", "Mã vàng", "Loại vàng",
    "Giá mua", "Giá bán", "Day change buy", "Day change sell",
    "Currency", "Số lần update"
]

def currency_of(code: str) -> str:
    return "USD" if code == "XAUUSD" else "VND"

def fetch_current():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def normalize(payload: dict) -> pd.DataFrame:
    now = datetime.now(TZ)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    rows = []

    # Case A: payload.data is list
    if isinstance(payload.get("data"), list):
        for item in payload["data"]:
            code = item.get("type_code") or item.get("code") or item.get("type")
            if not code:
                continue
            rows.append([
                date_str, time_str, code, item.get("name"),
                item.get("buy"), item.get("sell"),
                item.get("day_change_buy") or item.get("change_buy"),
                item.get("day_change_sell") or item.get("change_sell"),
                currency_of(code),
                item.get("updates"),
            ])
    else:
        # Case B: payload.prices is dict keyed by code (fallback)
        prices = payload.get("prices") or {}
        if not prices and isinstance(payload.get("data"), dict):
            prices = payload["data"]
        if not prices:
            maybe = {k: v for k, v in payload.items() if isinstance(v, dict) and ("buy" in v or "sell" in v)}
            prices = maybe

        for code, p in (prices or {}).items():
            if not isinstance(p, dict):
                continue
            rows.append([
                date_str, time_str, code, p.get("name"),
                p.get("buy"), p.get("sell"),
                p.get("day_change_buy"),
                p.get("day_change_sell"),
                currency_of(code),
                p.get("updates"),
            ])

    return pd.DataFrame(rows, columns=HEADERS)

def append_dedup(df_new: pd.DataFrame):
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Nếu file chưa tồn tại hoặc tồn tại nhưng rỗng -> ghi luôn df_new
    if (not os.path.exists(OUT_FILE)) or (os.path.getsize(OUT_FILE) == 0):
        df_new.to_csv(OUT_FILE, index=False)
        print(f"✅ Created new file: {OUT_FILE} rows={len(df_new)}")
        return

    # Nếu file có dữ liệu: đọc và append
    try:
        df_old = pd.read_csv(OUT_FILE)
    except Exception as e:
        # Nếu file lỗi/không đọc được -> tạo lại
        df_new.to_csv(OUT_FILE, index=False)
        print(f"⚠️ Recreated file due to read error ({e}): {OUT_FILE} rows={len(df_new)}")
        return

    df_all = pd.concat([df_old, df_new], ignore_index=True)

    # chống trùng theo snapshot: Ngày|Thời điểm|Mã vàng
    df_all["__key"] = (
        df_all["Ngày"].astype(str) + "|" +
        df_all["Thời điểm"].astype(str) + "|" +
        df_all["Mã vàng"].astype(str)
    )
    df_all = df_all.drop_duplicates(subset="__key", keep="last").drop(columns=["__key"])

    df_all = df_all.sort_values(["Ngày", "Thời điểm", "Mã vàng"])
    df_all.to_csv(OUT_FILE, index=False)
    print(f"✅ Updated file: {OUT_FILE} rows={len(df_all)}")

if __name__ == "__main__":
    payload = fetch_current()
    df_new = normalize(payload)
    append_dedup(df_new)

