import os
import json
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

API_URL = "https://www.vang.today/api/prices"

OUT_TABLE = "data/gold_live.csv"        # bảng (như hiện tại)
OUT_RAW = "data/gold_live_raw.csv"      # raw log (1 dòng / 1 snapshot)

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
TIME_FMT = "%d/%m/%Y %H:%M:%S"          # 25/01/2026 14:46:44

HEADERS = [
    "Ngày", "Thời điểm", "Mã vàng", "Loại vàng",
    "Giá mua", "Giá bán", "Day change buy", "Day change sell",
    "Currency", "Số lần update"
]

def currency_of(code: str) -> str:
    return "USD" if str(code) == "XAUUSD" else "VND"

def fetch_current():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def append_raw_snapshot(payload: dict):
    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)

    now_str = datetime.now(TZ).strftime(TIME_FMT)
    payload_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    # Optional dedup nhanh theo payload.timestamp (nếu API trả y hệt)
    ts = payload.get("timestamp")
    if ts is not None and os.path.exists(OUT_RAW) and os.path.getsize(OUT_RAW) > 0:
        try:
            with open(OUT_RAW, "rb") as f:
                f.seek(-min(4096, os.path.getsize(OUT_RAW)), os.SEEK_END)
                tail = f.read().decode("utf-8", errors="ignore").splitlines()
                last = tail[-1] if tail else ""
                if last and f'"timestamp":{ts}' in last:
                    print("ℹ️ Same payload timestamp as last line -> skip raw append")
                    return
        except Exception:
            pass

    # đúng format bạn muốn: datetime + space + json
    with open(OUT_RAW, "a", encoding="utf-8") as f:
        f.write(f"{now_str} {payload_str}\n")

    print(f"✅ Appended raw snapshot: {OUT_RAW}")

def normalize_to_table(payload: dict) -> pd.DataFrame:
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
                item.get("currency") or currency_of(code),
                item.get("updates"),
            ])
    else:
        # Case B: payload.prices dict keyed by code
        prices = payload.get("prices") or {}
        if not prices and isinstance(payload.get("data"), dict):
            prices = payload["data"]

        for code, p in (prices or {}).items():
            if not isinstance(p, dict):
                continue
            rows.append([
                date_str, time_str, code, p.get("name"),
                p.get("buy"), p.get("sell"),
                p.get("day_change_buy") or p.get("change_buy"),
                p.get("day_change_sell") or p.get("change_sell"),
                p.get("currency") or currency_of(code),
                p.get("updates"),
            ])

    return pd.DataFrame(rows, columns=HEADERS)

def append_dedup_table(df_new: pd.DataFrame):
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

    if (not os.path.exists(OUT_TABLE)) or (os.path.getsize(OUT_TABLE) == 0):
        df_new.to_csv(OUT_TABLE, index=False)
        print(f"✅ Created table: {OUT_TABLE} rows={len(df_new)}")
        return

    try:
        df_old = pd.read_csv(OUT_TABLE)
    except Exception as e:
        df_new.to_csv(OUT_TABLE, index=False)
        print(f"⚠️ Recreated table due to read error ({e}): {OUT_TABLE} rows={len(df_new)}")
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
    df_all.to_csv(OUT_TABLE, index=False)
    print(f"✅ Updated table: {OUT_TABLE} rows={len(df_all)}")

if __name__ == "__main__":
    payload = fetch_current()

    # 1) raw log
    append_raw_snapshot(payload)

    # 2) bảng
    df_new = normalize_to_table(payload)
    append_dedup_table(df_new)
