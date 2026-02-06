#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DSS Gold - Webgia Interest Pipeline (v2.3, APPEND vào CHÍNH 2 FILE CSV)
========================================================================
- Crawl: https://webgia.com/lai-suat/
- Parse bảng lãi suất tiền gửi kỳ hạn 12 tháng
- Tính:
    interest_rate_state  = AVG(Big4)
    interest_rate_market = AVG(Top 3 trusted private) + fallback an toàn
- Lưu dữ liệu:
    1) webgia_laisuat_latest_clean.csv  (APPEND theo thời gian, KHÔNG ghi đè)
    2) macro_features_latest.csv        (APPEND theo thời gian, KHÔNG ghi đè)
- PNG debug theo từng lần chạy:
    history/screenshots/debug_webgia_YYYYMMDD_HHMMSS.png
"""

import re
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# =========================
# CONFIG
# =========================
URL = "https://webgia.com/lai-suat/"

# ===== CHỈ 2 FILE CSV CHÍNH (APPEND) =====
OUT_RATES_CSV = "webgia_laisuat_latest_clean.csv"
OUT_FEATURE_CSV = "macro_features_latest.csv"

# screenshot archive
OUT_SCREENSHOT_DIR = "history/screenshots"

BIG4 = {"Vietcombank", "BIDV", "Agribank", "VietinBank"}
TRUSTED_PRIVATE = {"Techcombank", "VPBank", "MB", "ACB", "Sacombank", "VIB", "HDBank"}

TARGET_TERM_MONTH = 12
TARGET_TERM_TEXT = "12 tháng"

HEADLESS = True
ENABLE_SCHEDULER = False
RUN_EVERY_MINUTES = 1

MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES_SEC = 3

PAGE_TIMEOUT_MS = 90000
RENDER_WAIT_MS = 8000

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("webgia_pipeline")


# =========================
# TIME UTILS
# =========================
def now_vn_dt() -> datetime:
    return datetime.now(VN_TZ)

def now_vn_str() -> str:
    return now_vn_dt().strftime("%Y-%m-%d %H:%M:%S")

def ts_file_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


# =========================
# TEXT UTILS
# =========================
def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())

def strip_accents_vn(s: str) -> str:
    repl = {
        "ă":"a","â":"a","đ":"d","ê":"e","ô":"o","ơ":"o","ư":"u",
        "á":"a","à":"a","ả":"a","ã":"a","ạ":"a",
        "ấ":"a","ầ":"a","ẩ":"a","ẫ":"a","ậ":"a",
        "ắ":"a","ằ":"a","ẳ":"a","ẵ":"a","ặ":"a",
        "é":"e","è":"e","ẻ":"e","ẽ":"e","ẹ":"e",
        "ế":"e","ề":"e","ể":"e","ễ":"e","ệ":"e",
        "í":"i","ì":"i","ỉ":"i","ĩ":"i","ị":"i",
        "ó":"o","ò":"o","ỏ":"o","õ":"o","ọ":"o",
        "ố":"o","ồ":"o","ổ":"o","ỗ":"o","ộ":"o",
        "ớ":"o","ờ":"o","ở":"o","ỡ":"o","ợ":"o",
        "ú":"u","ù":"u","ủ":"u","ũ":"u","ụ":"u",
        "ứ":"u","ừ":"u","ử":"u","ữ":"u","ự":"u",
        "ý":"y","ỳ":"y","ỷ":"y","ỹ":"y","ỵ":"y",
        "Ă":"a","Â":"a","Đ":"d","Ê":"e","Ô":"o","Ơ":"o","Ư":"u",
        "Á":"a","À":"a","Ả":"a","Ã":"a","Ạ":"a",
        "Ấ":"a","Ầ":"a","Ẩ":"a","Ẫ":"a","Ậ":"a",
        "Ắ":"a","Ằ":"a","Ẳ":"a","Ẵ":"a","Ặ":"a",
        "É":"e","È":"e","Ẻ":"e","Ẽ":"e","Ẹ":"e",
        "Ế":"e","Ề":"e","Ể":"e","Ễ":"e","Ệ":"e",
        "Í":"i","Ì":"i","Ỉ":"i","Ĩ":"i","Ị":"i",
        "Ó":"o","Ò":"o","Ỏ":"o","Õ":"o","Ọ":"o",
        "Ố":"o","Ồ":"o","Ổ":"o","Ỗ":"o","Ộ":"o",
        "Ớ":"o","Ờ":"o","Ở":"o","Ỡ":"o","Ợ":"o",
        "Ú":"u","Ù":"u","Ủ":"u","Ũ":"u","Ụ":"u",
        "Ứ":"u","Ừ":"u","Ử":"u","Ữ":"u","Ự":"u",
        "Ý":"y","Ỳ":"y","Ỷ":"y","Ỹ":"y","Ỵ":"y",
    }
    return "".join(repl.get(ch, ch) for ch in s)

def normalize_text_key(s: Any) -> str:
    return strip_accents_vn(norm_space(s)).lower()

def canonical_key(s: Any) -> str:
    t = normalize_text_key(s)
    t = re.sub(r"[^a-z0-9]", "", t)
    return t

def parse_rate(value: Any):
    s = norm_space(value).lower()
    if not s or s == "-" or "webgia" in s:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    v = float(m.group(0).replace(",", "."))
    if v <= 0 or v > 30:
        return None
    return v


def normalize_bank_name(name: Any) -> str:
    raw = norm_space(name)
    k = canonical_key(raw)

    alias = {
        # Big4
        "vietcombank": "Vietcombank", "nganhangvietcombank": "Vietcombank",
        "bidv": "BIDV", "nganhangbidv": "BIDV", "dautuvaphattrienvietnam": "BIDV",
        "agribank": "Agribank", "nganhangnongnghiepvaphattriennongthon": "Agribank",
        "vietinbank": "VietinBank", "vietinbankctg": "VietinBank", "nganhangcongthuong": "VietinBank",

        # Trusted private
        "techcombank": "Techcombank", "tcb": "Techcombank", "kythuong": "Techcombank",
        "nganhangkythuong": "Techcombank", "nganhangkythuongvietnam": "Techcombank",

        "vpbank": "VPBank", "vpb": "VPBank", "vietnamprosperity": "VPBank",
        "thinhvuong": "VPBank", "nganhangvietnamthinhvuong": "VPBank",

        "mb": "MB", "mbbank": "MB", "quandoi": "MB",
        "nganhangquandoi": "MB", "nganhangtmcppquandoi": "MB",

        "acb": "ACB", "achau": "ACB", "nganhangachau": "ACB",
        "nganhangthuongmaicophanachau": "ACB",

        "sacombank": "Sacombank", "stb": "Sacombank", "saigonthuongtin": "Sacombank",
        "nganhangsaigonthuongtin": "Sacombank",

        "vib": "VIB", "vibbank": "VIB", "quocte": "VIB",
        "nganhangquocte": "VIB", "nganhangquoctevietnam": "VIB",
        "vietnaminternationalbank": "VIB",

        "hdbank": "HDBank", "hdb": "HDBank", "hochiminhcitydevelopmentbank": "HDBank",
        "phattrientphcm": "HDBank", "nganhangphattrientphcm": "HDBank",
    }
    if k in alias:
        return alias[k]

    # Fuzzy fallback
    if "techcombank" in k or "kythuong" in k or k == "tcb":
        return "Techcombank"
    if "achau" in k or k == "acb":
        return "ACB"
    if "saigonthuongtin" in k or "sacombank" in k or k == "stb":
        return "Sacombank"
    if "quocte" in k or "vietnaminternationalbank" in k or k in {"vib", "vibbank"}:
        return "VIB"
    if "hdbank" in k or "phattrientphcm" in k or k == "hdb":
        return "HDBank"
    if "quandoi" in k or k in {"mb", "mbbank"}:
        return "MB"
    if "thinhvuong" in k or "vpbank" in k or k == "vpb":
        return "VPBank"

    if "vietcombank" in k:
        return "Vietcombank"
    if k == "bidv" or "dautuvaphattrien" in k:
        return "BIDV"
    if "agribank" in k or "nongnghiepvaphattriennongthon" in k:
        return "Agribank"
    if "vietinbank" in k or "congthuong" in k:
        return "VietinBank"

    return raw


def is_header_like_bank_text(s: str) -> bool:
    t = canonical_key(s)
    bad_tokens = ["nganhang", "kyhantietkiem", "kyhan", "laisuat", "khongkyhan", "thang", "jpy", "usd", "eur"]
    return any(tok in t for tok in bad_tokens)


TRUSTED_PRIVATE_KEY = {canonical_key(x) for x in TRUSTED_PRIVATE}


# =========================
# FS + CSV APPEND UTILS
# =========================
def ensure_output_dirs():
    Path(OUT_SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

def append_csv(df_new: pd.DataFrame, path: str, dedup_keys: List[str] = None):
    p = Path(path)
    if p.exists():
        df_old = pd.read_csv(p)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    if dedup_keys:
        keys = [k for k in dedup_keys if k in df_all.columns]
        if keys:
            df_all = df_all.drop_duplicates(subset=keys, keep="last")

    df_all.to_csv(path, index=False, encoding="utf-8-sig")


# =========================
# PAGE TIME PARSER
# =========================
def extract_webgia_updated_at(page) -> str:
    body_text = page.evaluate("() => document.body ? document.body.innerText : ''")
    text = strip_accents_vn(norm_space(body_text)).lower()

    patterns = [
        r"cap nhat(?: luc)?\s*(\d{1,2}:\d{2})(?::(\d{2}))?\s*(\d{1,2}/\d{1,2}/\d{4})",
        r"cap nhat[:\s]*?(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}:\d{2})(?::(\d{2}))?",
    ]
    for i, p in enumerate(patterns):
        m = re.search(p, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            if i == 0:
                hhmm, ss, dmy = m.group(1), (m.group(2) or "00"), m.group(3)
            else:
                dmy, hhmm, ss = m.group(1), m.group(2), (m.group(3) or "00")

            d, mo, y = dmy.split("/")
            h, mi = hhmm.split(":")
            dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(ss), tzinfo=VN_TZ)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return ""


# =========================
# DOM EXTRACTION
# =========================
def extract_table_rows_from_dom(page) -> List[List[str]]:
    return page.evaluate(
        """
        () => {
          const tables = Array.from(document.querySelectorAll("table"));
          if (!tables.length) return [];

          let best = null;
          let bestScore = -1e9;

          for (const t of tables) {
            const text = (t.innerText || "").toLowerCase();
            const trCount = t.querySelectorAll("tr").length;

            let score = 0;
            if (text.includes("ngân hàng") || text.includes("ngan hang")) score += 30;
            if (text.includes("12 tháng") || text.includes("12 thang")) score += 30;

            const terms = ["01 tháng","03 tháng","06 tháng","09 tháng","12 tháng","13 tháng","18 tháng","24 tháng","36 tháng"];
            for (const tm of terms) if (text.includes(tm)) score += 8;

            score += trCount;
            if (text.includes("jpy") || text.includes("usd") || text.includes("eur")) score -= 30;
            if (trCount < 8) score -= 20;

            if (score > bestScore) {
              bestScore = score;
              best = t;
            }
          }

          if (!best) return [];

          const trs = Array.from(best.querySelectorAll("tr"));
          return trs.map(tr => {
            const cells = Array.from(tr.querySelectorAll("th,td"))
              .map(td => (td.innerText || "").replace(/\\s+/g, " ").trim());
            return cells;
          }).filter(r => r.length > 0);
        }
        """
    )


def detect_header_and_term_col(rows: List[List[str]]) -> Tuple[int, int, Dict[int, int]]:
    header_row_idx = -1
    bank_col_idx = 0
    term_col_map: Dict[int, int] = {}

    for i, r in enumerate(rows[:12]):
        line = " ".join([normalize_text_key(c) for c in r])
        hits = sum(1 for t in ["01", "03", "06", "09", "12", "13", "18", "24", "36"] if re.search(rf"\b{t}\b", line))
        if hits >= 5:
            header_row_idx = i
            break

    if header_row_idx == -1:
        header_row_idx = 1 if len(rows) > 1 else 0

    for j, raw in enumerate(rows[header_row_idx]):
        t = normalize_text_key(raw)
        if "ngan hang" in t:
            bank_col_idx = j
        m = re.search(r"\b(\d{1,2})\s*(thang)?\b", t)
        if m:
            month = int(m.group(1))
            if month in [1, 3, 6, 9, 12, 13, 18, 24, 36]:
                term_col_map[month] = j

    return header_row_idx, bank_col_idx, term_col_map


def rows_to_rate12_dataframe(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ngan_hang", "rate12"])

    header_idx, bank_col, term_map = detect_header_and_term_col(rows)
    term12_col = term_map.get(TARGET_TERM_MONTH)
    if term12_col is None:
        raise RuntimeError(f"Không tìm thấy cột {TARGET_TERM_MONTH} tháng. term_map={term_map}")

    header_cells_norm = [normalize_text_key(x) for x in rows[header_idx]]
    header_has_bank = any("ngan hang" in x for x in header_cells_norm)
    if bank_col == 0 and not header_has_bank:
        term12_col += 1

    logger.info(f"header_idx={header_idx}, bank_col={bank_col}, term12_col={term12_col}, term_map={term_map}, header_has_bank={header_has_bank}")

    parsed = []
    for r in rows[header_idx + 1:]:
        if bank_col >= len(r) or term12_col >= len(r):
            continue
        bank = normalize_bank_name(r[bank_col])
        if not bank or len(bank) < 2 or is_header_like_bank_text(bank):
            continue
        rate12 = parse_rate(r[term12_col])
        if rate12 is None:
            continue
        parsed.append({"ngan_hang": bank, "rate12": rate12})

    if not parsed:
        return pd.DataFrame(columns=["ngan_hang", "rate12"])

    return pd.DataFrame(parsed).drop_duplicates(subset=["ngan_hang"]).reset_index(drop=True)


# =========================
# FEATURE
# =========================
def compute_features(df_rate12: pd.DataFrame, updated_at_webgia: str, run_time_vn: str) -> pd.DataFrame:
    if df_rate12.empty:
        raise RuntimeError("Không có dữ liệu rate12 để tính feature.")

    df = df_rate12.copy()
    df["group"] = df["ngan_hang"].apply(lambda b: "big4" if b in BIG4 else "private")

    big4_df = df[df["group"] == "big4"].copy()
    private_df = df[df["group"] == "private"].copy()
    trusted_df = private_df[private_df["ngan_hang"].apply(lambda x: canonical_key(x) in TRUSTED_PRIVATE_KEY)].copy()

    big4_used = sorted(big4_df["ngan_hang"].unique().tolist())
    private_used = sorted(private_df["ngan_hang"].unique().tolist())
    trusted_used = sorted(trusted_df["ngan_hang"].unique().tolist())

    logger.info(f"Parsed banks with 12M: {len(df)}")
    logger.info(f"Big4 used: {big4_used}")
    logger.info(f"Private count: {len(private_used)}")
    logger.info(f"Trusted private count: {len(trusted_used)}")

    unmatched = [x for x in private_used if canonical_key(x) not in TRUSTED_PRIVATE_KEY]
    logger.info(f"Unmatched private ({len(unmatched)}): {unmatched}")

    if len(big4_used) < 3:
        raise RuntimeError(f"Thiếu dữ liệu Big4 (có {len(big4_used)}): {big4_used}")
    if len(private_used) < 5:
        raise RuntimeError(f"Thiếu dữ liệu private (có {len(private_used)})")

    interest_rate_state = round(float(big4_df["rate12"].mean()), 2)

    market_method = "top3_trusted_mean"
    trusted_count = int(trusted_df["ngan_hang"].nunique())

    if trusted_count >= 3:
        top3 = trusted_df.sort_values("rate12", ascending=False).head(3)
        interest_rate_market = round(float(top3["rate12"].mean()), 2)
        trusted_top3_used = ", ".join(top3["ngan_hang"].tolist())
        trusted_top3_rates = ", ".join([str(x) for x in top3["rate12"].tolist()])
    elif trusted_count >= 1:
        market_method = "fallback_trusted_mean"
        topk = trusted_df.sort_values("rate12", ascending=False)
        interest_rate_market = round(float(topk["rate12"].mean()), 2)
        trusted_top3_used = ", ".join(topk["ngan_hang"].tolist())
        trusted_top3_rates = ", ".join([str(x) for x in topk["rate12"].tolist()])
    else:
        market_method = "fallback_max_private"
        top1 = private_df.sort_values("rate12", ascending=False).head(1)
        interest_rate_market = round(float(top1["rate12"].iloc[0]), 2)
        trusted_top3_used = ", ".join(top1["ngan_hang"].tolist())
        trusted_top3_rates = ", ".join([str(x) for x in top1["rate12"].tolist()])

    spread = round(interest_rate_market - interest_rate_state, 2)

    return pd.DataFrame([{
        "updated_at_vn": run_time_vn,
        "updated_at_webgia": updated_at_webgia,
        "interest_rate_state": interest_rate_state,
        "interest_rate_market": interest_rate_market,
        "interest_rate_spread": spread,
        "market_method": market_method,
        "trusted_count": trusted_count,
        "trusted_top3_used": trusted_top3_used,
        "trusted_top3_rates": trusted_top3_rates,
        "n_big4": len(big4_used),
        "n_private": len(private_used),
        "big4_used": ", ".join(big4_used),
        "source": URL,
        "note": f"term={TARGET_TERM_TEXT} | state=AVG(Big4) | market=TOP3_TRUSTED_MEAN"
    }])


# =========================
# CORE RUN
# =========================
def run_once() -> None:
    ensure_output_dirs()

    run_dt = now_vn_dt()
    run_time_vn = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    snap_id = ts_file_str(run_dt)
    screenshot_path = f"{OUT_SCREENSHOT_DIR}/debug_webgia_{snap_id}.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
            locale="vi-VN",
            viewport={"width": 1366, "height": 900}
        )
        page = context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)

        page.goto(URL, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=45000)
        except PWTimeout:
            logger.warning("networkidle timeout, tiếp tục parse DOM...")

        page.wait_for_timeout(RENDER_WAIT_MS)

        # PNG theo từng lần chạy
        page.screenshot(path=screenshot_path, full_page=True)

        updated_at_webgia = extract_webgia_updated_at(page)
        logger.info(f"updated_at_webgia parsed: {updated_at_webgia if updated_at_webgia else '(empty)'}")

        rows = extract_table_rows_from_dom(page)
        browser.close()

    logger.info(f"Extracted rows from DOM: {len(rows)}")
    if not rows:
        raise RuntimeError("DOM không có table rows (có thể anti-bot hoặc trang chưa render).")

    df_rate12 = rows_to_rate12_dataframe(rows)
    if df_rate12.empty:
        raise RuntimeError("Parse DOM xong nhưng không có rate12 hợp lệ.")

    # ===== APPEND vào CSV lãi suất chính =====
    df_rates_append = df_rate12.copy()
    df_rates_append.insert(0, "updated_at_vn", run_time_vn)
    df_rates_append.insert(1, "updated_at_webgia", updated_at_webgia)
    append_csv(df_rates_append, OUT_RATES_CSV, dedup_keys=["updated_at_vn", "ngan_hang"])

    # ===== APPEND vào CSV feature chính =====
    feature_df = compute_features(df_rate12, updated_at_webgia, run_time_vn)
    append_csv(feature_df, OUT_FEATURE_CSV, dedup_keys=["updated_at_vn"])

    logger.info("SUCCESS")
    logger.info(feature_df.to_dict(orient="records")[0])
    logger.info(f"Saved rates append to: {Path(OUT_RATES_CSV).resolve()}")
    logger.info(f"Saved macro append to: {Path(OUT_FEATURE_CSV).resolve()}")
    logger.info(f"Saved screenshot    : {Path(screenshot_path).resolve()}")


def run_with_retry():
    last_err = None
    for i in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Run attempt {i}/{MAX_RETRIES}")
            run_once()
            return
        except Exception as e:
            last_err = e
            logger.error(f"Attempt {i} failed: {e}")
            if i < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_RETRIES_SEC * i)
    raise RuntimeError(f"FAILED after {MAX_RETRIES} attempts: {last_err}")


# =========================
# OPTIONAL SCHEDULER
# =========================
def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(run_with_retry, "interval", minutes=RUN_EVERY_MINUTES, max_instances=1, coalesce=True)
    logger.info(f"Scheduler started: every {RUN_EVERY_MINUTES} minutes")
    scheduler.start()


if __name__ == "__main__":
    if ENABLE_SCHEDULER:
        run_with_retry()
        start_scheduler()
    else:
        run_with_retry()
