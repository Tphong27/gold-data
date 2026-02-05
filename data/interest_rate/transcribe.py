#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import logging
from datetime import datetime
from typing import Any, List, Dict, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ================= CONFIG =================
URL = "https://webgia.com/lai-suat/"

OUT_RATES_CSV = "webgia_laisuat_latest_clean.csv"
OUT_FEATURE_CSV = "macro_features_latest.csv"
OUT_DEBUG_SCREENSHOT = "debug_webgia.png"

BIG4 = {"Vietcombank", "BIDV", "Agribank", "VietinBank"}
TARGET_TERM_MONTH = 12

HEADLESS = True
MAX_RETRIES = 3
PAGE_TIMEOUT_MS = 90000
RENDER_WAIT_MS = 7000

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("webgia_actions")


# ================= UTILS =================
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


def key_text(s: Any) -> str:
    return strip_accents_vn(norm_space(s)).lower()


def parse_rate(v: Any):
    s = norm_space(v).lower()
    if not s or s == "-" or "webgia" in s:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    x = float(m.group(0).replace(",", "."))
    return x if 0 < x < 30 else None


def normalize_bank(s: Any) -> str:
    t = norm_space(s)
    k = t.lower()
    m = {
        "vietcombank": "Vietcombank",
        "bidv": "BIDV",
        "agribank": "Agribank",
        "vietinbank": "VietinBank",
        "vietin bank": "VietinBank",
        "vpbank": "VPBank",
        "hdbank": "HDBank",
        "vib": "VIB",
    }
    return m.get(k, t)


def is_bad_bank_label(s: str) -> bool:
    t = key_text(s).replace(" ", "")
    bad = ["nganhang", "kyhan", "khongkyhan", "laisuat", "thang", "usd", "eur", "jpy"]
    return any(b in t for b in bad)


# ================= DOM PARSE =================
def extract_best_table_rows(page) -> List[List[str]]:
    rows = page.evaluate(
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

            ["01 tháng","03 tháng","06 tháng","09 tháng","12 tháng","13 tháng","18 tháng","24 tháng","36 tháng"]
              .forEach(x => { if (text.includes(x)) score += 8; });

            score += trCount;
            if (text.includes("usd") || text.includes("eur") || text.includes("jpy")) score -= 30;
            if (trCount < 8) score -= 20;

            if (score > bestScore) {
              bestScore = score;
              best = t;
            }
          }

          if (!best) return [];

          return Array.from(best.querySelectorAll("tr"))
            .map(tr => Array.from(tr.querySelectorAll("th,td"))
              .map(td => (td.innerText || "").replace(/\\s+/g, " ").trim()))
            .filter(r => r.length > 0);
        }
        """
    )
    return rows


def detect_header_and_cols(rows: List[List[str]]) -> Tuple[int, int, Dict[int, int]]:
    header_idx = -1
    bank_col = 0
    term_map: Dict[int, int] = {}

    # Tìm hàng header kỳ hạn thật
    for i, r in enumerate(rows[:12]):
        line = key_text(" ".join(r))
        hits = 0
        for t in ["01", "03", "06", "09", "12", "13", "18", "24", "36"]:
            if re.search(rf"\\b{t}\\b", line):
                hits += 1
        if hits >= 5:
            header_idx = i
            break

    if header_idx == -1:
        header_idx = 1 if len(rows) > 1 else 0

    header = rows[header_idx]
    for j, cell in enumerate(header):
        ct = key_text(cell)
        if "ngan hang" in ct:
            bank_col = j
        m = re.search(r"\\b(\\d{1,2})\\s*(thang)?\\b", ct)
        if m:
            mm = int(m.group(1))
            if mm in [1,3,6,9,12,13,18,24,36]:
                term_map[mm] = j

    return header_idx, bank_col, term_map


def rows_to_df_rate12(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ngan_hang", "rate12"])

    header_idx, bank_col, term_map = detect_header_and_cols(rows)
    term12_col = term_map.get(12)

    if term12_col is None:
        raise RuntimeError(f"Không tìm thấy cột 12 tháng. term_map={term_map}")

    # FIX lệch cột do row header không có ô "Ngân hàng" (rowspan)
    header_has_bank = any("ngan hang" in key_text(x) for x in rows[header_idx])
    if bank_col == 0 and not header_has_bank:
        term12_col += 1

    logger.info(f"header_idx={header_idx}, bank_col={bank_col}, term12_col={term12_col}, term_map={term_map}, header_has_bank={header_has_bank}")

    parsed = []
    for r in rows[header_idx + 1:]:
        if bank_col >= len(r):
            continue

        bank = normalize_bank(r[bank_col])
        if not bank or len(bank) < 2 or is_bad_bank_label(bank):
            continue

        if term12_col >= len(r):
            continue

        rate = parse_rate(r[term12_col])
        if rate is None:
            continue

        parsed.append({"ngan_hang": bank, "rate12": rate})

    if not parsed:
        return pd.DataFrame(columns=["ngan_hang", "rate12"])

    df = pd.DataFrame(parsed).drop_duplicates(subset=["ngan_hang"]).reset_index(drop=True)
    return df


# ================= BUSINESS =================
def compute_feature(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("Không có dữ liệu rate12.")

    tmp = df.copy()
    tmp["group"] = tmp["ngan_hang"].apply(lambda x: "big4" if x in BIG4 else "private")

    big4 = tmp[tmp["group"] == "big4"]
    private = tmp[tmp["group"] == "private"]

    big4_used = sorted(big4["ngan_hang"].unique().tolist())
    n_private = private["ngan_hang"].nunique()

    logger.info(f"Big4 used: {big4_used}, private_count={n_private}")
    logger.info("Big4 sample: %s", big4[["ngan_hang", "rate12"]].to_dict(orient="records"))

    if len(big4_used) < 3:
        raise RuntimeError(f"Thiếu Big4: {big4_used}")
    if n_private < 5:
        raise RuntimeError(f"Thiếu private banks: {n_private}")

    state = round(float(big4["rate12"].mean()), 2)
    market = round(float(private["rate12"].max()), 2)
    spread = round(market - state, 2)

    return pd.DataFrame([{
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interest_rate_state": state,
        "interest_rate_market": market,
        "interest_rate_spread": spread,
        "n_big4": len(big4_used),
        "n_private": int(n_private),
        "big4_used": ", ".join(big4_used),
        "source": URL,
        "note": "term=12 tháng | state=AVG(Big4) | market=MAX(private)"
    }])


# ================= RUN =================
def run_once():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
            locale="vi-VN",
            viewport={"width": 1366, "height": 900}
        )
        page = ctx.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)

        page.goto(URL, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=45000)
        except PWTimeout:
            logger.warning("networkidle timeout, continue...")
        page.wait_for_timeout(RENDER_WAIT_MS)

        page.screenshot(path=OUT_DEBUG_SCREENSHOT, full_page=True)

        rows = extract_best_table_rows(page)
        browser.close()

    logger.info(f"Extracted row count: {len(rows)}")
    if not rows:
        raise RuntimeError("Không lấy được rows từ DOM table.")

    df_rate = rows_to_df_rate12(rows)
    if df_rate.empty:
        raise RuntimeError("Parse xong nhưng không có rate12 hợp lệ.")

    feat = compute_feature(df_rate)

    df_rate.to_csv(OUT_RATES_CSV, index=False, encoding="utf-8-sig")
    feat.to_csv(OUT_FEATURE_CSV, index=False, encoding="utf-8-sig")

    logger.info("SUCCESS: %s", feat.to_dict(orient="records")[0])


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
                time.sleep(i * 3)
    raise RuntimeError(f"FAILED after retries: {last_err}")


if __name__ == "__main__":
    run_with_retry()
