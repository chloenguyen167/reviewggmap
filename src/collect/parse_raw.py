"""Gộp dữ liệu thô (Playwright / SerpApi) về một schema chung (A7).

Đầu ra:
  interim/reviews_parsed.parquet     — 1 dòng / review, reviewer đã được hash
  interim/places.parquet             — metadata địa điểm (+ độ phủ thu thập)
  interim/contributors.parquet       — lịch sử reviewer (nếu có thu thập)
  interim/private_reviewer_map.parquet — hash → contributor_id THẬT (file riêng tư,
                                         chỉ dùng để thu thập lịch sử reviewer; không chia sẻ)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.relative_date import parse_relative_date
from src.collect.rpc_timestamps import extract_from_dir
from src.utils.common import get_logger, get_salt, hash_id, read_jsonl, write_table
from src.utils.config import Paths

RPC_TOLERANCE = timedelta(days=2)


# ---------------------------------------------------------------------------
# Hàm parse nhỏ (có unit test)
# ---------------------------------------------------------------------------
# Khối thuộc tính Google chèn dưới nội dung review (nhà hàng/quán cà phê)
_ATTR_LINE_RE = re.compile(
    r"^(Loại đơn đặt món|Loại hình bữa ăn|Giá mỗi người|Đồ ăn:|Dịch vụ:|Bầu không khí:|Độ ồn|"
    r"Thời gian chờ|Quy mô nhóm|Lựa chọn đỗ xe|Điểm đỗ xe|Những món ăn đề xuất|Đặt chỗ|"
    r"Món chay|Khả năng tiếp cận|Phòng|Khách sạn|Vị trí:|Phòng:|"
    r"Meal type|Price per person|Food:|Service:|Atmosphere:|Noise level|Wait time|Group size|Parking)",
    re.IGNORECASE)
_MORE_RE = re.compile(r"\s*(Xem thêm|See more|More)\s*$")


def clean_pw_text(text, owner_response=None) -> tuple[str, bool]:
    """Gỡ phần giao diện lẫn vào nội dung review do selector cũ lấy khung cha (.MyEned).

    - review chỉ chấm sao + có phản hồi của chủ quán → text là phản hồi đó → trả về ""
    - bỏ nút "Xem thêm" và khối thuộc tính (Loại đơn đặt món, Đồ ăn: 5, Giá mỗi người…)
    Trả về (text sạch, bị_cắt_cụt) — bị cắt cụt = Google chỉ hiển thị một phần ("…").
    """
    t = (text or "").strip()
    if not t:
        return "", False
    own = (owner_response or "").strip()
    if own and own.endswith(t[-80:]) and len(t) <= len(own):
        return "", False
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if _ATTR_LINE_RE.match(s):
            break
        if s in ("Xem thêm", "See more"):
            continue
        lines.append(ln)
    t = _MORE_RE.sub("", "\n".join(lines)).strip()
    truncated = t.endswith("…")
    return t.rstrip("…").strip() if truncated else t, truncated


def parse_rating_label(label) -> float | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    if isinstance(label, (int, float)):
        return float(label)
    t = str(label).lower()
    m = re.search(r"([1-5])(?:[.,]0)?\s*(?:/\s*5|sao|stars?|trên|out of)", t)
    if not m:
        m = re.search(r"\b([1-5])\b", t)
    return float(m.group(1)) if m else None


def _int_before(pattern: str, text: str) -> int | None:
    m = re.search(r"([\d.,]+)\s*(?:" + pattern + r")", text)
    if not m:
        return None
    digits = re.sub(r"[.,]", "", m.group(1))
    return int(digits) if digits.isdigit() else None


def parse_reviewer_meta(meta: str | None) -> dict:
    """'Local Guide · 132 bài đánh giá · 1.024 ảnh' → {local_guide, n_reviews, n_photos}"""
    if not meta:
        return {"reviewer_local_guide": None, "reviewer_n_reviews": None, "reviewer_n_photos": None}
    t = str(meta).lower()
    return {
        "reviewer_local_guide": ("local guide" in t) or ("người đóng góp địa phương" in t),
        "reviewer_n_reviews": _int_before(r"bài đánh giá|đánh giá|reviews?", t),
        "reviewer_n_photos": _int_before(r"ảnh|photos?", t),
    }


def _to_utc(x) -> datetime | None:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "":
        return None
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


# ---------------------------------------------------------------------------
# Playwright
# ---------------------------------------------------------------------------
def _parse_playwright(P: Paths, salt: str, log) -> tuple[list[dict], dict]:
    rows, private = [], {}
    files = sorted((P.raw / "reviews").glob("pw_*.jsonl"))
    n_rpc_used = n_rpc_rejected = 0
    for f in files:
        recs = read_jsonl(f)
        if not recs:
            continue
        key = recs[0]["place_key"]
        known = {r["review_id"] for r in recs if r.get("review_id")}
        rpc = extract_from_dir(P.raw / "network" / key, known_ids=known)
        for r in recs:
            scraped = _to_utc(r["scraped_at"])
            rd = parse_relative_date(r.get("date_text"), scraped)
            date_source, precision = "relative", rd.precision_days
            review_date, lo, hi = rd.date_est, rd.date_lo, rd.date_hi
            ts = rpc.get(r.get("review_id"))
            if ts is not None and rd.parsed:
                # Chỉ chấp nhận timestamp RPC nếu khớp với ngày hiển thị.
                # Review đã chỉnh sửa: ngày hiển thị là ngày sửa → ngày tạo phải <= hi.
                ok_hi = ts <= hi + RPC_TOLERANCE
                ok_lo = rd.is_edited or ts >= lo - RPC_TOLERANCE
                if ok_hi and ok_lo:
                    review_date, lo, hi, precision, date_source = ts, ts, ts, 0.0, "rpc"
                    n_rpc_used += 1
                else:
                    n_rpc_rejected += 1
            elif ts is not None and not rd.parsed:
                review_date, lo, hi, precision, date_source = ts, ts, ts, 0.0, "rpc_unverified"
                n_rpc_used += 1

            text, truncated = clean_pw_text(r.get("text"), r.get("owner_response"))
            h = hash_id(r.get("contributor_id") or r.get("reviewer_name"), salt)
            if h and r.get("contributor_id"):
                private[h] = r["contributor_id"]
            rows.append({
                "review_id": r.get("review_id"), "source": "playwright", "place_key": key,
                "reviewer_hash": h, **parse_reviewer_meta(r.get("reviewer_meta")),
                "rating": parse_rating_label(r.get("rating_label")),
                "text": text, "text_truncated": truncated, "date_text": r.get("date_text"),
                "review_date": review_date, "date_lo": lo, "date_hi": hi,
                "date_precision_days": precision, "date_source": date_source, "is_edited": rd.is_edited,
                "has_owner_response": bool(r.get("owner_response")), "n_photos": int(r.get("n_photos") or 0),
                "scraped_at": scraped,
            })
    log.info("Playwright: %d file, %d review | timestamp RPC dùng=%d, bị loại (không khớp)=%d",
             len(files), len(rows), n_rpc_used, n_rpc_rejected)
    return rows, private


# ---------------------------------------------------------------------------
# SerpApi
# ---------------------------------------------------------------------------
def _parse_serpapi(P: Paths, salt: str, log) -> tuple[list[dict], dict]:
    rows, private = [], {}
    files = sorted((P.raw / "reviews").glob("serp_*.jsonl"))
    for f in files:
        for page in read_jsonl(f):
            scraped = _to_utc(page["fetched_at"])
            for rv in page.get("reviews", []) or []:
                user = rv.get("user") or {}
                cid = user.get("contributor_id")
                h = hash_id(cid or user.get("name"), salt)
                if h and cid:
                    private[h] = cid
                iso = _to_utc(rv.get("iso_date"))
                if iso is not None:
                    lo = hi = review_date = iso
                    precision, src, edited = 0.0, "iso", bool(rv.get("iso_date_of_last_edit"))
                else:
                    rd = parse_relative_date(rv.get("date"), scraped)
                    review_date, lo, hi, precision, src, edited = (rd.date_est, rd.date_lo, rd.date_hi,
                                                                   rd.precision_days, "relative", rd.is_edited)
                text = rv.get("snippet") or (rv.get("extracted_snippet") or {}).get("original") or ""
                rows.append({
                    "review_id": rv.get("review_id"), "source": "serpapi", "place_key": page["place_key"],
                    "reviewer_hash": h, "reviewer_local_guide": user.get("local_guide"),
                    "reviewer_n_reviews": user.get("reviews"), "reviewer_n_photos": user.get("photos"),
                    "rating": parse_rating_label(rv.get("rating")), "text": text, "date_text": rv.get("date"),
                    "review_date": review_date, "date_lo": lo, "date_hi": hi,
                    "date_precision_days": precision, "date_source": src, "is_edited": edited,
                    "has_owner_response": bool(rv.get("response")), "n_photos": len(rv.get("images") or []),
                    "scraped_at": scraped,
                })
    log.info("SerpApi: %d file, %d review", len(files), len(rows))
    return rows, private


# ---------------------------------------------------------------------------
# Lịch sử reviewer
# ---------------------------------------------------------------------------
def _parse_contributors(P: Paths, salt: str) -> pd.DataFrame:
    rows = []
    for f in sorted((P.raw / "contributors").glob("*.jsonl")):
        for r in read_jsonl(f):
            scraped = _to_utc(r.get("scraped_at")) or datetime.now(timezone.utc)
            iso = _to_utc(r.get("iso_date"))
            if iso is not None:
                d, prec = iso, 0.0
            else:
                rd = parse_relative_date(r.get("date_text"), scraped)
                d, prec = rd.date_est, rd.precision_days
            rows.append({"reviewer_hash": hash_id(r["contributor_id"], salt), "place_name": r.get("place_name"),
                         "lat": r.get("lat"), "lng": r.get("lng"),
                         "rating": parse_rating_label(r.get("rating", r.get("rating_label"))),
                         "review_date": d, "date_precision_days": prec})
    return pd.DataFrame(rows)


def _places_table(P: Paths, cfg: dict, reviews: pd.DataFrame) -> pd.DataFrame:
    pf = P.resolve(cfg["collect"]["places_file"])
    places = pd.read_csv(pf) if pf.exists() else pd.DataFrame({"place_key": reviews["place_key"].unique()})
    stats = []
    for d in (P.raw / "reviews").glob("*.done.json"):
        stats.append(json.loads(d.read_text(encoding="utf-8")))
    if stats:
        st = pd.DataFrame(stats)[["place_key", "expected"]].drop_duplicates("place_key")
        places = places.merge(st, on="place_key", how="left")
    fetched = reviews.groupby("place_key").size().rename("fetched").reset_index()
    places = places.merge(fetched, on="place_key", how="left")
    exp = places["expected"] if "expected" in places else places.get("user_rating_count")
    if exp is not None:
        places["coverage"] = places["fetched"] / pd.to_numeric(exp, errors="coerce")
    return places


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("parse", P.logs)
    salt = get_salt(cfg)
    rows_pw, priv_pw = _parse_playwright(P, salt, log)
    rows_sp, priv_sp = _parse_serpapi(P, salt, log)
    df = pd.DataFrame(rows_pw + rows_sp)
    if df.empty:
        raise RuntimeError(f"Không có dữ liệu thô trong {P.raw / 'reviews'} — chạy bước collect trước.")
    for c in ("review_date", "date_lo", "date_hi", "scraped_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["reviewer_n_reviews"] = pd.to_numeric(df["reviewer_n_reviews"], errors="coerce")
    df["reviewer_n_photos"] = pd.to_numeric(df["reviewer_n_photos"], errors="coerce")
    df["reviewer_local_guide"] = df["reviewer_local_guide"].astype("boolean")
    # review_id bị thiếu → tạo id xác định từ nội dung
    miss = df["review_id"].isna()
    df.loc[miss, "review_id"] = [
        "gen_" + hash_id(f"{r.place_key}|{r.reviewer_hash}|{r.text}|{r.date_text}", "rid", 20)
        for r in df[miss].itertuples()]

    # Loại địa điểm có độ phủ thu thập quá thấp: phần lấy được là mẫu bị chọn lệch (vd. Google chỉ hiện
    # ~30 review theo thứ tự "liên quan", không sắp xếp được) → làm sai chuỗi thời gian (M2) và tỷ lệ.
    places = _places_table(P, cfg, df)
    min_cov = cfg["collect"].get("min_coverage", 0.5)
    if "coverage" in places:
        low = places[places["coverage"].notna() & (places["coverage"] < min_cov)]
        if len(low):
            for r in low.itertuples():
                log.warning("LOẠI địa điểm %s: độ phủ %.0f%% (%s/%s review) < %.0f%%", getattr(r, "name", r.place_key),
                            100 * r.coverage, int(r.fetched), int(r.expected) if r.expected == r.expected else "?",
                            100 * min_cov)
            df = df[~df["place_key"].isin(low["place_key"])]
        places["excluded_low_coverage"] = places["place_key"].isin(low["place_key"])

    out = P.interim / "reviews_parsed.parquet"
    df.to_parquet(out, index=False)
    places.to_parquet(P.interim / "places.parquet", index=False)
    pd.DataFrame({"reviewer_hash": list({**priv_pw, **priv_sp}.keys()),
                  "contributor_id": list({**priv_pw, **priv_sp}.values())}).to_parquet(
        P.interim / "private_reviewer_map.parquet", index=False)
    contrib = _parse_contributors(P, salt)
    if not contrib.empty:
        contrib["review_date"] = pd.to_datetime(contrib["review_date"], utc=True)
        contrib.to_parquet(P.interim / "contributors.parquet", index=False)
        log.info("Lịch sử reviewer: %d dòng", len(contrib))
    log.info("Đã ghi %s (%d review, %d địa điểm)", out.name, len(df), df["place_key"].nunique())
    return out
