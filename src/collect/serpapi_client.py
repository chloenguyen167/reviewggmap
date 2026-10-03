"""Thu thập qua SerpApi (phương án thay thế khi không dùng Playwright).

Áp dụng đúng kỹ thuật ingestion Chương 2:
  - timeout cố định, retry với exponential backoff + jitter (tenacity)
  - HTTP 429: tôn trọng header Retry-After
  - 401/403: dừng ngay (lỗi key/quota), không retry
  - đếm lượt gọi, dừng khi chạm max_calls
  - checkpoint từng trang vào jsonl; chạy lại tiếp tục từ next_page_token cuối
  - bẫy "trang rỗng nhưng Success" (serpapi/public-roadmap#4378): gửi lại token với no_cache=true
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from src.collect.place_url import place_key
from src.utils.common import append_jsonl, get_logger, read_jsonl
from src.utils.config import Paths

ENDPOINT = "https://serpapi.com/search.json"


class FatalHTTPError(Exception):
    pass


class RetryableHTTPError(Exception):
    def __init__(self, msg, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class QuotaExceeded(Exception):
    pass


class SerpApiClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        sc = cfg["collect"]["serpapi"]
        self.h = cfg["collect"]["http"]
        self.key = os.environ.get(sc["api_key_env"])
        if not self.key:
            raise RuntimeError(f"Chưa đặt biến môi trường {sc['api_key_env']} (xem .env.example)")
        self.max_calls = sc["max_calls"]
        self.calls = 0
        self.log = get_logger("serpapi", Paths(cfg).logs)
        self.session = requests.Session()

    def get(self, params: dict) -> dict:
        if self.calls >= self.max_calls:
            raise QuotaExceeded(f"Đã dùng {self.calls}/{self.max_calls} lượt gọi")

        @retry(stop=stop_after_attempt(self.h["max_retries"]),
               wait=wait_exponential_jitter(initial=self.h["backoff_base_s"], max=60),
               retry=retry_if_exception(lambda e: isinstance(e, (RetryableHTTPError, requests.ConnectionError,
                                                                 requests.Timeout))),
               reraise=True)
        def _do():
            self.calls += 1
            t0 = time.time()
            r = self.session.get(ENDPOINT, params={**params, "api_key": self.key}, timeout=self.h["timeout_s"])
            safe = {k: v for k, v in params.items() if k != "api_key"}
            self.log.info("GET %s -> %s (%.1fs)", safe, r.status_code, time.time() - t0)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(int(ra))
                raise RetryableHTTPError("429 Too Many Requests")
            if r.status_code in (401, 403):
                raise FatalHTTPError(f"{r.status_code}: kiểm tra API key / quota")
            if r.status_code >= 500:
                raise RetryableHTTPError(f"{r.status_code} server error")
            r.raise_for_status()
            return r.json()

        return _do()


def search_places(cfg: dict) -> pd.DataFrame:
    cl = SerpApiClient(cfg)
    rows = []
    for q in cfg["collect"]["search_queries"]:
        data = cl.get({"engine": "google_maps", "q": q, "hl": cfg["collect"]["serpapi"]["hl"], "type": "search"})
        for it in data.get("local_results", [])[: cfg["collect"]["max_results_per_query"]]:
            gps = it.get("gps_coordinates") or {}
            rows.append({"query": q, "name": it.get("title"), "place_id": it.get("place_id"),
                         "data_id": it.get("data_id"), "rating": it.get("rating"),
                         "user_rating_count": it.get("reviews"), "lat": gps.get("latitude"),
                         "lng": gps.get("longitude"), "type": it.get("type"),
                         "url": f"https://www.google.com/maps/place/?q=place_id:{it.get('place_id')}"})
    df = pd.DataFrame(rows).drop_duplicates("place_id")
    df["place_key"] = [place_key(n, u, p) for n, u, p in zip(df["name"], df["url"], df["place_id"])]
    return df


def fetch_place_reviews(cl: SerpApiClient, cfg: dict, row: dict, out_dir: Path) -> dict:
    sc = cfg["collect"]["serpapi"]
    key = row["place_key"]
    out = out_dir / f"serp_{key}.jsonl"
    done = out_dir / f"serp_{key}.done.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))

    # Tiếp tục từ checkpoint nếu có
    token, page_idx, fetched = None, 0, 0
    if out.exists():
        pages = read_jsonl(out)
        if pages:
            token = pages[-1].get("next_page_token")
            page_idx = pages[-1]["page_idx"] + 1
            fetched = sum(len(p.get("reviews", [])) for p in pages)
            if token is None:                                   # trang cuối đã lấy xong
                page_idx = -1

    expected = row.get("user_rating_count")
    while page_idx >= 0:
        params = {"engine": "google_maps_reviews", "hl": sc["hl"], "sort_by": sc["sort_by"]}
        params.update({"data_id": row["data_id"]} if row.get("data_id") else {"place_id": row["place_id"]})
        if token:
            params["next_page_token"] = token
            params["num"] = 20
        data = cl.get(params)
        reviews = data.get("reviews", []) or []

        # Bẫy trang rỗng giữa chừng
        retries = 0
        while (not reviews and token and expected and fetched < 0.95 * float(expected)
               and retries < sc["empty_page_retries"]):
            retries += 1
            cl.log.warning("Trang rỗng ở trang %d (%d/%s) → thử lại no_cache=true", page_idx, fetched, expected)
            data = cl.get({**params, "no_cache": "true"})
            reviews = data.get("reviews", []) or []

        if page_idx == 0 and not expected:
            expected = (data.get("place_info") or {}).get("reviews")
        token = (data.get("serpapi_pagination") or {}).get("next_page_token")
        append_jsonl(out, [{"source": "serpapi", "place_key": key, "page_idx": page_idx,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "next_page_token": token, "reviews": reviews}])
        fetched += len(reviews)
        page_idx = page_idx + 1 if (token and reviews) else -1
        time.sleep(random.uniform(0.5, 1.5))

    stats = {"place_key": key, "name": row.get("name"), "expected": expected, "fetched": fetched,
             "coverage": (fetched / float(expected)) if expected else None, "calls_used": cl.calls}
    done.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    return stats


def run_search(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    df = search_places(cfg)
    out = P.raw / "places" / "places_candidates.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    f = cfg["collect"]["place_filter"]
    df[df["user_rating_count"].between(f["min_reviews"], f["max_reviews"])].to_csv(
        P.resolve(cfg["collect"]["places_file"]), index=False, encoding="utf-8-sig")
    return out


def run_reviews(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    cl = SerpApiClient(cfg)
    places = pd.read_csv(P.resolve(cfg["collect"]["places_file"]))
    stats = []
    for row in places.to_dict("records"):
        try:
            stats.append(fetch_place_reviews(cl, cfg, row, P.raw / "reviews"))
        except QuotaExceeded as e:
            cl.log.error("%s — dừng. Chạy lại sau để tiếp tục từ checkpoint.", e)
            break
        except FatalHTTPError as e:
            cl.log.error("Lỗi không thể thử lại: %s", e)
            break
    out = P.tables / "collect_stats_serpapi.csv"
    pd.DataFrame(stats).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def run_contributors(cfg: dict, contributor_ids: list[str]) -> int:
    P = Paths(cfg).ensure()
    cl = SerpApiClient(cfg)
    n = 0
    for cid in contributor_ids:
        out = P.raw / "contributors" / f"serp_{cid}.jsonl"
        if out.exists():
            continue
        try:
            data = cl.get({"engine": "google_maps_contributor_reviews", "contributor_id": cid,
                           "hl": cfg["collect"]["serpapi"]["hl"]})
        except QuotaExceeded:
            break
        recs = []
        for rv in data.get("reviews", []) or []:
            pi = rv.get("place_info") or {}
            gps = pi.get("gps_coordinates") or {}
            recs.append({"source": "serpapi", "contributor_id": cid, "place_name": pi.get("title"),
                         "lat": gps.get("latitude"), "lng": gps.get("longitude"),
                         "rating": rv.get("rating"), "iso_date": rv.get("iso_date"), "date_text": rv.get("date"),
                         "text": rv.get("snippet"), "review_id": rv.get("review_id")})
        append_jsonl(out, recs)
        n += len(recs)
    return n
