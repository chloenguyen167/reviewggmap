"""Thu thập Google Maps bằng Playwright (không cần API key).

Ba chức năng:
  search_places     : tìm địa điểm theo truy vấn → raw/places/places_candidates.csv
  scrape_reviews    : lấy toàn bộ review cho từng địa điểm trong places_selected.csv
  scrape_contributors: lấy lịch sử review của một số reviewer nghi vấn

Nguyên tắc:
  - Chạy tuần tự, nghỉ ngẫu nhiên giữa các lần cuộn và giữa các địa điểm.
  - Mỗi địa điểm ghi ra một file riêng; có file *.done.json thì lần chạy sau bỏ qua.
  - Selector nằm trong config.yaml → khi Google đổi giao diện chỉ cần sửa config.
  - Bắt các gói RPC (listugcposts) để lấy timestamp tuyệt đối (xem rpc_timestamps.py).

LƯU Ý: Script chỉ chạy được trên máy truy cập được google.com/maps.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.collect.place_url import contributor_id_from_url, parse_place_url, place_key, with_lang
from src.utils.common import get_logger
from src.utils.config import Paths

# ---------------------------------------------------------------------------
# JavaScript chạy trong trang
# ---------------------------------------------------------------------------
JS_FIND_SCROLLER = """
(cardSel) => {
  const card = document.querySelector(cardSel);
  if (!card) return false;
  document.querySelectorAll('[data-gm-scroll]').forEach(e => e.removeAttribute('data-gm-scroll'));
  let el = card.parentElement;
  while (el && el !== document.body) {
    const st = getComputedStyle(el);
    if ((st.overflowY === 'auto' || st.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 5) {
      el.setAttribute('data-gm-scroll', '1');
      return true;
    }
    el = el.parentElement;
  }
  return false;
}
"""

JS_SCROLL = """
() => {
  const el = document.querySelector('[data-gm-scroll]');
  if (el) { el.scrollTop = el.scrollHeight; return el.scrollHeight; }
  window.scrollTo(0, document.body.scrollHeight);
  return document.body.scrollHeight;
}
"""

JS_COUNT = """
(cardSel) => new Set([...document.querySelectorAll(cardSel)].map(e => e.getAttribute('data-review-id'))).size
"""

# Bọc XHR/fetch ngay trong trang: lưu body các gói RPC review vào window.__rpcBodies.
# Ổn định hơn bắt qua CDP (sự kiện response/requestfinished bị mất body ngẫu nhiên do bộ đệm của Chrome).
JS_RPC_HOOK = """
(() => {
  const KWS = %s;
  const match = (u) => typeof u === 'string' && KWS.some(k => u.includes(k));
  window.__rpcBodies = window.__rpcBodies || [];
  const push = (t) => { if (typeof t === 'string' && t.length) window.__rpcBodies.push(t); };
  const open = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (m, u) { this.__rpcUrl = String(u); return open.apply(this, arguments); };
  const send = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function () {
    if (match(this.__rpcUrl)) this.addEventListener('load', () => {
      try { push(this.responseType === '' || this.responseType === 'text' ? this.responseText : ''); } catch (e) {}
    });
    return send.apply(this, arguments);
  };
  const f = window.fetch;
  if (f) window.fetch = function (input) {
    const u = typeof input === 'string' ? input : (input && input.url) || '';
    const p = f.apply(this, arguments);
    if (match(u)) p.then(r => r.clone().text().then(push)).catch(() => {});
    return p;
  };
})();
"""

JS_EXPAND = """
(moreSel) => { let n = 0; document.querySelectorAll(moreSel).forEach(b => { try { b.click(); n++; } catch (e) {} }); return n; }
"""

JS_EXTRACT = """
(S) => {
  const cards = [...document.querySelectorAll(S.review_card)];
  const seen = new Set(); const out = [];
  for (const c of cards) {
    const id = c.getAttribute('data-review-id');
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const q = (sel) => sel ? c.querySelector(sel) : null;
    const txt = (sel) => { const e = q(sel); return e ? e.innerText.trim() : null; };
    // Nội dung review: thử từng selector THEO THỨ TỰ ƯU TIÊN (querySelector với danh sách
    // "a, b" trả về phần tử đứng trước trong DOM → lấy nhầm khung cha có nút "Xem thêm"),
    // và bỏ qua phần tử nằm trong phản hồi của chủ quán (review chỉ chấm sao).
    const reviewText = () => {
      for (const sel of (S.text || '').split(',').map(s => s.trim()).filter(Boolean)) {
        for (const e of c.querySelectorAll(sel)) {
          if (S.owner_response && e.closest(S.owner_response)) continue;
          return e.innerText.trim();
        }
      }
      return null;
    };
    let stars = null;
    const se = q(S.stars);
    if (se) stars = se.getAttribute('aria-label') || se.innerText;
    if (!stars) stars = txt(S.stars_alt);
    let prof = null;
    const pe = q(S.profile_link);
    if (pe) prof = pe.getAttribute('data-href') || pe.getAttribute('href');
    out.push({
      review_id: id,
      reviewer_name: txt(S.name) || c.getAttribute('aria-label'),
      reviewer_meta: txt(S.meta),
      rating_label: stars,
      date_text: txt(S.date),
      text: reviewText(),
      owner_response: txt(S.owner_response),
      n_photos: S.photo ? c.querySelectorAll(S.photo).length : 0,
      profile_url: prof,
    });
  }
  return out;
}
"""

JS_EXTRACT_RESULTS = """
(S) => {
  const out = [];
  document.querySelectorAll(S.result_link).forEach(a => {
    const card = a.closest('div[jsaction]') || a.parentElement;
    const r = card ? card.querySelector(S.result_rating) : null;
    const n = card ? card.querySelector(S.result_count) : null;
    let count = n ? n.innerText : null;
    if (!count && card) {   // dự phòng: aria-label "4,0 sao 805 bài đánh giá"
      const img = card.querySelector('span[role="img"][aria-label*="sao"], span[role="img"][aria-label*="star"]');
      const m = img ? img.getAttribute('aria-label').match(/([\\d.,]+)\\s*(bài đánh giá|reviews?)/i) : null;
      count = m ? m[1] : null;
    }
    out.push({ name: a.getAttribute('aria-label'), url: a.href,
               rating_text: r ? r.innerText : null, count_text: count });
  });
  return out;
}
"""


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _pause(rng: random.Random, lo_hi_ms) -> None:
    lo, hi = lo_hi_ms
    time.sleep(rng.uniform(lo, hi) / 1000.0)


def _num(text: str | None) -> float | None:
    """'4,6' -> 4.6 ; '(1.234)' -> 1234"""
    if not text:
        return None
    t = re.sub(r"[()\s]", "", text)
    if re.fullmatch(r"\d+,\d", t):            # điểm TB kiểu VN
        return float(t.replace(",", "."))
    digits = re.sub(r"[^\d]", "", t)
    return float(digits) if digits else None


def _try_click(page, selectors, timeout_ms: int = 4000) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=timeout_ms)
            loc.click()
            return True
        except Exception:
            continue
    return False


class MapsScraper:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pc = cfg["collect"]["playwright"]
        self.S = self.pc["selectors"]
        self.P = Paths(cfg).ensure()
        self.log = get_logger("scrape", self.P.logs)
        self.rng = random.Random(cfg["project"]["seed"])
        self._pw = self._browser = self._ctx = None
        self._responses: list = []
        self.n_net_failed = 0
        self._page = None

    # ---- vòng đời trình duyệt -------------------------------------------
    def __enter__(self):
        from playwright.sync_api import sync_playwright  # import muộn: module khác không cần playwright
        self._pw = sync_playwright().start()
        kw = dict(headless=self.pc["headless"], locale="vi-VN", viewport=self.pc["viewport"],
                  args=["--disable-blink-features=AutomationControlled"],
                  ignore_default_args=["--enable-automation"])
        if self.pc.get("executable_path"):
            kw["executable_path"] = self.pc["executable_path"]
        if kw["headless"]:      # UA "HeadlessChrome" → Google trả Maps "chế độ bị hạn chế" dù đã đăng nhập
            kw["user_agent"] = self._headful_user_agent(kw.get("executable_path"))
        if self.pc.get("user_data_dir"):
            self._ctx = self._pw.chromium.launch_persistent_context(self.pc["user_data_dir"], **kw)
        else:
            launch_kw = {k: kw[k] for k in ("headless", "executable_path", "args", "ignore_default_args") if k in kw}
            self._browser = self._pw.chromium.launch(**launch_kw)
            self._ctx = self._browser.new_context(locale="vi-VN", viewport=self.pc["viewport"],
                                                  user_agent=kw.get("user_agent"))
        self._ctx.set_default_navigation_timeout(self.pc["nav_timeout_ms"])
        return self

    def _headful_user_agent(self, executable_path: str | None) -> str:
        b = self._pw.chromium.launch(headless=True, executable_path=executable_path)
        try:
            p = b.new_page()
            ua = p.evaluate("navigator.userAgent")
        finally:
            b.close()
        return ua.replace("HeadlessChrome", "Chrome")

    def __exit__(self, *exc):
        try:
            if self._ctx:
                self._ctx.close()
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    def _new_page(self):
        page = self._ctx.new_page()
        if self.pc.get("capture_network"):
            kws = self.pc["network_url_keywords"]
            # "requestfinished" (không phải "response"): batchexecute trả body dạng chunked, lúc sự kiện
            # "response" bắn ra body có thể chưa về → "No data found for resource with given identifier"
            page.on("requestfinished", lambda req: self._grab(req) if any(k in req.url for k in kws) else None)
            page.add_init_script(JS_RPC_HOOK % json.dumps(kws))
        self._page = page
        return page

    def _grab(self, req) -> None:
        """Đọc body ngay khi request xong: đợi tới lúc drain thì Chrome có thể đã giải phóng body."""
        try:
            r = req.response()
            if r is not None:
                self._responses.append(r.text())
        except Exception as e:
            self.n_net_failed += 1
            self.log.debug("Không đọc được body %s: %s", req.url[:80], str(e)[:200])

    def _handle_consent(self, page) -> None:
        if "consent." in page.url:
            self.log.info("Gặp trang đồng ý cookie → bấm chấp nhận")
            _try_click(page, self.S["consent_accept"], 5000)
            page.wait_for_load_state("domcontentloaded")

    @staticmethod
    def _is_limited_view(page) -> bool:
        """Google ẩn tab đánh giá và số review khi chưa đăng nhập."""
        try:
            body = page.inner_text("body", timeout=3000)
        except Exception:
            return False
        return "chế độ bị hạn chế" in body or "limited view" in body.lower()

    def _drain_network(self, net_dir: Path) -> int:
        n = 0
        net_dir.mkdir(parents=True, exist_ok=True)
        try:   # body do script trong trang giữ lại
            self._responses.extend(self._page.evaluate("() => (window.__rpcBodies || []).splice(0)"))
        except Exception as e:
            self.log.debug("Không đọc được __rpcBodies: %s", e)
        while self._responses:
            body = self._responses.pop(0)
            idx = len(list(net_dir.glob("*.txt")))
            (net_dir / f"{idx:05d}.txt").write_text(body, encoding="utf-8")
            n += 1
        return n

    # ---- cuộn danh sách review ------------------------------------------
    def _scroll_reviews(self, page, max_reviews: int, net_dir: Path | None) -> int:
        card = self.S["review_card"]
        try:
            page.locator(card).first.wait_for(state="attached", timeout=15000)
        except Exception:
            self.log.warning("Không thấy thẻ review nào (có thể quán không có review, "
                             "hoặc Google đang hiện chế độ 'xem giới hạn' → thử user_data_dir).")
            return 0
        if not page.evaluate(JS_FIND_SCROLLER, card):
            self.log.warning("Không tìm được khung cuộn — sẽ cuộn cả trang.")
        idle, last = 0, -1
        while idle < self.pc["max_idle_rounds"]:
            page.evaluate(JS_SCROLL)
            _pause(self.rng, self.pc["scroll_pause_ms"])
            count = page.evaluate(JS_COUNT, card)
            if net_dir is not None:
                self._drain_network(net_dir)
            if count >= max_reviews:
                break
            idle = idle + 1 if count == last else 0
            last = count
        page.evaluate(JS_EXPAND, self.S["more_button"])
        _pause(self.rng, self.pc["scroll_pause_ms"])
        return page.evaluate(JS_COUNT, card)

    # ---- 1. tìm địa điểm ----------------------------------------------------
    def search_places(self, queries: list[str], max_results: int) -> pd.DataFrame:
        rows = []
        page = self._new_page()
        for q in queries:
            url = with_lang(f"https://www.google.com/maps/search/{q.replace(' ', '+')}", self.pc["lang"])
            self.log.info("Tìm: %s", q)
            page.goto(url)
            self._handle_consent(page)
            _pause(self.rng, (2500, 4000))
            if self._is_limited_view(page):
                raise RuntimeError(
                    "Google Maps đang ở 'chế độ bị hạn chế' (chưa đăng nhập trên www.google.com): "
                    "không có số review/tab đánh giá. Chạy `python -m src.collect.login` "
                    "rồi đặt collect.playwright.user_data_dir trong config.yaml.")
            if "/maps/place/" in page.url:                     # chỉ có 1 kết quả → vào thẳng trang quán
                items = [{"name": page.title().replace(" - Google Maps", ""), "url": page.url,
                          "rating_text": None, "count_text": None}]
            else:
                feed = self.S["results_feed"]
                idle, last = 0, -1
                while idle < 4:
                    page.evaluate(f"() => {{ const f = document.querySelector('{feed}'); if (f) f.scrollTop = f.scrollHeight; }}")
                    _pause(self.rng, self.pc["scroll_pause_ms"])
                    n = page.locator(self.S["result_link"]).count()
                    if n >= max_results:
                        break
                    idle = idle + 1 if n == last else 0
                    last = n
                items = page.evaluate(JS_EXTRACT_RESULTS, self.S)[:max_results]
            for it in items:
                meta = parse_place_url(it["url"])
                rows.append({"query": q, "name": it["name"], "url": it["url"],
                             "rating": _num(it["rating_text"]), "user_rating_count": _num(it["count_text"]),
                             **meta})
            _pause(self.rng, (3000, 6000))
        page.close()
        df = pd.DataFrame(rows)
        if not df.empty:
            df["place_key"] = [place_key(n, u, p) for n, u, p in zip(df["name"], df["url"], df["place_id"])]
            df = df.drop_duplicates("place_key")
        return df

    # ---- 2. lấy review của một địa điểm -----------------------------------
    def scrape_place(self, row: dict) -> dict:
        key = row["place_key"]
        out_file = self.P.raw / "reviews" / f"pw_{key}.jsonl"
        done_file = self.P.raw / "reviews" / f"pw_{key}.done.json"
        if done_file.exists():
            self.log.info("Bỏ qua (đã xong): %s", row.get("name"))
            return json.loads(done_file.read_text(encoding="utf-8"))

        net_dir = (self.P.raw / "network" / key) if self.pc.get("capture_network") else None
        self._responses.clear()
        self.n_net_failed = 0
        page = self._new_page()
        scraped_at = datetime.now(timezone.utc).isoformat()
        page.goto(with_lang(row["url"], self.pc["lang"]))
        self._handle_consent(page)
        _pause(self.rng, (2000, 3500))

        if not _try_click(page, self.S["reviews_tab"], 8000):
            self.log.warning("Không bấm được tab 'Bài đánh giá' cho %s", row.get("name"))
        _pause(self.rng, (1500, 2500))
        if _try_click(page, self.S["sort_button"], 5000):
            _pause(self.rng, (600, 1200))
            if not _try_click(page, self.S["sort_newest"], 4000):
                self.log.warning("Không chọn được 'Mới nhất' — giữ thứ tự mặc định.")
            _pause(self.rng, (1500, 2500))

        n_seen = self._scroll_reviews(page, self.pc["max_reviews_per_place"], net_dir)
        cards = page.evaluate(JS_EXTRACT, self.S)
        if net_dir is not None:
            self._drain_network(net_dir)
        page.close()

        tmp = out_file.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for c in cards:
                rec = {"source": "playwright", "place_key": key, "place_url": row["url"],
                       "scraped_at": scraped_at, **c,
                       "contributor_id": contributor_id_from_url(c.get("profile_url"))}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(out_file)

        stats = {"place_key": key, "name": row.get("name"), "expected": row.get("user_rating_count"),
                 "fetched": len(cards), "seen_in_dom": n_seen, "scraped_at": scraped_at,
                 "net_saved": len(list(net_dir.glob("*.txt"))) if net_dir is not None else 0,
                 "net_failed": self.n_net_failed}
        done_file.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
        self.log.info("%s: lấy %d review (Google báo %s) | gói RPC lưu=%d, lỗi=%d", row.get("name"), len(cards),
                      row.get("user_rating_count"), stats["net_saved"], stats["net_failed"])
        return stats

    # ---- thử lại không phá dữ liệu cũ ----------------------------------------
    def _stash(self, key: str) -> None:
        dest = self.P.raw / "reviews_prev" / key
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for f in (self.P.raw / "reviews").glob(f"pw_{key}.*"):
            shutil.move(str(f), dest / f.name)
        net = self.P.raw / "network" / key
        if net.exists():
            shutil.move(str(net), dest / "network")

    def _restore(self, key: str) -> None:
        src = self.P.raw / "reviews_prev" / key
        for f in (self.P.raw / "reviews").glob(f"pw_{key}.*"):
            f.unlink()
        shutil.rmtree(self.P.raw / "network" / key, ignore_errors=True)
        for f in src.glob("pw_*"):
            shutil.move(str(f), self.P.raw / "reviews" / f.name)
        if (src / "network").exists():
            shutil.move(str(src / "network"), self.P.raw / "network" / key)
        shutil.rmtree(src, ignore_errors=True)

    def _scrape_with_retry(self, row: dict) -> dict:
        """Lấy review; nếu đạt < retry_coverage số Google báo (hoặc lỗi) → thử lại, giữ kết quả tốt nhất."""
        c = self.cfg["collect"]
        key, exp = row["place_key"], row.get("user_rating_count")
        cov = c.get("retry_coverage", c.get("min_coverage", 0.5))
        low = lambda st: bool(exp and exp == exp and exp > 0 and st.get("fetched", 0) < cov * exp)
        done = self.P.raw / "reviews" / f"pw_{key}.done.json"
        best = json.loads(done.read_text(encoding="utf-8")) if done.exists() else None
        if best is not None and not low(best):
            self._skipped = True
            return self.scrape_place(row)                 # đã đủ → scrape_place tự bỏ qua
        self._skipped = False
        for attempt in range(1 + c.get("retry_attempts", 2)):
            if best is not None:
                self._stash(key)                          # cất bản cũ, lần này tệ hơn thì khôi phục
            if attempt:
                self.log.warning("%s: độ phủ thấp (%s/%s) → thử lại (lần %d) sau 60 giây", row.get("name"),
                                 best.get("fetched") if best else 0, int(exp or 0), attempt + 1)
                time.sleep(60)
            try:
                st = self.scrape_place(row)
            except Exception as e:
                self.log.error("Lỗi ở %s (lần %d): %s", row.get("name"), attempt + 1, str(e).splitlines()[0])
                st = None
            if st is not None and (best is None or st.get("fetched", 0) >= best.get("fetched", 0)):
                best = st
                shutil.rmtree(self.P.raw / "reviews_prev" / key, ignore_errors=True)
            elif best is not None:
                self._restore(key)
            if best is not None and not low(best):
                break
        if best is None:
            raise RuntimeError("không lấy được review sau mọi lần thử")
        if low(best):
            self.log.warning("%s: vẫn chỉ %s/%s review sau khi thử lại", row.get("name"), best.get("fetched"), int(exp))
        return best

    def scrape_reviews(self, places: pd.DataFrame) -> pd.DataFrame:
        stats = []
        for i, row in enumerate(places.to_dict("records")):
            try:
                stats.append(self._scrape_with_retry(row))
            except Exception as e:                        # một quán lỗi không làm dừng cả phiên
                self.log.error("Lỗi ở %s: %s", row.get("name"), e)
                stats.append({"place_key": row["place_key"], "name": row.get("name"), "error": str(e)})
            if i < len(places) - 1 and not getattr(self, "_skipped", False):
                time.sleep(self.rng.uniform(*self.pc["delay_between_places_s"]))
        return pd.DataFrame(stats)

    # ---- 3. lịch sử reviewer ------------------------------------------------
    def scrape_contributor(self, contributor_id: str, max_reviews: int) -> int:
        out_file = self.P.raw / "contributors" / f"pw_{contributor_id}.jsonl"
        if out_file.exists():
            return -1
        page = self._new_page()
        scraped_at = datetime.now(timezone.utc).isoformat()
        page.goto(with_lang(f"https://www.google.com/maps/contrib/{contributor_id}/reviews", self.pc["lang"]))
        self._handle_consent(page)
        _pause(self.rng, (2000, 3500))
        self._scroll_reviews(page, max_reviews, None)
        cards = page.evaluate(JS_EXTRACT, self.S)
        page.close()
        with open(out_file, "w", encoding="utf-8") as f:
            for c in cards:
                # Trên trang hồ sơ, trường "name" của thẻ là TÊN ĐỊA ĐIỂM, không phải tên người.
                f.write(json.dumps({"source": "playwright", "contributor_id": contributor_id,
                                    "scraped_at": scraped_at, "place_name": c.get("reviewer_name"),
                                    "rating_label": c.get("rating_label"), "date_text": c.get("date_text"),
                                    "text": c.get("text"), "review_id": c.get("review_id")},
                                   ensure_ascii=False) + "\n")
        return len(cards)


# ---------------------------------------------------------------------------
# Hàm gọi từ run_pipeline.py
# ---------------------------------------------------------------------------
def run_search(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    c = cfg["collect"]
    with MapsScraper(cfg) as s:
        df = s.search_places(c["search_queries"], c["max_results_per_query"])
    out = P.raw / "places" / "places_candidates.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    f = c["place_filter"]
    if not df.empty and df["user_rating_count"].isna().all():
        get_logger("scrape").warning(
            "Không đọc được số review của ứng viên nào → tiêu chí A1 sẽ loại hết. "
            "Kiểm tra selector result_count hoặc đăng nhập (user_data_dir).")
    if not df.empty:
        sel = df[df["user_rating_count"].between(f["min_reviews"], f["max_reviews"])]
        sel.to_csv(P.resolve(c["places_file"]), index=False, encoding="utf-8-sig")
        get_logger("scrape").info("Ứng viên: %d | đạt tiêu chí A1: %d → %s (có thể sửa tay trước khi lấy review)",
                                  len(df), len(sel), c["places_file"])
    return out


def run_reviews(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    places = pd.read_csv(P.resolve(cfg["collect"]["places_file"]))
    if "place_key" not in places:
        places["place_key"] = [place_key(n, u, p) for n, u, p in
                               zip(places["name"], places["url"], places.get("place_id", [None] * len(places)))]
    if places.empty:
        raise RuntimeError(f"{cfg['collect']['places_file']} rỗng: chưa có địa điểm nào đạt tiêu chí A1. "
                           "Chạy lại bước search (sau khi đăng nhập) hoặc điền file này bằng tay.")
    with MapsScraper(cfg) as s:
        stats = s.scrape_reviews(places)
    out = P.tables / "collect_stats_playwright.csv"
    stats.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def run_contributors(cfg: dict, contributor_ids: list[str]) -> int:
    n = 0
    with MapsScraper(cfg) as s:
        for i, cid in enumerate(contributor_ids):
            try:
                n += max(s.scrape_contributor(cid, cfg["collect"]["contributors"]["max_reviews_each"]), 0)
            except Exception as e:
                s.log.error("Lỗi contributor %s: %s", cid, e)
            if i < len(contributor_ids) - 1:
                time.sleep(s.rng.uniform(*s.pc["delay_between_places_s"]) / 2)
    return n
