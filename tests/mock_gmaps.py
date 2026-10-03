"""Trang Google Maps GIẢ LẬP + web server cục bộ để kiểm thử scraper Playwright.

Mô phỏng đúng các hành vi scraper dựa vào:
  - tab "Bài đánh giá" (ẩn danh sách cho tới khi bấm)
  - nút "Sắp xếp" → menu có mục "Mới nhất"
  - khung cuộn riêng (overflow-y:auto), cuộn tới đáy thì tải thêm 10 thẻ
  - mỗi lần tải thêm, trang gọi fetch('/maps/rpc/listugcposts?...') trả về JSON có tiền tố )]}'
  - nội dung dài bị cắt "…" kèm nút "Thêm" (button.w8nwRe) — bấm mới hiện đủ
Class name dùng giống config.yaml (jftiEf, d4r55, RfnDt, kvMYJc, rsqaWe, wiI7pd, w8nwRe, CDe7pd).
Việc tương thích với Google Maps THẬT vẫn phải thử trên máy có mạng (Phase 0).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

N_REVIEWS = 45
NOW = datetime.now(timezone.utc)


def make_reviews():
    out = []
    for i in range(N_REVIEWS):
        age = timedelta(days=i * 9 + 1, hours=3)
        created = NOW - age
        d = age.days
        rel = (f"{d} ngày trước" if d < 7 else f"{d // 7} tuần trước" if d < 30 else
               f"{int(d // 30.44)} tháng trước" if d < 365 else f"{int(d // 365.25)} năm trước")
        rel = rel.replace("1 ngày", "một ngày").replace("1 tuần", "một tuần").replace("1 tháng", "một tháng")
        long_text = f"Review số {i}: đồ uống ngon, nhân viên thân thiện. " + ("Chi tiết thêm. " * (6 if i % 3 == 0 else 0))
        out.append({
            "id": f"ChZDSUhNMG9nS0VJQ0FnSU{i:08d}AAAA", "name": f"Người dùng {i}", "cid": str(10 ** 20 + i),
            "meta": ("Local Guide · " if i % 4 == 0 else "") + f"{i + 1} bài đánh giá",
            "stars": 5 - (i % 5), "rel": rel, "text": long_text.strip(), "ts": int(created.timestamp() * 1e6),
            "owner": "Cảm ơn bạn!" if i % 7 == 0 else None,
        })
    return out


REVIEWS = make_reviews()

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Quán Test - Google Maps</title>
<style> #panel{height:420px;overflow-y:auto;border:1px solid #ccc} .jftiEf{height:120px;border-bottom:1px solid #eee}
.hidden{display:none} </style></head><body>
<button role="tab" id="tab-overview">Tổng quan</button>
<button role="tab" id="tab-reviews" onclick="document.getElementById('wrap').classList.remove('hidden')">Bài đánh giá</button>
<div id="wrap" class="hidden">
  <button aria-label="Sắp xếp bài đánh giá" onclick="document.getElementById('menu').classList.remove('hidden')">Sắp xếp</button>
  <div id="menu" class="hidden">
    <div role="menuitemradio">Phù hợp nhất</div>
    <div role="menuitemradio" onclick="window.sorted=true;document.getElementById('menu').classList.add('hidden')">Mới nhất</div>
  </div>
  <div id="panel"><div id="list"></div></div>
</div>
<script>
const DATA = __DATA__;
let loaded = 0;
function card(r){
  const short = r.text.length > 60 ? r.text.slice(0, 60) + '…' : r.text;
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
  const more = r.text.length > 60 ? `<button class="w8nwRe" aria-label="Xem thêm" data-full="${esc(r.text)}" onclick="this.previousElementSibling.innerText=this.dataset.full;this.remove()">Thêm</button>` : '';
  const owner = r.owner ? `<div class="CDe7pd">Phản hồi của chủ sở hữu ${r.owner}</div>` : '';
  return `<div class="jftiEf" data-review-id="${r.id}" aria-label="${r.name}">
     <button data-href="https://www.google.com/maps/contrib/${r.cid}/reviews"><div class="d4r55">${r.name}</div></button>
     <div class="RfnDt">${r.meta}</div>
     <span class="kvMYJc" role="img" aria-label="${r.stars} sao"></span><span class="rsqaWe">${r.rel}</span>
     <div data-review-id="${r.id}"><span class="wiI7pd">${short}</span>${more}</div>${owner}</div>`;
}
function loadMore(){
  if (loaded >= DATA.length) return;
  const page = DATA.slice(loaded, loaded + 10);
  fetch('/maps/rpc/listugcposts?start=' + loaded).then(r => r.text());
  document.getElementById('list').insertAdjacentHTML('beforeend', page.map(card).join(''));
  loaded += page.length;
}
document.getElementById('tab-reviews').addEventListener('click', () => { if (!loaded) loadMore(); });
document.getElementById('panel').addEventListener('scroll', (e) => {
  const el = e.target; if (el.scrollTop + el.clientHeight >= el.scrollHeight - 5) setTimeout(loadMore, 150);
});
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/place"):
            body = PAGE.replace("__DATA__", json.dumps(REVIEWS, ensure_ascii=False)).encode("utf-8")
            ctype = "text/html; charset=utf-8"
        elif "listugcposts" in u.path:
            start = int(parse_qs(u.query).get("start", ["0"])[0])
            recs = [[r["id"], [r["name"], None, [r["ts"]]], r["text"]] for r in REVIEWS[start:start + 10]]
            body = (")]}'\n" + json.dumps([None, recs, "ChIJmockplace"], ensure_ascii=False)).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
