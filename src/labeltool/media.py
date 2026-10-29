"""Trích ảnh / video đính kèm của từng review từ các gói RPC đã lưu (raw/network).

DOM chỉ đếm số ảnh thu nhỏ đang hiển thị (thường tối đa 4) nên không có URL ảnh. Gói RPC chứa
đầy đủ: mỗi review có một danh sách media, mỗi phần tử có ảnh (hoặc ảnh bìa của video) dạng
`https://lh3.googleusercontent.com/grass-cs/<id>` và, nếu là video, luồng `gpms-cs-s/<id>=m18` (mp4).
Ảnh đại diện (`/a/`, `/a-/`) và huy hiệu (`local-reviews/`) bị bỏ qua.

URL lưu ở dạng gốc (bỏ phần `=w..-h..`) để giao diện tự chọn kích thước khi hiển thị.
Đầu ra: interim/review_media.parquet (review_id, idx, photo, video)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.collect.rpc_timestamps import _load_body
from src.utils.common import get_logger
from src.utils.config import Paths

HOST = "googleusercontent.com/"
SKIP_KINDS = {"a", "a-", "local-reviews"}


def _kind(u: str) -> str | None:
    if HOST not in u:
        return None
    k = u.split(HOST, 1)[1].split("/", 1)[0]
    return None if k in SKIP_KINDS else k


def _base(u: str) -> str:
    return u.split("=", 1)[0]


def _strings(node, out: list):
    if isinstance(node, str):
        if _kind(node):
            out.append(node)
    elif isinstance(node, list):
        for c in node:
            _strings(c, out)
    elif isinstance(node, dict):
        for c in node.values():
            _strings(c, out)
    return out


def _media_items(record) -> list[tuple[str, str | None]]:
    """[(ảnh, video|None)] theo thứ tự hiển thị trong một bản ghi review."""
    items: list[tuple[str, str | None]] = []
    seen: set = set()

    def visit(n):
        if not isinstance(n, list):
            return
        # phần tử media: [id, [..., ..., ..., ..., ..., ..., [<url ảnh>, ...], ...], ...]
        if (len(n) > 1 and isinstance(n[1], list) and len(n[1]) > 6 and isinstance(n[1][6], list)
                and n[1][6] and isinstance(n[1][6][0], str) and _kind(n[1][6][0]) not in (None, "gpms-cs-s")):
            photo = _base(n[1][6][0])
            vids = [s for s in _strings(n, []) if _kind(s) == "gpms-cs-s"]
            mp4 = next((s for s in vids if s.endswith("=m18")), vids[0] if vids else None)
            if photo not in seen:
                seen.add(photo)
                items.append((photo, mp4))
            return
        for c in n:
            visit(c)

    visit(record)
    if not items:                                  # cấu trúc đổi → lấy mọi URL ảnh theo thứ tự
        for s in _strings(record, []):
            if _kind(s) != "gpms-cs-s" and _base(s) not in seen:
                seen.add(_base(s))
                items.append((_base(s), None))
    return items


def _records(node, known: set, out: dict):
    """Giống rpc_timestamps._walk: bản ghi review = cây con chứa đúng MỘT review_id đã biết."""
    if isinstance(node, str):
        return {node} if node in known else set()
    if not isinstance(node, list):
        return set()
    ids: set = set()
    child = []
    for c in node:
        ci = _records(c, known, out)
        child.append((c, ci))
        ids |= ci
    if len(ids) > 1:
        for c, ci in child:
            if len(ci) == 1:
                rid = next(iter(ci))
                items = _media_items(c)
                if len(items) > len(out.get(rid, [])):
                    out[rid] = items
    return ids


def extract(net_dir: Path, known: set) -> dict[str, list[tuple[str, str | None]]]:
    out: dict = {}
    for p in sorted(net_dir.rglob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if HOST not in text:
            continue
        data = _load_body(text)
        if data is None:
            continue
        ids = _records(data, known, out)
        if len(ids) == 1:                          # body chỉ có một review
            rid = next(iter(ids))
            items = _media_items(data)
            if len(items) > len(out.get(rid, [])):
                out[rid] = items
    return out


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("media", P.logs)
    df = pd.read_parquet(P.interim / "reviews_clean.parquet", columns=["review_id", "n_photos"])
    found = extract(P.raw / "network", set(df["review_id"]))
    rows = [{"review_id": rid, "idx": i, "photo": ph, "video": v}
            for rid, items in found.items() for i, (ph, v) in enumerate(items)]
    media = pd.DataFrame(rows, columns=["review_id", "idx", "photo", "video"])
    out = P.interim / "review_media.parquet"
    media.to_parquet(out, index=False)
    with_dom = set(df.loc[df["n_photos"] > 0, "review_id"])
    log.info("Media: %d ảnh/video của %d review (%d video) | review có ảnh trên giao diện: %d, tìm được URL: %d (%.1f%%)",
             len(media), media["review_id"].nunique(), int(media["video"].notna().sum()), len(with_dom),
             len(with_dom & set(found)), 100 * len(with_dom & set(found)) / max(len(with_dom), 1))
    return out
