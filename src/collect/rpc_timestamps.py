"""Trích (review_id -> timestamp) từ các gói RPC mà trang Google Maps tự tải.

Khi cuộn danh sách review, trang gọi các endpoint nội bộ (URL chứa `listugcposts`).
Body là JSON có tiền tố chống XSSI `)]}'` và cấu trúc mảng lồng nhau không được công bố.
Vì cấu trúc có thể thay đổi, ta KHÔNG dựa vào chỉ số cố định mà dùng heuristic:

  - review_id: chuỗi base64-url dài, thường bắt đầu bằng "Ch" hoặc "Ci"
  - timestamp: số nguyên micro-giây trong khoảng năm 2005–2040

Duyệt cây: một nút là "bản ghi review" nếu cây con của nó chứa đúng MỘT review_id
và ít nhất một timestamp. Chọn nút như vậy ở mức cao nhất, lấy timestamp nhỏ nhất
(thời điểm đăng; các mốc lớn hơn thường là chỉnh sửa / phản hồi của chủ quán).

Kết quả KHÔNG được tin tuyệt đối: parse_raw.py chỉ dùng timestamp nếu nó nằm trong
khoảng ngày suy ra từ ngày tương đối hiển thị trên giao diện.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REVIEW_ID_RE = re.compile(r"^C[hi][A-Za-z0-9_\-]{18,}$")
TS_MIN, TS_MAX = 1_104_537_600_000_000, 2_208_988_800_000_000   # 2005-01-01 .. 2040-01-01 (µs)


def _expand_json_strings(node):
    """batchexecute bọc payload thật thành CHUỖI JSON bên trong mảng ["wrb.fr", rpcid, "<json>"]."""
    if isinstance(node, str):
        s = node.lstrip()
        if len(s) > 1 and s[0] in "[{":
            try:
                return _expand_json_strings(json.loads(s))
            except json.JSONDecodeError:
                return node
        return node
    if isinstance(node, list):
        return [_expand_json_strings(c) for c in node]
    if isinstance(node, dict):
        return {k: _expand_json_strings(v) for k, v in node.items()}
    return node


def _load_body(text: str):
    text = text.lstrip()
    if text.startswith(")]}'"):
        text = text[4:]
    try:
        return _expand_json_strings(json.loads(text))
    except json.JSONDecodeError:
        pass
    # Định dạng batchexecute (từ 2026): nhiều chunk "<độ dài>\n<json>" nối tiếp nhau
    dec, pos, chunks = json.JSONDecoder(), 0, []
    while pos < len(text):
        m = re.compile(r"\s*\d+\s*\n").match(text, pos)
        if m:
            pos = m.end()
        try:
            obj, pos = dec.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        chunks.append(obj)
    return _expand_json_strings(chunks) if chunks else None


def _is_review_id(s: str, known: set | None) -> bool:
    if known is not None:                      # chính xác: chỉ nhận id đã thấy trên DOM
        return s in known
    return bool(REVIEW_ID_RE.match(s)) and not s.startswith("ChIJ")   # ChIJ... là place_id


def _walk(node, out: dict, known: set | None = None):
    """Trả về (tập review_id, danh sách timestamp) của cây con; ghi bản ghi vào out."""
    if isinstance(node, str):
        return ({node} if _is_review_id(node, known) else set()), []
    if isinstance(node, bool) or node is None:
        return set(), []
    if isinstance(node, (int, float)):
        v = int(node)
        return set(), ([v] if TS_MIN <= v <= TS_MAX and float(node).is_integer() else [])
    if isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return set(), []

    ids: set = set()
    tss: list = []
    child_results = []
    for c in children:
        ci, ct = _walk(c, out, known)
        child_results.append((ci, ct))
        ids |= ci
        tss.extend(ct)

    # Nếu nút hiện tại chứa nhiều id, các con có đúng 1 id là bản ghi review cấp cao nhất.
    if len(ids) > 1:
        for ci, ct in child_results:
            if len(ci) == 1 and ct:
                rid = next(iter(ci))
                out[rid] = min(ct) if rid not in out else min(out[rid], min(ct))
    return ids, tss


def extract_from_body(text: str, known_ids: set | None = None) -> dict[str, datetime]:
    data = _load_body(text)
    if data is None:
        return {}
    found: dict[str, int] = {}
    ids, tss = _walk(data, found, known_ids)
    if len(ids) == 1 and tss:                       # body chỉ có đúng 1 review
        found[next(iter(ids))] = min(tss)
    return {rid: datetime.fromtimestamp(us / 1e6, tz=timezone.utc) for rid, us in found.items()}


def extract_from_dir(net_dir: Path, known_ids: set | None = None) -> dict[str, datetime]:
    out: dict[str, datetime] = {}
    if not net_dir.exists():
        return out
    for p in sorted(net_dir.rglob("*.txt")):
        for rid, ts in extract_from_body(p.read_text(encoding="utf-8", errors="ignore"), known_ids).items():
            if rid not in out or ts < out[rid]:
                out[rid] = ts
    return out
