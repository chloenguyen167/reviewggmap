"""Tách place_id, data_id, tọa độ từ URL Google Maps.

Ví dụ URL: https://www.google.com/maps/place/Ten+Quan/@10.77,106.70,17z/data=
           !4m7!3m6!1s0x31752f...:0x1a2b...!8m2!3d10.7769!4d106.7009!16s%2Fg%2F11...!19sChIJ...
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import unquote, urlencode, urlparse, parse_qsl, urlunparse

_RE_PLACE_ID = re.compile(r"!19s(ChIJ[\w\-]+)")
_RE_DATA_ID = re.compile(r"!1s(0x[0-9a-fA-F]+:0x[0-9a-fA-F]+)")
_RE_3D4D = re.compile(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)")
_RE_AT = re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)")
_RE_CONTRIB = re.compile(r"/contrib/(\d+)")


def parse_place_url(url: str) -> dict:
    u = unquote(url or "")
    out = {"place_id": None, "data_id": None, "lat": None, "lng": None}
    if m := _RE_PLACE_ID.search(u):
        out["place_id"] = m.group(1)
    if m := _RE_DATA_ID.search(u):
        out["data_id"] = m.group(1)
    if m := _RE_3D4D.search(u):
        out["lat"], out["lng"] = float(m.group(1)), float(m.group(2))
    elif m := _RE_AT.search(u):
        out["lat"], out["lng"] = float(m.group(1)), float(m.group(2))
    return out


def contributor_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    m = _RE_CONTRIB.search(url)
    return m.group(1) if m else None


def with_lang(url: str, lang: str = "vi") -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query))
    q["hl"] = lang
    return urlunparse(p._replace(query=urlencode(q)))


def place_key(name: str | None, url: str | None, place_id: str | None = None) -> str:
    """Khóa ổn định, an toàn làm tên file cho một địa điểm."""
    if place_id:
        return place_id
    base = unicodedata.normalize("NFKD", name or "place").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-").lower()[:40] or "place"
    h = hashlib.md5((url or name or "").encode()).hexdigest()[:8]
    return f"{slug}-{h}"
