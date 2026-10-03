"""Đổi ngày tương đối của Google Maps ("3 tháng trước", "a week ago") thành khoảng ngày.

Google làm tròn tuổi review xuống đơn vị lớn nhất, nên "2 tháng trước" nghĩa là
tuổi nằm trong [2 tháng, 3 tháng). Hàm trả về:
    date_lo, date_hi  : khoảng chứa ngày đăng thật
    date_est          : điểm giữa khoảng
    precision_days    : độ rộng khoảng (≈ sai số tối đa)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

UNIT_DAYS = {
    "minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30.44, "year": 365.25,
}

_UNITS = [
    # (regex tiếng Việt / tiếng Anh, đơn vị)
    (r"phút|minutes?|mins?", "minute"),
    (r"giờ|tiếng|hours?", "hour"),
    (r"ngày|days?", "day"),
    (r"tuần|weeks?", "week"),
    (r"tháng|months?", "month"),
    (r"năm|years?", "year"),
]
_NUM_WORDS = {"một": 1, "a": 1, "an": 1, "one": 1, "hai": 2, "two": 2, "ba": 3, "three": 3}

_PAT = re.compile(
    r"(?P<num>\d+|một|hai|ba|a|an|one|two|three)\s*(?P<unit>" +
    "|".join(p for p, _ in _UNITS) + r")\s*(?:trước|ago)", re.IGNORECASE)

_EDITED = re.compile(r"^(đã chỉnh sửa|thời gian chỉnh sửa|edited)\s*:?\s*", re.IGNORECASE)
_SUFFIX = re.compile(r"\s*(trên|on)\s+\w+.*$", re.IGNORECASE)   # "... trên Google"


@dataclass
class RelDate:
    date_lo: datetime | None
    date_hi: datetime | None
    date_est: datetime | None
    precision_days: float | None
    is_edited: bool
    parsed: bool


def parse_relative_date(text: str | None, scraped_at: datetime) -> RelDate:
    if not text:
        return RelDate(None, None, None, None, False, False)
    t = unicodedata.normalize("NFC", str(text)).strip().lower()
    edited = bool(_EDITED.search(t))
    t = _EDITED.sub("", t)
    t = _SUFFIX.sub("", t).strip()

    if t in {"vừa xong", "just now", "mới đây"}:
        return RelDate(scraped_at - timedelta(hours=1), scraped_at, scraped_at, 1 / 24, edited, True)
    if t in {"hôm qua", "yesterday"}:
        t = "1 ngày trước"

    m = _PAT.search(t)
    if not m:
        return RelDate(None, None, None, None, edited, False)
    raw_num = m.group("num").lower()
    n = int(raw_num) if raw_num.isdigit() else _NUM_WORDS.get(raw_num, 1)
    unit_txt = m.group("unit").lower()
    unit = next(u for p, u in _UNITS if re.fullmatch(p, unit_txt, re.IGNORECASE))
    d = UNIT_DAYS[unit]
    hi = scraped_at - timedelta(days=n * d)
    lo = scraped_at - timedelta(days=(n + 1) * d)
    est = lo + (hi - lo) / 2
    return RelDate(lo, hi, est, d, edited, True)
