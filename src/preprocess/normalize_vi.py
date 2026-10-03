"""Chuẩn hóa văn bản review tiếng Việt (B3).

Thứ tự: đếm đặc trưng "thô" (URL, SĐT, emoji, chữ in hoa) TRƯỚC khi xóa chúng,
vì chính các dấu hiệu này là tín hiệu spam quảng cáo.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from src.utils.config import PROJECT_ROOT

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|vn|net|org|shop|link|me)(?:/\S*)?)", re.I)
PHONE_RE = re.compile(r"(?:\+?84|0)(?:[\s.\-]?\d){8,10}\b")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F900-\U0001F9FF]")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")              # "ngonnnn" -> "ngon" (không đụng tới số: 1000)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")

VI_CHARS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
EN_STOP = {"the", "and", "is", "was", "very", "good", "great", "food", "place", "service", "staff", "nice",
           "with", "for", "but", "not", "this", "it", "we", "my", "friendly", "delicious", "coffee"}
VI_NODIAC_STOP = {"ngon", "quan", "nhan", "vien", "phuc", "vu", "khong", "gia", "rat", "binh", "thuong", "dep",
                  "sach", "se", "quay", "lai", "mon", "an", "uong", "tot"}


@lru_cache(maxsize=4)
def load_teencode(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#") or "\t" not in line:
                continue
            k, v = line.split("\t", 1)
            out[unicodedata.normalize("NFC", k.strip().lower())] = v.strip()
    return out


def detect_lang(text: str) -> str:
    if not text or not text.strip():
        return "none"
    t = text.lower()
    if any(ch in VI_CHARS for ch in t):
        return "vi"
    toks = re.findall(r"[a-z]+", t)
    if not toks:
        return "other"
    en = sum(w in EN_STOP for w in toks) / len(toks)
    vi = sum(w in VI_NODIAC_STOP for w in toks) / len(toks)
    if vi >= en and vi >= 0.15:
        return "vi"            # tiếng Việt không dấu
    if en >= 0.15:
        return "en"
    return "other"


def raw_features(text: str) -> dict:
    t = text or ""
    letters = [c for c in t if c.isalpha()]
    return {
        "n_chars": len(t),
        "n_urls": len(URL_RE.findall(t)),
        "n_phones": len(PHONE_RE.findall(t)),
        "n_emojis": len(EMOJI_RE.findall(t)),
        "n_exclaim": t.count("!"),
        "upper_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
    }


def normalize(text: str, teencode: dict[str, str]) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).lower()
    t = URL_RE.sub(" ", t)
    t = PHONE_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = REPEAT_RE.sub(r"\1", t)
    t = NON_WORD_RE.sub(" ", t)
    toks = [teencode.get(w, w) for w in t.split()]
    return SPACE_RE.sub(" ", " ".join(toks)).strip()


_SEGMENTER = None


def segment(text: str, method: str = "pyvi") -> str:
    """Tách từ (cho PhoBERT). Thiếu thư viện thì trả về nguyên văn."""
    global _SEGMENTER
    if not text or method == "none":
        return text or ""
    if _SEGMENTER is None:
        try:
            from pyvi import ViTokenizer
            _SEGMENTER = ViTokenizer.tokenize
        except Exception:
            _SEGMENTER = lambda s: s  # noqa: E731
    return _SEGMENTER(text)
