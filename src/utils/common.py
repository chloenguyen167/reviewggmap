"""Tiện ích dùng chung: logging, đọc/ghi, hash ẩn danh, percentile."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_salt(cfg: dict) -> str:
    env = cfg.get("project", {}).get("salt_env", "REVIEWER_SALT")
    salt = os.environ.get(env)
    if not salt:
        # Không dừng chương trình, nhưng cảnh báo: salt mặc định chỉ dùng cho demo.
        get_logger("hashing").warning(
            "Chưa đặt biến môi trường %s — dùng salt mặc định (CHỈ phù hợp cho demo).", env)
        salt = "demo-salt-change-me"
    return salt


def hash_id(value, salt: str, length: int = 16) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return None
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:length]


def pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank trong [0,1] trên các giá trị không NaN.

    Giá trị <= 0 (không có tín hiệu) được gán 0 thay vì hạng trung bình của nhóm
    bằng nhau — tránh việc một review không trùng lặp với ai lại có s_dup = 0.45.
    """
    out = s.rank(method="max", pct=True)
    out[s <= 0] = 0.0
    return out


def one_sided(X: pd.DataFrame) -> pd.DataFrame:
    """Cắt một phía: giá trị nhỏ hơn median bị nâng lên median.

    Các cột phải được định hướng trước sao cho "lớn = đáng ngờ". Sau phép cắt này,
    Isolation Forest chỉ còn cô lập được các điểm lệch về phía đáng ngờ
    (một Local Guide có 5.000 review không còn bị coi là "bất thường").
    """
    X = X.astype(float)
    med = X.median()
    return X.clip(lower=med, axis=1)


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
