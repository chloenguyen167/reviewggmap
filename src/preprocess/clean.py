"""Làm sạch (B2) + chuẩn hóa văn bản (B3).

Đầu vào : interim/reviews_parsed.parquet
Đầu ra  : interim/reviews_clean.parquet, reports/tables/cleaning_log.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess.normalize_vi import detect_lang, load_teencode, normalize, raw_features, segment
from src.utils.common import get_logger, write_table
from src.utils.config import Paths


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("clean", P.logs)
    tz = cfg["project"]["timezone"]
    df = pd.read_parquet(P.interim / "reviews_parsed.parquet")
    steps = [("đầu vào", len(df))]

    # 1. Trùng kỹ thuật (do phân trang / cuộn lặp) — KHÁC trùng nội dung (tín hiệu spam, giữ lại)
    df = df.sort_values("scraped_at").drop_duplicates("review_id", keep="last")
    steps.append(("bỏ trùng review_id", len(df)))
    df = df.drop_duplicates(["place_key", "reviewer_hash", "text", "date_text"], keep="last")
    steps.append(("bỏ trùng (quán, reviewer, nội dung, ngày)", len(df)))

    # 2. Validity
    df = df[df["rating"].between(1, 5)]
    steps.append(("bỏ rating ngoài 1..5", len(df)))
    if cfg["preprocess"].get("drop_future_dates", True):
        future = df["review_date"].notna() & (df["review_date"] > df["scraped_at"] + pd.Timedelta(days=1))
        df = df[~future]
        steps.append(("bỏ ngày trong tương lai", len(df)))

    # 3. Thời gian theo giờ VN
    local = df["review_date"].dt.tz_convert(tz)
    df["review_day"] = local.dt.tz_localize(None).dt.normalize()
    df["review_week"] = (df["review_day"] - pd.to_timedelta(df["review_day"].dt.dayofweek, unit="D"))

    # 4. Văn bản
    teen = load_teencode(cfg["preprocess"]["teencode_file"])
    df["text"] = df["text"].fillna("").astype(str)
    feats = pd.DataFrame([raw_features(t) for t in df["text"]], index=df.index)
    df = pd.concat([df, feats], axis=1)
    df["lang"] = [detect_lang(t) for t in df["text"]]
    df["text_norm"] = [normalize(t, teen) for t in df["text"]]
    df["n_words"] = df["text_norm"].str.split().str.len().fillna(0).astype(int)
    df["has_text"] = df["n_words"] > 0
    seg = cfg["preprocess"].get("word_segment", "pyvi")
    df["text_seg"] = [segment(t, seg) if lang == "vi" else t for t, lang in zip(df["text_norm"], df["lang"])]

    df = df.reset_index(drop=True)
    out = P.interim / "reviews_clean.parquet"
    df.to_parquet(out, index=False)
    write_table(pd.DataFrame(steps, columns=["bước", "số dòng còn lại"]), P.tables / "cleaning_log.csv")
    log.info("Làm sạch: %s", " → ".join(f"{n}" for _, n in steps))
    return out
