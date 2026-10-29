"""Đánh giá chất lượng dữ liệu theo 6 chiều (Chương 1) — bảng đưa thẳng vào paper.

Đầu ra: reports/tables/data_quality.csv, reports/tables/coverage_by_place.csv,
        reports/tables/accuracy_check_sample.csv (20 review để đối chiếu tay với Google Maps)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, write_table
from src.utils.config import Paths


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("quality", P.logs)
    parsed = pd.read_parquet(P.interim / "reviews_parsed.parquet")
    clean = pd.read_parquet(P.interim / "reviews_clean.parquet")
    places = pd.read_parquet(P.interim / "places.parquet")
    n = len(parsed)

    rows = []

    def add(dim, check, value, note=""):
        rows.append({"chiều": dim, "kiểm tra": check, "giá trị": value, "ghi chú": note})

    # Completeness
    add("Completeness", "% review không có nội dung (chỉ chấm sao)", round(100 * (~clean["has_text"]).mean(), 2))
    add("Completeness", "% review không xác định được ngày", round(100 * parsed["review_date"].isna().mean(), 2))
    add("Completeness", "% review thiếu số review của tài khoản", round(100 * parsed["reviewer_n_reviews"].isna().mean(), 2))
    if "coverage" in places:
        cov = places["coverage"].dropna()
        add("Completeness", "Độ phủ thu thập trung bình (fetched/expected)", round(cov.mean(), 3) if len(cov) else None,
            f"min={cov.min():.2f}, max={cov.max():.2f}" if len(cov) else "")
        if "excluded_low_coverage" in places:
            ex = places[places["excluded_low_coverage"]]
            add("Completeness", "Số địa điểm bị loại vì độ phủ thấp", len(ex),
                "; ".join(f"{getattr(r, 'name', r.place_key)} ({r.fetched:.0f}/{r.expected:.0f})" for r in ex.itertuples()))
    # Uniqueness
    add("Uniqueness", "Số review_id trùng (trước làm sạch)", int(parsed["review_id"].duplicated().sum()))
    add("Uniqueness", "Số nội dung trùng y hệt (≥8 từ, sau làm sạch)",
        int(clean.loc[clean["n_words"] >= 8, "text_norm"].duplicated(keep=False).sum()),
        "tín hiệu spam tiềm năng — không xóa")
    # Validity
    add("Validity", "Số rating ngoài 1..5", int((~parsed["rating"].between(1, 5)).sum()))
    fut = parsed["review_date"] > parsed["scraped_at"] + pd.Timedelta(days=1)
    add("Validity", "Số review có ngày trong tương lai", int(fut.sum()))
    # Consistency
    if "rating" in places.columns:
        m = clean.groupby("place_key")["rating"].mean().rename("rating_corpus")
        cmp = places.set_index("place_key")[["rating"]].join(m, how="inner").dropna()
        if len(cmp):
            diff = (cmp["rating"] - cmp["rating_corpus"]).abs()
            add("Consistency", "|điểm TB Google báo − điểm TB tính lại| (trung bình)", round(diff.mean(), 3),
                f"max={diff.max():.2f}")
    # Timeliness
    last = clean["scraped_at"].max()
    add("Timeliness", "% review trong 24 tháng gần nhất",
        round(100 * (clean["review_date"] >= last - pd.Timedelta(days=730)).mean(), 2))
    prec = clean["date_precision_days"]
    add("Timeliness", "% review có ngày chính xác tới ngày (sai số ≤ 1 ngày)", round(100 * (prec <= 1).mean(), 2),
        "quyết định dữ liệu dùng được cho Module Burst")
    add("Timeliness", "Phân bố nguồn ngày", clean["date_source"].value_counts().to_dict())
    # Accuracy
    add("Accuracy", "Đối chiếu tay 20 review ngẫu nhiên với Google Maps", "xem accuracy_check_sample.csv",
        "điền cột 'khớp' sau khi kiểm tra")

    write_table(pd.DataFrame(rows), P.tables / "data_quality.csv")
    cols = [c for c in ["place_key", "name", "user_rating_count", "expected", "fetched", "coverage",
                        "excluded_low_coverage"] if c in places]
    write_table(places[cols], P.tables / "coverage_by_place.csv")
    samp = clean.sample(min(20, len(clean)), random_state=cfg["project"]["seed"])
    samp = samp.merge(places[[c for c in ["place_key", "name", "url"] if c in places]], on="place_key", how="left")
    samp = samp[[c for c in ["review_id", "name", "url", "rating", "date_text", "text"] if c in samp]].assign(khop="")
    write_table(samp, P.tables / "accuracy_check_sample.csv")
    log.info("Đã ghi bảng chất lượng dữ liệu (%d chỉ số) — %d review thô", len(rows), n)
    return P.tables / "data_quality.csv"
