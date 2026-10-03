"""EDA (B4) — các hình mô tả dữ liệu cho paper.

fig_eda_rating.png          phân bố số sao (thường lệch về 5★ — hình chữ J)
fig_eda_length.png          phân bố số từ / review (thang log)
fig_eda_daily_top3.png      số review/ngày của 3 quán nhiều review nhất (nhìn thấy burst bằng mắt)
fig_eda_account_size.png    phân bố số review của tài khoản
eda_summary.csv             số liệu tóm tắt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import plotstyle as ps
from src.utils.common import get_logger, write_table
from src.utils.config import Paths


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("eda", P.logs)
    ps.apply()
    df = pd.read_parquet(P.interim / "reviews_clean.parquet")
    places = pd.read_parquet(P.interim / "places.parquet")
    names = places.set_index("place_key")["name"].to_dict() if "name" in places else {}

    # 1. Rating
    fig, ax = plt.subplots(figsize=(5, 3))
    cnt = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    ax.bar([f"{int(i)}★" for i in cnt.index], cnt.values / cnt.sum() * 100, color=ps.SERIES[0], width=0.6)
    ax.set_ylabel("% review")
    ax.set_title("Phân bố số sao")
    ax.grid(axis="x", visible=False)
    ps.save(fig, P.figures / "fig_eda_rating.png")

    # 2. Độ dài
    fig, ax = plt.subplots(figsize=(5, 3))
    w = df.loc[df["has_text"], "n_words"]
    bins = np.unique(np.logspace(0, np.log10(max(w.max(), 10)), 30).astype(int))
    ax.hist(w, bins=bins, color=ps.SERIES[0], edgecolor=ps.SURFACE, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Số từ / review (thang log)")
    ax.set_ylabel("Số review")
    ax.set_title(f"Độ dài review ({(~df['has_text']).mean():.0%} chỉ chấm sao, không có chữ)")
    ps.save(fig, P.figures / "fig_eda_length.png")

    # 3. Chuỗi theo ngày của 3 quán nhiều review nhất (chỉ review có ngày chính xác)
    exact = df[df["date_precision_days"] <= 1]
    top = exact["place_key"].value_counts().head(3).index.tolist()
    if top:
        fig, axes = plt.subplots(len(top), 1, figsize=(7, 1.9 * len(top)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, top):
            s = exact[exact["place_key"] == key].groupby("review_day").size()
            s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0)
            ax.plot(s.index, s.values, color=ps.SERIES[0], linewidth=1.2)
            ax.set_title(str(names.get(key, key))[:50], fontsize=9, loc="left")
            ax.set_ylabel("review/ngày")
        fig.suptitle("Số review mỗi ngày — 3 địa điểm nhiều review nhất", fontsize=11, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout()
        ps.save(fig, P.figures / "fig_eda_daily_top3.png")

    # 4. Quy mô tài khoản
    acc = df.drop_duplicates("reviewer_hash")["reviewer_n_reviews"].dropna()
    if len(acc):
        fig, ax = plt.subplots(figsize=(5, 3))
        edges = [0, 1, 3, 10, 50, 200, np.inf]
        labels = ["1", "2–3", "4–10", "11–50", "51–200", ">200"]
        b = pd.cut(acc, edges, labels=labels).value_counts().reindex(labels)
        ax.bar(labels, b.values / b.sum() * 100, color=ps.SERIES[0], width=0.6)
        ax.set_xlabel("Tổng số review của tài khoản")
        ax.set_ylabel("% tài khoản")
        ax.set_title("Quy mô tài khoản reviewer")
        ax.grid(axis="x", visible=False)
        ps.save(fig, P.figures / "fig_eda_account_size.png")

    summary = pd.DataFrame([
        ("Số địa điểm", df["place_key"].nunique()),
        ("Số review", len(df)),
        ("Số reviewer (hash)", df["reviewer_hash"].nunique()),
        ("Khoảng thời gian", f"{df['review_day'].min():%Y-%m-%d} → {df['review_day'].max():%Y-%m-%d}"),
        ("% review 5★", round(100 * (df["rating"] == 5).mean(), 1)),
        ("% review 1★", round(100 * (df["rating"] == 1).mean(), 1)),
        ("% chỉ chấm sao", round(100 * (~df["has_text"]).mean(), 1)),
        ("Số từ trung vị (có chữ)", float(df.loc[df["has_text"], "n_words"].median())),
        ("% tiếng Việt / tiếng Anh", f"{(df['lang'] == 'vi').mean():.0%} / {(df['lang'] == 'en').mean():.0%}"),
        ("% tài khoản ≤ 3 review", round(100 * (acc <= 3).mean(), 1) if len(acc) else None),
        ("% ngày chính xác (≤1 ngày)", round(100 * (df["date_precision_days"] <= 1).mean(), 1)),
    ], columns=["chỉ số", "giá trị"])
    write_table(summary, P.tables / "eda_summary.csv")
    log.info("EDA: %d hình, bảng tóm tắt %d dòng", len(list(P.figures.glob("fig_eda_*.png"))), len(summary))
    return P.tables / "eda_summary.csv"
