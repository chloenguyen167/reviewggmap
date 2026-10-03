"""Tạo tập mẫu để gán nhãn thủ công (D2).

Nhóm A: top-N theo TỪNG điểm (s_dup, s_content, s_burst, s_behav, suspicion) → hợp lại.
Nhóm B: N review lấy NGẪU NHIÊN ĐỀU từ toàn corpus (có thể trùng với A — giữ đúng tính
        ngẫu nhiên để ước lượng tỷ lệ spam nền và FPR).

Đầu ra (data/labels/):
  gold_sample_blind.csv  — cho người gán nhãn: KHÔNG có điểm số, KHÔNG biết thuộc nhóm nào,
                           thứ tự đã xáo trộn
  annotator_A.csv / annotator_B.csv — mẫu điền nhãn (6 cờ + kết luận)
  gold_key.csv           — (riêng tư) item_id → review_id, nhóm, module đã chọn
  corpus_lookup.csv      — toàn bộ review (không điểm số) để người gán nhãn tự tra cứu nội dung trùng
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, write_table
from src.utils.config import Paths

SCORE_COLS = ["s_dup", "s_content", "s_burst", "s_behav", "suspicion"]
FLAG_COLS = ["f1_chung_chung", "f2_quang_cao", "f3_trung_noi_dung", "f4_tai_khoan_it_review_cung_muc_sao",
             "f5_nhieu_quan_xa_nhau_cung_luc", "f6_trong_cum_dang_bat_thuong"]


def _context(df: pd.DataFrame) -> pd.DataFrame:
    """Bối cảnh thô (không phải điểm module) giúp người gán nhãn áp rubric cờ 4–6."""
    d = df.copy()
    daily = d[d["date_precision_days"] <= 1].groupby(["place_key", "review_day"]).size()
    span_days = d.groupby("place_key")["review_day"].agg(lambda s: max((s.max() - s.min()).days, 1))
    d["place_daily_avg"] = d["place_key"].map(d.groupby("place_key").size() / span_days).round(2)
    pm3 = []
    for key, day, prec in zip(d["place_key"], d["review_day"], d["date_precision_days"]):
        if prec is None or np.isnan(prec) or prec > 1:
            pm3.append(np.nan)
            continue
        s = 0
        for k in range(-3, 4):
            s += daily.get((key, day + pd.Timedelta(days=k)), 0)
        pm3.append(s)
    d["place_reviews_pm3d"] = pm3
    # Các review khác của cùng reviewer trong corpus
    grp = d.sort_values("review_day").groupby("reviewer_hash")
    other = {}
    for h, g in grp:
        if len(g) > 1:
            other[h] = [(r.place_key, f"{r.place_name or r.place_key} ({r.review_day:%d/%m/%Y}, {int(r.rating)}★)")
                        for r in g.itertuples()]
    d["reviewer_other_reviews"] = [
        "; ".join([txt for k, txt in other.get(h, []) if k != pk][:5]) if h in other else ""
        for h, pk in zip(d["reviewer_hash"], d["place_key"])]
    return d


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("sample_gold", P.logs)
    e = cfg["evaluation"]
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(P.processed / "scored_reviews.parquet")
    places = pd.read_parquet(P.interim / "places.parquet")
    df = df.merge(places[[c for c in ["place_key", "name", "url"] if c in places]].rename(
        columns={"name": "place_name", "url": "place_url"}), on="place_key", how="left")
    if "place_name" not in df:
        df["place_name"] = df["place_key"]
    if "place_url" not in df:
        df["place_url"] = ""

    # Nhóm A
    picked: dict[str, list[str]] = {}
    for col in SCORE_COLS:
        top = df[df[col].notna() & (df[col] > 0)].nlargest(e["group_a_top_per_module"], col)
        for rid in top["review_id"]:
            picked.setdefault(rid, []).append(col)
    # Nhóm B
    b_ids = set(rng.choice(df["review_id"].to_numpy(), size=min(e["group_b_random"], len(df)), replace=False))

    ids = list(dict.fromkeys(list(picked) + sorted(b_ids)))
    g = df[df["review_id"].isin(ids)].copy()
    g = _context(df).loc[g.index]
    g["in_group_a"] = g["review_id"].isin(picked.keys())
    g["in_group_b"] = g["review_id"].isin(b_ids)
    g["picked_by"] = g["review_id"].map(lambda r: ",".join(picked.get(r, [])))
    g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    g["item_id"] = np.arange(1, len(g) + 1)

    blind_cols = ["item_id", "place_name", "place_url", "rating", "review_day", "date_text", "text",
                  "reviewer_n_reviews", "reviewer_local_guide", "reviewer_n_photos", "place_daily_avg",
                  "place_reviews_pm3d", "reviewer_other_reviews", "has_owner_response", "n_photos"]
    blind = g[[c for c in blind_cols if c in g]]
    write_table(blind, P.labels / "gold_sample_blind.csv")
    tmpl = g[["item_id"]].copy()
    for f in FLAG_COLS:
        tmpl[f] = ""
    tmpl["label"] = ""          # Genuine | Uncertain | Spam
    tmpl["ghi_chu"] = ""
    # Người gán: TOÀN BỘ Nhóm B + human_a_n mục đầu (theo item_id) của phần chỉ thuộc Nhóm A.
    # Claude gán TOÀN BỘ Nhóm A (gồm human_a_n mục chung với người để kiểm chứng).
    only_a = g.loc[g["in_group_a"] & ~g["in_group_b"]].sort_values("item_id")
    g["annotator"] = np.where(g["in_group_b"] | g["item_id"].isin(only_a["item_id"].head(e["human_a_n"])),
                              "human", "claude")
    g["for_claude"] = g["in_group_a"]
    human = tmpl[tmpl["item_id"].isin(g.loc[g["annotator"] == "human", "item_id"])]
    for who in ("A", "B"):
        p = P.labels / f"annotator_{who}.csv"
        if not p.exists():                          # không ghi đè nhãn đã gán
            write_table(human, p)
    write_table(g[["item_id", "review_id", "in_group_a", "in_group_b", "picked_by", "annotator", "for_claude"]],
                P.labels / "gold_key.csv")
    write_table(df[["review_id", "place_name", "rating", "review_day", "text"]].rename(
        columns={"review_id": "id_tra_cuu"}), P.labels / "corpus_lookup.csv")
    log.info("Gold set: %d mục (A=%d, B=%d, giao nhau=%d) | người gán %d mục (pilot = %d mục đầu), "
             "Claude gán %d mục Nhóm A (%d mục chung với người)",
             len(g), int(g["in_group_a"].sum()), int(g["in_group_b"].sum()),
             int((g["in_group_a"] & g["in_group_b"]).sum()), len(human), e["pilot_n"],
             int(g["for_claude"].sum()), int((g["for_claude"] & (g["annotator"] == "human")).sum()))
    return P.labels / "gold_sample_blind.csv"
