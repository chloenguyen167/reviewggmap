"""Module 3 — Đặc trưng hành vi tài khoản / review (không giám sát).

Đặc trưng được ĐỊNH HƯỚNG (lớn = đáng ngờ) rồi CẮT MỘT PHÍA tại median trước khi
đưa vào Isolation Forest, để IF chỉ cô lập các điểm lệch về phía đáng ngờ.

Nguồn đặc trưng:
  - hồ sơ tài khoản: số review, Local Guide, số ảnh
  - review: sao cực đoan, lệch so với điểm TB quán, không có chữ, độ dài, link/SĐT
  - trong corpus (miễn phí): số quán khác nhau reviewer đánh giá trong cùng ngày,
    khoảng cách địa lý lớn nhất giữa các quán đó (km)
  - lịch sử reviewer (nếu đã thu thập contributors): tỷ lệ sao cực đoan, số quán/ngày, km/ngày
Đầu ra: processed/features_m3.parquet
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.common import get_logger, one_sided
from src.utils.config import Paths

REASON_TEXT = {
    "newness": "tài khoản ít review ({n_rev:.0f})",
    "not_local_guide": "không phải Local Guide",
    "few_photos_acc": "tài khoản gần như không có ảnh",
    "extreme": "chấm sao cực đoan ({rating:.0f}★)",
    "rating_dev": "lệch {rating_dev:.1f} sao so với điểm TB quán",
    "star_only_extreme": "chỉ chấm sao cực đoan, không viết gì",
    "shortness": "nội dung rất ngắn",
    "url_phone": "chèn link/SĐT",
    "same_day_places": "đánh giá {same_day_places:.0f} quán trong cùng ngày",
    "geo_km_day": "các quán cách nhau {geo_km_day:.0f} km trong cùng ngày",
    "contrib_extreme_share": "{contrib_extreme_share:.0%} review trong lịch sử là 1★/5★",
    "contrib_places_day": "lịch sử: tới {contrib_places_day:.0f} quán/ngày",
    "contrib_km_day": "lịch sử: quán cách nhau {contrib_km_day:.0f} km/ngày",
}


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lng2 - lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _max_pair_km(coords: list[tuple[float, float]]) -> float:
    coords = [c for c in coords if c[0] is not None and not np.isnan(c[0])]
    if len(coords) < 2:
        return 0.0
    return max(haversine_km(a[0], a[1], b[0], b[1]) for a, b in combinations(coords, 2))


def _same_day_activity(frame: pd.DataFrame, key_cols=("reviewer_hash", "day")) -> pd.DataFrame:
    """Với mỗi (reviewer, ngày): số địa điểm khác nhau và khoảng cách lớn nhất (km)."""
    keys = list(key_cols)
    out = frame.groupby(keys).agg(places=("place", "nunique")).reset_index()
    out["km"] = 0.0
    multi = out[out["places"] >= 2][keys]
    if len(multi):                                   # chỉ tính km cho nhóm có ≥ 2 quán
        sub = frame.merge(multi, on=keys).drop_duplicates(keys + ["place"])
        km = sub.groupby(keys)[["lat", "lng"]].apply(lambda x: _max_pair_km(list(zip(x["lat"], x["lng"]))))
        out = out.merge(km.rename("km2").reset_index(), on=keys, how="left")
        out["km"] = out["km2"].fillna(0.0)
        out = out.drop(columns="km2")
    return out


def compute(df: pd.DataFrame, places: pd.DataFrame, contrib: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    c = cfg["m3"]
    d = df.copy()
    pm = d.groupby("place_key")["rating"].mean()
    d["rating_dev"] = (d["rating"] - d["place_key"].map(pm)).abs()
    coords = places.set_index("place_key")[["lat", "lng"]] if {"lat", "lng"} <= set(places.columns) else None

    # --- hoạt động trong corpus, cùng ngày (chỉ review có ngày chính xác) ----------------
    exact = d[(d["date_precision_days"] <= 1) & d["reviewer_hash"].notna()].copy()
    exact = exact.assign(day=exact["review_day"], place=exact["place_key"])
    if coords is not None:
        exact = exact.join(coords, on="place_key")
    else:
        exact["lat"] = exact["lng"] = np.nan
    act = _same_day_activity(exact) if len(exact) else pd.DataFrame(columns=["reviewer_hash", "day", "places", "km"])
    d = d.merge(act.rename(columns={"day": "review_day", "places": "same_day_places", "km": "geo_km_day"}),
                on=["reviewer_hash", "review_day"], how="left")
    d.loc[d["date_precision_days"] > 1, ["same_day_places", "geo_km_day"]] = np.nan
    d["corpus_n_reviews"] = d.groupby("reviewer_hash")["review_id"].transform("size")

    # --- lịch sử reviewer (tùy chọn) ----------------------------------------------------
    if contrib is not None and not contrib.empty:
        cc = contrib.copy()
        agg = cc.groupby("reviewer_hash").agg(contrib_n=("rating", "size"),
                                             contrib_extreme_share=("rating", lambda r: r.isin([1, 5]).mean()))
        ex = cc[cc["date_precision_days"] <= 1].assign(day=lambda x: x["review_date"].dt.normalize(),
                                                        place=lambda x: x["place_name"])
        if len(ex):
            a2 = _same_day_activity(ex).groupby("reviewer_hash").agg(contrib_places_day=("places", "max"),
                                                                    contrib_km_day=("km", "max"))
            agg = agg.join(a2, how="left")
        d = d.merge(agg.reset_index(), on="reviewer_hash", how="left")

    # --- định hướng: lớn = đáng ngờ ---------------------------------------------------
    n_rev = d["reviewer_n_reviews"].fillna(d["corpus_n_reviews"])
    F = pd.DataFrame({
        "newness": -np.log1p(n_rev),
        "not_local_guide": 1 - d["reviewer_local_guide"].astype("float").fillna(0.0),
        "few_photos_acc": -np.log1p(d["reviewer_n_photos"].fillna(d["reviewer_n_photos"].median()
                                                                   if d["reviewer_n_photos"].notna().any() else 0)),
        "extreme": d["rating"].isin([1, 5]).astype(float),
        "rating_dev": d["rating_dev"].fillna(0),
        "star_only_extreme": ((~d["has_text"]) & d["rating"].isin([1, 5])).astype(float),
        "shortness": np.where(d["has_text"], -np.log1p(d["n_words"]), 0.0),
        "url_phone": (d["n_urls"] + d["n_phones"]).astype(float),
        "same_day_places": d["same_day_places"].fillna(1),
        "geo_km_day": d["geo_km_day"].fillna(0),
    })
    for col in ("contrib_extreme_share", "contrib_places_day", "contrib_km_day"):
        if col in d:
            F[col] = d[col].fillna(d[col].median() if d[col].notna().any() else 0)
    F = F.loc[:, F.std() > 0]                         # bỏ cột hằng số (vd. không có dữ liệu ảnh)

    Fo = one_sided(F)
    iso = IsolationForest(n_estimators=300, contamination=c["contamination"], random_state=cfg["project"]["seed"])
    iso.fit(Fo)
    raw = -iso.score_samples(Fo)
    raw = raw - raw.min()                             # ≥ 0; điểm thấp nhất = 0

    # --- lý do: 2 đặc trưng đóng góp mạnh nhất (theo percentile sau khi cắt một phía) ----
    pct = Fo.rank(pct=True)
    pct[Fo.eq(Fo.median(), axis=1)] = 0             # đặc trưng ở mức bình thường không phải lý do
    vals = d.assign(n_rev=n_rev)
    reasons = [""] * len(d)
    for i in np.where(raw >= np.quantile(raw, 0.8))[0]:    # chỉ viết lý do cho top 20% nghi vấn
        top = pct.iloc[i].sort_values(ascending=False)
        top = top[top > 0.9].index[:2]
        parts = []
        for f in top:
            try:
                parts.append(REASON_TEXT.get(f, f).format(**vals.iloc[i].to_dict()))
            except (KeyError, ValueError, TypeError):
                parts.append(f)
        reasons[i] = "hành vi: " + "; ".join(parts) if parts else ""

    feat = pd.DataFrame({"review_id": d["review_id"], "s_behav_raw": raw, "behav_if_flag": iso.predict(Fo) == -1,
                         "reason_behav": reasons})
    return pd.concat([feat, F.add_prefix("b_").reset_index(drop=True)], axis=1)


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("m3", P.logs)
    df = pd.read_parquet(P.interim / "reviews_clean.parquet")
    places = pd.read_parquet(P.interim / "places.parquet")
    cpath = P.interim / "contributors.parquet"
    contrib = pd.read_parquet(cpath) if cpath.exists() else None
    feat = compute(df, places, contrib, cfg)
    out = P.processed / "features_m3.parquet"
    feat.to_parquet(out, index=False)
    log.info("M3: %d đặc trưng, %d review bị IF gắn cờ (%.1f%%)%s",
             sum(c.startswith("b_") for c in feat.columns), feat["behav_if_flag"].sum(),
             100 * feat["behav_if_flag"].mean(), " (có dùng lịch sử reviewer)" if contrib is not None else "")
    return out
