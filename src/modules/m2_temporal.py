"""Module 2 — Phát hiện đợt tăng đột biến review (temporal burst).

Đơn vị phân tích: (địa điểm, ngày) — hoặc (địa điểm, tuần) nếu m2.freq = W.
Chỉ dùng review có độ chính xác ngày đủ tốt (≤1 ngày với D, ≤7 ngày với W); review khác → NaN.

1. Robust z-score của số review so với cửa sổ trượt `window` bước TRƯỚC đó:
       z = (x − median) / max(1.4826·MAD, √max(median, 1))
   Sàn √max(median,1) (≈ độ lệch chuẩn Poisson) tránh chia cho 0 trên chuỗi thưa.
2. Isolation Forest trên đặc trưng ngày (số lượng, z, lệch điểm, tỷ lệ 5★/1★,
   tỷ lệ tài khoản mới, tỷ lệ không có chữ) — đã định hướng + cắt một phía.
3. Ngày bị gắn cờ nếu (z ≥ ngưỡng và x ≥ min_count) hoặc IF gắn cờ; mở rộng ±expand bước.
4. Điểm review = điểm ngày × mức "đồng hướng" (review 5★ trong đợt burst 5★ bị nghi hơn).

M2b — PHỐI HỢP LIÊN QUÁN (tín hiệu thời gian thứ hai, module riêng trong fusion: "coord"):
   M2a chỉ thấy đột biến TRONG TỪNG QUÁN; một chiến dịch rải mỗi quán vài bản sao gần như không làm quán
   nào đột biến. M2b xét các cụm văn bản gần trùng do M1a tìm ra: với mỗi review (ngày chính xác) trong cụm,
   đếm số bản trùng ở QUÁN KHÁC đăng trong ±coord_window_days ngày, so với kỳ vọng nếu các bản trùng rải
   ngẫu nhiên trên toàn khoảng thời gian của corpus:
       λ = (số bản trùng có ngày ở quán khác) × min(2·w / span, 1)
       s_coord = −log10 P(Poisson(λ) ≥ số bản trùng trong cửa sổ)
   Câu khen sáo rỗng tự nhiên rải qua nhiều năm → bất ngờ thấp; chiến dịch dồn vào vài ngày → bất ngờ cao.
   Review không thuộc cụm trùng nào → NaN (không áp dụng; fusion bỏ qua module này cho review đó).
   (Trước đây phần này nằm trong M1a; tách ra để M1a chỉ đo văn bản và mọi bằng chứng thời gian ở M2.)

Đầu ra: processed/features_m2.parquet, processed/features_m2b.parquet, processed/days_m2.parquet,
        reports/tables/m2_top_bursts.csv, reports/figures/fig_m2_bursts.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import IsolationForest

from src.utils.common import get_logger, one_sided, write_table
from src.utils.config import Paths


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def robust_z(counts: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    prev = counts.shift(1)
    minp = max(7, window // 4)
    med = prev.rolling(window, min_periods=minp).median()
    mad = prev.rolling(window, min_periods=minp).apply(_mad, raw=True)
    denom = np.maximum(1.4826 * mad, np.sqrt(np.maximum(med, 1.0)))
    return (counts - med) / denom, med


def _align(rating: float, day_mean: float) -> float:
    if np.isnan(day_mean):
        return 0.6
    if day_mean >= 4.0:
        return {5: 1.0, 4: 0.6}.get(int(rating), 0.2)
    if day_mean <= 2.5:
        return {1: 1.0, 2: 0.8}.get(int(rating), 0.2)
    return 0.6


def compute(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg["m2"]
    step_col, max_prec, freq = ("review_day", 1.0, "D") if c["freq"] == "D" else ("review_week", 7.0, "W-MON")
    elig = df[df["review_date"].notna() & (df["date_precision_days"] <= max_prec)].copy()
    elig["step"] = elig[step_col]
    newmax = c["new_account_max_reviews"]
    elig["is_new"] = (elig["reviewer_n_reviews"] <= newmax).astype(float).where(elig["reviewer_n_reviews"].notna())
    place_mean = df.groupby("place_key")["rating"].mean()

    days = []
    for key, g in elig.groupby("place_key"):
        agg = g.groupby("step").agg(n=("review_id", "size"), mean_rating=("rating", "mean"),
                                    share_5=("rating", lambda r: (r == 5).mean()),
                                    share_1=("rating", lambda r: (r <= 1).mean()),
                                    share_new=("is_new", "mean"),
                                    share_no_text=("has_text", lambda h: 1 - h.mean()))
        full_idx = pd.date_range(agg.index.min(), agg.index.max(), freq="D" if freq == "D" else "7D")
        cnt = agg["n"].reindex(full_idx, fill_value=0).astype(float)
        if (cnt > 0).sum() >= c["min_active_steps"]:
            z, med = robust_z(cnt, c["window"])
        else:
            z = med = pd.Series(np.nan, index=full_idx)
        agg = agg.join(pd.DataFrame({"z": z, "baseline": med}), how="left")
        agg["place_key"] = key
        agg["rating_dev"] = (agg["mean_rating"] - place_mean.get(key, np.nan)).abs()
        days.append(agg.reset_index(names="step"))
    days = pd.concat(days, ignore_index=True) if days else pd.DataFrame()
    if days.empty:
        return pd.DataFrame({"review_id": df["review_id"]}), days

    # --- Isolation Forest trên các ngày có review ----------------------------------
    X = pd.DataFrame({
        "log_n": np.log1p(days["n"]),
        "z": days["z"].fillna(0).clip(-5, 20),
        "rating_dev": days["rating_dev"].fillna(0),
        "extreme_share": np.maximum(days["share_5"], days["share_1"]),
        "share_new": days["share_new"].fillna(days["share_new"].median() if days["share_new"].notna().any() else 0),
        "share_no_text": days["share_no_text"],
    })
    Xo = one_sided(X)
    iso = IsolationForest(n_estimators=300, contamination=c["contamination"], random_state=cfg["project"]["seed"])
    iso.fit(Xo)
    days["if_anom"] = -iso.score_samples(Xo)
    # 1 review đơn lẻ không phải "đợt" — trường hợp đó để Module 3 (hành vi) xử lý
    days["if_flag"] = (iso.predict(Xo) == -1) & (days["n"] >= c.get("if_min_count", 2))
    days["s_if"] = days["if_anom"].rank(pct=True)
    days["z_flag"] = (days["z"] >= c["z_threshold"]) & (days["n"] >= c["min_count"])
    days["s_z"] = np.where(days["n"] >= c["min_count"], (days["z"] / (2 * c["z_threshold"])).clip(0, 1), 0.0)
    days["s_z"] = days["s_z"].fillna(0)
    days["flag"] = days["z_flag"] | days["if_flag"]
    days["day_score"] = np.where(days["flag"], np.maximum(days["s_z"], days["s_if"]), 0.0)

    # --- Mở rộng ±expand bước quanh ngày bị gắn cờ ---------------------------------
    step_td = pd.Timedelta(days=1 if freq == "D" else 7)
    days = days.sort_values(["place_key", "step"])
    exp = days["day_score"].copy()
    lookup = days.set_index(["place_key", "step"])["day_score"]
    for k in range(1, c["expand"] + 1):
        for sign in (-1, 1):
            nb_idx = list(zip(days["place_key"], days["step"] + sign * k * step_td))
            nb = lookup.reindex(nb_idx).fillna(0).to_numpy()
            exp = np.maximum(exp, 0.7 * nb)
    days["day_score_exp"] = exp

    # --- Gán điểm cho review ---------------------------------------------------------
    r = elig[["review_id", "place_key", "step", "rating"]].merge(
        days[["place_key", "step", "n", "z", "baseline", "mean_rating", "flag", "day_score_exp"]],
        on=["place_key", "step"], how="left")
    align = np.array([_align(a, b) for a, b in zip(r["rating"], r["mean_rating"])])
    r["s_burst_raw"] = r["day_score_exp"] * (0.5 + 0.5 * align)
    r["reason_burst"] = [
        (f"trong đợt tăng đột biến {s:%d/%m/%Y}: {int(n)} review/{'ngày' if freq == 'D' else 'tuần'}"
         + (f" (nền ~{b:.1f}, z={zz:.1f})" if pd.notna(zz) else ""))
        if sc > 0 else ""
        for s, n, b, zz, sc in zip(r["step"], r["n"], r["baseline"], r["z"], r["day_score_exp"])]
    feat = df[["review_id"]].merge(
        r[["review_id", "n", "z", "flag", "s_burst_raw", "reason_burst"]].rename(
            columns={"n": "burst_step_n", "z": "burst_z", "flag": "burst_flag"}), on="review_id", how="left")
    return feat, days


def _figure(days: pd.DataFrame, names: dict, P: Paths, freq: str) -> None:
    import matplotlib.pyplot as plt
    from src.utils import plotstyle as ps
    ps.apply()
    top = days[days["flag"]].groupby("place_key")["day_score"].max().sort_values(ascending=False).head(3).index
    if len(top) == 0:
        return
    fig, axes = plt.subplots(len(top), 1, figsize=(7.5, 2.0 * len(top)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, top):
        d = days[days["place_key"] == key].set_index("step")
        idx = pd.date_range(d.index.min(), d.index.max(), freq="D" if freq == "D" else "7D")
        n = d["n"].reindex(idx, fill_value=0)
        ax.plot(idx, n.values, color=ps.SERIES[0], linewidth=1.1, label="số review")
        fl = d[d["flag"]]
        ax.scatter(fl.index, fl["n"], s=28, color=ps.ALERT, edgecolor=ps.SURFACE, linewidth=1.5, zorder=3,
                   label="bị gắn cờ burst")
        ax.set_title(str(names.get(key, key))[:55], fontsize=9, loc="left")
        ax.set_ylabel("review/" + ("ngày" if freq == "D" else "tuần"))
    axes[0].legend(loc="upper left")
    fig.suptitle("Các đợt tăng đột biến được phát hiện (Module 2)", fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    ps.save(fig, P.figures / "fig_m2_bursts.png")


def coordination(df: pd.DataFrame, m1a: pd.DataFrame, win: int) -> pd.DataFrame:
    """M2b: mức bất ngờ Poisson của các bản gần trùng ở QUÁN KHÁC đăng trong ±win ngày."""
    exact = df["date_precision_days"].fillna(np.inf) <= 1
    out = pd.DataFrame({"review_id": df["review_id"], "coord_n_window": np.nan, "coord_lambda": np.nan,
                        "s_coord_raw": np.nan, "reason_coord": ""})
    if not exact.any() or "dup_cluster_id" not in m1a:
        return out
    days = df.loc[exact, "review_date"]
    span = max((days.max() - days.min()) / pd.Timedelta(days=1), 2 * win)
    d = df.loc[exact, ["review_id", "place_key", "review_date"]].merge(
        m1a.loc[m1a["dup_cluster_id"].fillna(-1) >= 0, ["review_id", "dup_cluster_id"]], on="review_id")
    res = {}
    for _, g in d.groupby("dup_cluster_id"):
        t = g["review_date"].to_numpy(dtype="datetime64[ns]")
        pl = g["place_key"].to_numpy()
        for i, rid in enumerate(g["review_id"]):
            other = pl != pl[i]
            n_other = int(other.sum())
            k = int((other & (np.abs((t - t[i]) / np.timedelta64(1, "D")) <= win)).sum())
            lam = n_other * min(2 * win / span, 1.0)
            sur = float(-poisson.logsf(k - 1, lam) / np.log(10)) if k > 0 and lam > 0 else 0.0
            res[rid] = (k, lam, sur, f"bản trùng ở {k} quán khác đăng trong ±{win} ngày "
                                     f"(kỳ vọng ngẫu nhiên {lam:.2f})" if k else "")
    m = out["review_id"].map(res)
    has = m.notna()
    out.loc[has, ["coord_n_window", "coord_lambda", "s_coord_raw", "reason_coord"]] = pd.DataFrame(
        m[has].tolist(), index=out.index[has], columns=["coord_n_window", "coord_lambda", "s_coord_raw", "reason_coord"])
    return out


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("m2", P.logs)
    df = pd.read_parquet(P.interim / "reviews_clean.parquet")
    feat, days = compute(df, cfg)
    out = P.processed / "features_m2.parquet"
    feat.to_parquet(out, index=False)
    if not days.empty:
        days.to_parquet(P.processed / "days_m2.parquet", index=False)
        places = pd.read_parquet(P.interim / "places.parquet")
        names = places.set_index("place_key")["name"].to_dict() if "name" in places else {}
        top = days[days["flag"]].sort_values("day_score", ascending=False).head(30).copy()
        top["name"] = top["place_key"].map(names)
        write_table(top[["name", "place_key", "step", "n", "baseline", "z", "mean_rating", "share_5", "share_1",
                         "share_new", "z_flag", "if_flag", "day_score"]], P.tables / "m2_top_bursts.csv")
        _figure(days, names, P, cfg["m2"]["freq"])
    n_el = feat["s_burst_raw"].notna().sum() if "s_burst_raw" in feat else 0
    log.info("M2: %d review đủ độ chính xác ngày (%.0f%%), %d ngày bị gắn cờ, %d review trong đợt burst",
             n_el, 100 * n_el / max(len(df), 1), int(days["flag"].sum()) if not days.empty else 0,
             int((feat.get("s_burst_raw", pd.Series(dtype=float)) > 0).sum()))

    m1a_p = P.processed / "features_m1a.parquet"
    if m1a_p.exists():
        co = coordination(df, pd.read_parquet(m1a_p), cfg["m2"].get("coord_window_days", 14))
        co.to_parquet(P.processed / "features_m2b.parquet", index=False)
        log.info("M2b: %d review trong cụm gần trùng có ngày chính xác, %d có bản trùng ở quán khác trong cửa sổ, "
                 "%d có mức bất ngờ ≥ 2 (p ≤ 0,01)", int(co["s_coord_raw"].notna().sum()),
                 int((co["coord_n_window"] > 0).sum()), int((co["s_coord_raw"] >= 2).sum()))
    else:
        log.warning("Chưa có features_m1a.parquet → bỏ qua M2b (chạy m1a trước m2)")
    return out
