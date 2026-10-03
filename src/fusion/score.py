"""Fusion — gộp điểm các module thành Suspicion Score và chia 3 vùng quyết định.

1. Mỗi điểm thô → percentile rank [0,1] (điểm ≤ 0, tức "không có tín hiệu", giữ nguyên 0).
2. Module không áp dụng được cho một review (NaN) → bỏ ra và CHUẨN HÓA LẠI trọng số
   trên các module còn lại (không điền 0, để review chỉ chấm sao không bị "oan" hay "lọt").
3. Suspicion = Σ wᵢ·sᵢ / Σ wᵢ (trên các module có mặt).
4. Ba vùng: auto_flag (≥ θ_high) / review (≥ θ_low) / safe — giống cơ chế 3 vùng
   của bài Entity Resolution. θ là HẠN NGẠCH theo phân vị do nhóm chọn (top 2% / top 10%) để giới hạn
   khối lượng cần xem tay, KHÔNG phải ngưỡng đã hiệu chỉnh theo xác suất spam.
5. Ma trận tương quan Spearman giữa các điểm module (module_correlation.csv) để kiểm tra giả định độc lập
   của phép gộp Fisher.
Đầu ra: processed/scored_reviews.parquet, reports/tables/top100_suspicious.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, pct_rank, write_table
from src.utils.config import Paths

MODULES = {  # tên module → (file đặc trưng, cột điểm thô, cột lý do)
    "dup": ("features_m1a.parquet", "s_dup_raw", "reason_dup"),
    "content": ("features_m1b.parquet", "s_content_raw", "reason_content"),
    "burst": ("features_m2.parquet", "s_burst_raw", "reason_burst"),
    "coord": ("features_m2b.parquet", "s_coord_raw", "reason_coord"),   # M2b phối hợp liên quán
    "behav": ("features_m3.parquet", "s_behav_raw", "reason_behav"),
}


def load_scores(P: Paths) -> pd.DataFrame:
    base = pd.read_parquet(P.interim / "reviews_clean.parquet")
    df = base.copy()
    for name, (fname, col, rcol) in MODULES.items():
        f = P.processed / fname
        if not f.exists():
            df[f"s_{name}"] = np.nan
            df[rcol] = ""
            continue
        feat = pd.read_parquet(f).drop_duplicates("review_id")
        keep = [c for c in feat.columns if c not in df.columns or c == "review_id"]
        df = df.merge(feat[keep], on="review_id", how="left")
        df[f"s_{name}"] = pct_rank(df[col]) if col in df else np.nan
        if rcol not in df:
            df[rcol] = ""
    return df


def suspicion(df: pd.DataFrame, modules: list[str], weights: dict[str, float] | None = None,
              method: str = "mean") -> pd.Series:
    """Gộp điểm các module có mặt (bỏ qua NaN). NaN nếu không module nào áp dụng được.

    mean   : Σ wᵢ·sᵢ / Σ wᵢ — dễ giải thích, nhưng một tín hiệu rất mạnh bị các module
             "bình thường" kéo xuống (spam thường chỉ nổi ở 1–2 module).
    fisher : phương pháp Fisher gộp p-value, pᵢ = 1 − sᵢ (sᵢ là percentile nên pᵢ ≈ p-value
             thực nghiệm "phía trên"); X = −2 Σ wᵢ ln pᵢ ~ χ²(2k). Một bằng chứng cực mạnh đủ
             để đẩy điểm lên, nhiều bằng chứng vừa phải cũng cộng dồn. Trả về x/(1+x) với
             x = −log10(p_gộp) để giữ thứ tự mà không bị bão hòa ở 1.0.
    max    : điểm cao nhất trong các module (nhạy nhất, nhưng dễ báo động giả).
    """
    from scipy.stats import chi2
    weights = weights or {}
    S = df[[f"s_{m}" for m in modules]].to_numpy(dtype=float)
    W = np.array([weights.get(m, 1.0) for m in modules], dtype=float)
    mask = ~np.isnan(S)
    k = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "mean":
            num = np.nansum(S * W, axis=1)
            den = (mask * W).sum(axis=1)
            out = np.where(den > 0, num / den, np.nan)
        elif method == "max":
            out = np.where(k > 0, np.nanmax(np.where(mask, S, -np.inf), axis=1), np.nan)
        elif method == "fisher":
            pv = np.clip(1.0 - np.where(mask, S, 0.0), 1e-6, 1.0)
            X = -2.0 * (W * np.log(pv) * mask).sum(axis=1)
            logp = chi2.logsf(X, 2 * np.maximum(k, 1)) / np.log(10)   # log10 p_gộp
            x = -logp
            out = np.where(k > 0, x / (1.0 + x), np.nan)
        else:
            raise ValueError(f"fusion.method không hợp lệ: {method}")
    return pd.Series(out, index=df.index)


def assign_zones(score: pd.Series, zcfg: dict) -> tuple[pd.Series, float, float]:
    hi = float(score.quantile(zcfg["high_quantile"]))
    lo = float(score.quantile(zcfg["low_quantile"]))
    zone = np.where(score >= hi, "auto_flag", np.where(score >= lo, "review", "safe"))
    return pd.Series(zone, index=score.index), hi, lo


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("fusion", P.logs)
    fc = cfg["fusion"]
    df = load_scores(P)
    mods = list(MODULES)
    df["suspicion"] = suspicion(df, mods, fc["weights"], fc.get("method", "mean"))
    df["n_modules"] = df[[f"s_{m}" for m in mods]].notna().sum(axis=1)
    df["zone"], hi, lo = assign_zones(df["suspicion"], fc["zones"])

    # Lý do: chỉ ghép lý do của module có điểm cao
    thr = fc["reason_min_score"]
    rs = []
    for row in df[[f"s_{m}" for m in mods] + [MODULES[m][2] for m in mods]].itertuples(index=False):
        parts = [getattr(row, MODULES[m][2]) for m in mods
                 if pd.notna(getattr(row, f"s_{m}")) and getattr(row, f"s_{m}") >= thr and getattr(row, MODULES[m][2])]
        rs.append(" | ".join(parts))
    df["reasons"] = rs

    out = P.processed / "scored_reviews.parquet"
    df.to_parquet(out, index=False)
    cols = ["review_id", "place_key", "rating", "review_day", "text", "suspicion", "zone",
            "s_dup", "s_content", "s_burst", "s_behav", "reasons"]
    write_table(df.sort_values("suspicion", ascending=False)[cols].head(100), P.tables / "top100_suspicious.csv")
    # Giả định của phép gộp Fisher: các điểm module ĐỘC LẬP → báo cáo tương quan Spearman giữa các điểm module
    # (trên các review có cả hai điểm; kèm số review dùng cho từng cặp)
    sc = [f"s_{m}" for m in mods]
    corr = df[sc].corr(method="spearman", min_periods=30)
    npair = pd.DataFrame({a: {b: int((df[a].notna() & df[b].notna()).sum()) for b in sc} for a in sc})
    write_table(corr.round(3).reset_index().rename(columns={"index": "module"}), P.tables / "module_correlation.csv")
    write_table(npair.reset_index().rename(columns={"index": "module"}), P.tables / "module_correlation_n.csv")
    off = corr.where(~np.eye(len(sc), dtype=bool)).abs().stack()
    if len(off):
        (log.warning if off.max() > 0.3 else log.info)(
            "Tương quan Spearman lớn nhất giữa 2 module: %s–%s = %.2f → %s", *off.idxmax(), off.max(),
            "VƯỢT 0,3: giả định độc lập của Fisher đáng ngờ, cần nêu trong Limitations / so với cách gộp mean"
            if off.max() > 0.3 else "giả định độc lập chấp nhận được (|ρ| ≤ 0,3)")

    zc = df["zone"].value_counts()
    log.info("Fusion: θ_high=%.3f, θ_low=%.3f | auto_flag=%d, review=%d, safe=%d",
             hi, lo, zc.get("auto_flag", 0), zc.get("review", 0), zc.get("safe", 0))
    return out
