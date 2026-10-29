"""Weak supervision — gộp các luật (labeling function, LF) bằng label model Dawid–Skene.

Phương pháp chính của đề tài (thay cho gộp điểm Fisher trong score.py):
  1. 8 LF một chiều, mỗi LF là một điều kiện đọc được: bỏ phiếu SPAM (1), GENUINE (0) hoặc bỏ phiếu trắng (−1).
  2. Label model Dawid–Skene (EM, không cần nhãn): học xác suất mỗi LF bỏ phiếu khi review là spam/genuine,
     rồi suy ra P(spam | các phiếu) cho từng review.
Đầu ra: processed/ws_scores.parquet (review_id, ws_score, phiếu của từng LF),
        reports/tables/ws_lf_summary.csv (độ phủ, chồng lấp, xung đột, tham số đã học của từng LF)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, write_table
from src.utils.config import Paths

SPAM, GENUINE, ABSTAIN = 1, 0, -1

LF_DESC = {
    "LF_dup": "near-duplicate text (≥8 words) shared with another account",
    "LF_ad": "advertising cues (URL, phone, Zalo, promo keywords)",
    "LF_coord": "same text at another place within ±14 days (cross-place coordination)",
    "LF_burst": "1★/5★ review inside a detected burst at the place",
    "LF_thin": "account with ≤3 reviews, 1★/5★, not a Local Guide",
    "LF_veteran": "Local Guide with ≥20 reviews",
    "LF_detailed": "≥30 words and not a duplicate",
    "LF_moderate": "2–4★ outside any burst",
}


def labeling_functions(S: pd.DataFrame) -> pd.DataFrame:
    """Ma trận phiếu n × 8 (1 = spam, 0 = genuine, −1 = bỏ phiếu trắng)."""
    ext = S["rating"].isin([1, 5])
    lg = S["reviewer_local_guide"].fillna(False).astype(bool)
    dup = (S["dup_cross_user"].fillna(0) >= 1) & (S["n_words"] >= 8)
    burst = S["burst_flag"].fillna(False).astype(bool) & ext
    spam_lf = {
        "LF_dup": dup,
        "LF_ad": S["content_rule"].fillna(0) > 0,
        "LF_coord": S["s_coord_raw"].fillna(0) > 0,
        "LF_burst": burst,
        "LF_thin": (S["reviewer_n_reviews"] <= 3) & ext & ~lg,
    }
    gen_lf = {
        "LF_veteran": lg & (S["reviewer_n_reviews"] >= 20),
        "LF_detailed": (S["n_words"] >= 30) & ~dup,
        "LF_moderate": S["rating"].isin([2, 3, 4]) & ~burst,
    }
    L = {k: np.where(v, SPAM, ABSTAIN) for k, v in spam_lf.items()}
    L.update({k: np.where(v, GENUINE, ABSTAIN) for k, v in gen_lf.items()})
    return pd.DataFrame(L, index=S.index)


def dawid_skene(L: np.ndarray, n_iter: int = 200, tol: float = 1e-7, smooth: float = 0.1):
    """Label model nhị phân, các LF độc lập có điều kiện theo nhãn thật y, có bỏ phiếu trắng.

    Tham số: π = P(y=1); θ_c[j, v] = P(λ_j = v | y = c), v ∈ {trắng, genuine, spam}.
    EM: khởi tạo bằng bỏ phiếu đa số (mềm) → M-step ước lượng π, θ → E-step tính P(y=1 | λ).
    Trả về (posterior, π, bảng θ theo LF).
    """
    V = np.stack([(L == ABSTAIN), (L == GENUINE), (L == SPAM)], axis=2).astype(float)   # n × m × 3
    pos, neg = (L == SPAM).sum(1), (L == GENUINE).sum(1)
    q = (pos + 0.5) / (pos + neg + 1.0)
    prev = -np.inf
    for _ in range(n_iter):
        pi = float(np.clip(q.mean(), 1e-3, 1 - 1e-3))
        th1 = (q[:, None, None] * V).sum(0) + smooth
        th0 = ((1 - q)[:, None, None] * V).sum(0) + smooth
        th1 /= th1.sum(1, keepdims=True)
        th0 /= th0.sum(1, keepdims=True)
        l1 = np.log(pi) + (V * np.log(th1)[None]).sum((1, 2))
        l0 = np.log(1 - pi) + (V * np.log(th0)[None]).sum((1, 2))
        mx = np.maximum(l1, l0)
        ll = float((mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx))).sum())
        q = 1.0 / (1.0 + np.exp(l0 - l1))
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    theta = pd.DataFrame({"P(vote spam | spam)": th1[:, 2], "P(vote spam | genuine)": th0[:, 2],
                          "P(vote genuine | spam)": th1[:, 1], "P(vote genuine | genuine)": th0[:, 1]})
    return q, pi, theta


def lf_summary(L: pd.DataFrame, theta: pd.DataFrame) -> pd.DataFrame:
    """Độ phủ / chồng lấp / xung đột (thuật ngữ của Snorkel) + tham số label model đã học."""
    A = L.to_numpy()
    fired = A != ABSTAIN
    n_fired = fired.sum(1)
    has_spam, has_gen = (A == SPAM).any(1), (A == GENUINE).any(1)
    rows = []
    for j, c in enumerate(L.columns):
        f = fired[:, j]
        pol = "spam" if (A[f, j] == SPAM).all() else "genuine"
        conflict = f & (has_gen if pol == "spam" else has_spam)
        rows.append({"LF": c, "votes": pol, "description": LF_DESC.get(c, ""), "coverage": f.mean(),
                     "overlap": (f & (n_fired > 1)).mean(), "conflict": conflict.mean(),
                     **theta.iloc[j].to_dict()})
    return pd.DataFrame(rows)


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("ws", P.logs)
    S = pd.read_parquet(P.processed / "scored_reviews.parquet").reset_index(drop=True)
    L = labeling_functions(S)
    q, pi, theta = dawid_skene(L.to_numpy())
    out = pd.concat([S[["review_id"]], pd.Series(q, name="ws_score"), L], axis=1)
    f = P.processed / "ws_scores.parquet"
    out.to_parquet(f, index=False)
    summ = lf_summary(L, theta)
    write_table(summ.round(4), P.tables / "ws_lf_summary.csv")
    log.info("WS: %d review, phủ ≥1 LF = %.1f%%, π ước lượng = %.3f, P(spam) ≥ 0,99: %.1f%% corpus",
             len(S), 100 * (L.to_numpy() != ABSTAIN).any(1).mean(), pi, 100 * (q >= 0.99).mean())
    return f
