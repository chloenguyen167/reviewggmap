"""Chỉ số đánh giá chính (D4) trên gold set.

Nhãn nhị phân: Spam = 1, Genuine = 0; Uncertain bị loại (báo cáo riêng tỷ lệ Uncertain).

- Precision@K   : tỷ lệ Spam trong top-K mục gold theo suspicion; K ∈ k_list
- PR-AUC, ROC-AUC trên toàn bộ gold set
- Base rate     : % Spam trong Nhóm B (mẫu ngẫu nhiên đều) + khoảng Wilson 95%
- FPR nhóm B    : % review Genuine ở nhóm B mà rơi vào vùng auto_flag
- Recall ƯỚC LƯỢNG = (precision vùng auto_flag × |vùng auto_flag|) / (base rate × |corpus|)
- Khoảng tin cậy 95% bằng bootstrap (lấy mẫu lại các mục gold)
- Độ vững (metrics_robustness.csv): 3 cách xử lý Uncertain (loại / =Spam / =Genuine) × nhóm review
  (tất cả / có chữ / chỉ chấm sao) × nguồn nhãn (tất cả / người / Claude)
Đầu ra: reports/tables/metrics_main.csv, metrics_robustness.csv, pk_curve.csv, fp_top.csv, fn_top.csv,
        reports/figures/fig_pk_curve.png, fig_pr_curve.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.utils.common import get_logger, write_table
from src.utils.config import Paths


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    if k > len(y) or k <= 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    return float(np.mean(y[order[:k]]))


def ranking_metrics(y: np.ndarray, score: np.ndarray, k_list: list[int]) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    out = {f"P@{k}": precision_at_k(y, score, k) for k in k_list}
    if len(set(y)) == 2:
        out["PR-AUC"] = float(average_precision_score(y, score))
        out["ROC-AUC"] = float(roc_auc_score(y, score))
    else:
        out["PR-AUC"] = out["ROC-AUC"] = float("nan")
    return out


def bootstrap_ci(y, score, k_list, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        samples.append(ranking_metrics(y[idx], score[idx], k_list))
    s = pd.DataFrame(samples)
    return {c: (float(s[c].quantile(0.025)), float(s[c].quantile(0.975))) for c in s.columns}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


UNCERTAIN_POLICIES = {"loại Uncertain": None, "Uncertain = Spam": 1, "Uncertain = Genuine": 0}


def robustness_table(g_all: pd.DataFrame, k_list: list[int]) -> pd.DataFrame:
    """P@K / PR-AUC theo cách xử lý Uncertain, theo nhóm có chữ/chỉ sao, theo nguồn nhãn."""
    has_text = g_all["has_text"].fillna(False).astype(bool) if "has_text" in g_all else pd.Series(True, g_all.index)
    src = g_all["label_source"] if "label_source" in g_all else pd.Series("human", g_all.index)
    subsets = {"tất cả": pd.Series(True, g_all.index), "có chữ": has_text, "chỉ chấm sao": ~has_text}
    sources = {"mọi nguồn": pd.Series(True, g_all.index), "người": src == "human", "Claude": src == "llm"}
    rows = []
    for pol, unc in UNCERTAIN_POLICIES.items():
        y_all = g_all["label"].map({"Spam": 1, "Genuine": 0, "Uncertain": unc})
        for sname, smask in subsets.items():
            for oname, omask in sources.items():
                m = smask & omask & y_all.notna()
                if m.sum() == 0:
                    continue
                y = y_all[m].astype(int).to_numpy()
                r = ranking_metrics(y, g_all.loc[m, "suspicion"].fillna(0).to_numpy(), k_list)
                rows.append({"Uncertain": pol, "nhóm review": sname, "nguồn nhãn": oname, "n": int(m.sum()),
                             "n_spam": int(y.sum()), **r})
    return pd.DataFrame(rows)


def load_gold(P: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = pd.read_parquet(P.processed / "scored_reviews.parquet")
    gold = pd.read_csv(P.labels / "gold_final.csv", encoding="utf-8-sig")
    g = gold.merge(scored, on="review_id", how="left", suffixes=("", "_scored"))
    return g, scored


def run(cfg: dict) -> Path:
    import matplotlib.pyplot as plt
    from src.utils import plotstyle as ps
    P = Paths(cfg).ensure()
    log = get_logger("metrics", P.logs)
    e = cfg["evaluation"]
    g, scored = load_gold(P)
    write_table(robustness_table(g, e["k_list"]).round(4), P.tables / "metrics_robustness.csv")
    n_unc = int((g["label"] == "Uncertain").sum())
    g = g[g["label"].isin(["Spam", "Genuine"])].copy()
    g["y"] = (g["label"] == "Spam").astype(int)
    y, s = g["y"].to_numpy(), g["suspicion"].fillna(0).to_numpy()

    # Nhãn NGƯỜI là số chính (K nhỏ); K lớn mở rộng trên nhãn người + Claude — mỗi dòng ghi rõ nguồn.
    src = g["label_source"] if "label_source" in g else pd.Series("human", index=g.index)
    hm = (src == "human").to_numpy()
    rows = []
    has_llm = bool((~hm).any())
    if not has_llm:
        log.warning("Không có nhãn Claude trong gold_final → chỉ báo cáo K = %s trên nhãn người",
                    e.get("k_list_human", e["k_list"]))
    for nguon, mask, ks in (("người", hm, e.get("k_list_human", e["k_list"])),
                            ("người + Claude", np.ones(len(g), bool), e.get("k_list_extended", []))):
        if not mask.any() or (nguon != "người" and (not ks or not has_llm)):
            continue
        m_ = ranking_metrics(y[mask], s[mask], ks)
        ci = bootstrap_ci(y[mask], s[mask], ks, e["bootstrap"], cfg["project"]["seed"])
        rows += [{"chỉ số": k, "nguồn nhãn": nguon, "n": int(mask.sum()), "giá trị": v,
                  "CI95_thấp": ci[k][0], "CI95_cao": ci[k][1]} for k, v in m_.items()]
    main = {r["chỉ số"]: r["giá trị"] for r in rows if r.get("nguồn nhãn") == "người"}

    B = g[g["in_group_b"]]
    k_spam, n_b = int(B["y"].sum()), len(B)
    base = k_spam / n_b if n_b else float("nan")
    lo, hi = wilson(k_spam, n_b)
    rows.append({"chỉ số": "Base rate (nhóm B)", "nguồn nhãn": "người", "n": n_b, "giá trị": base, "CI95_thấp": lo, "CI95_cao": hi})
    gb = B[B["y"] == 0]
    fpr = float((gb["zone"] == "auto_flag").mean()) if len(gb) else float("nan")
    fk, fn_ = int((gb["zone"] == "auto_flag").sum()), len(gb)
    rows.append({"chỉ số": "FPR vùng auto_flag (nhóm B)", "nguồn nhãn": "người", "n": fn_, "giá trị": fpr,
                 "CI95_thấp": wilson(fk, fn_)[0], "CI95_cao": wilson(fk, fn_)[1]})
    flag_gold = g[g["zone"] == "auto_flag"]
    prec_flag = float(flag_gold["y"].mean()) if len(flag_gold) else float("nan")
    n_flag = int((scored["zone"] == "auto_flag").sum())
    recall_est = (prec_flag * n_flag) / (base * len(scored)) if base and base > 0 else float("nan")
    rows += [
        {"chỉ số": "Precision vùng auto_flag (mục gold trong vùng)", "giá trị": prec_flag},
        {"chỉ số": "Recall ƯỚC LƯỢNG của vùng auto_flag", "giá trị": min(recall_est, 1.0) if recall_est == recall_est else recall_est},
        {"chỉ số": "Số mục gold dùng (Spam+Genuine)", "giá trị": len(g)},
        {"chỉ số": "Số mục Uncertain (bị loại)", "giá trị": n_unc},
        {"chỉ số": "Tỷ lệ Spam trong gold", "giá trị": float(y.mean())},
    ]
    mt = pd.DataFrame(rows)
    write_table(mt.round(4), P.tables / "metrics_main.csv")

    # Đường cong P@K có dải bootstrap
    ks = np.arange(5, len(y) + 1, 5)
    rng = np.random.default_rng(cfg["project"]["seed"])
    boot = np.array([[precision_at_k(y[idx], s[idx], k) for k in ks]
                     for idx in (rng.integers(0, len(y), len(y)) for _ in range(min(e["bootstrap"], 300)))])
    pk = pd.DataFrame({"K": ks, "P@K": [precision_at_k(y, s, k) for k in ks],
                       "lo": np.nanpercentile(boot, 2.5, axis=0), "hi": np.nanpercentile(boot, 97.5, axis=0)})
    write_table(pk.round(4), P.tables / "pk_curve.csv")

    ps.apply()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.fill_between(pk["K"], pk["lo"], pk["hi"], color=ps.SEQ[0], linewidth=0, label="CI 95% (bootstrap)")
    ax.plot(pk["K"], pk["P@K"], color=ps.SERIES[0], label="Precision@K")
    ax.axhline(base, color=ps.INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"tỷ lệ spam nền ≈ {base:.1%}", (pk["K"].iloc[-1], base), textcoords="offset points",
                xytext=(0, 4), ha="right", fontsize=8, color=ps.INK_2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("K (số review nghi vấn nhất được xét)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Độ chính xác theo K")
    ax.legend(loc="upper right")
    ps.save(fig, P.figures / "fig_pk_curve.png")

    if len(set(y)) == 2:
        pr, rc, _ = precision_recall_curve(y, s)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(rc, pr, color=ps.SERIES[0])
        ax.set_xlabel("Recall (trong gold set)")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall (PR-AUC = {main['PR-AUC']:.3f})")
        ax.set_ylim(0, 1.02)
        ps.save(fig, P.figures / "fig_pr_curve.png")

    # Phân tích lỗi (D6)
    cols = ["item_id", "label", "suspicion", "zone", "s_dup", "s_content", "s_burst", "s_behav", "rating", "text", "reasons"]
    write_table(g[g["y"] == 0].nlargest(10, "suspicion")[cols], P.tables / "fp_top.csv")
    write_table(g[g["y"] == 1].nsmallest(10, "suspicion")[cols], P.tables / "fn_top.csv")
    pr = next(r for r in rows if r["chỉ số"] == "PR-AUC" and r.get("nguồn nhãn") == "người")
    log.info("[nhãn người] %s, PR-AUC=%.3f [%.3f–%.3f] | base rate=%.3f, FPR(B)=%.3f | %s",
             ", ".join(f"{k}={v:.3f}" for k, v in main.items() if k.startswith("P@")), pr["giá trị"],
             pr["CI95_thấp"], pr["CI95_cao"], base, fpr,
             ", ".join(f"{r['chỉ số']}={r['giá trị']:.3f} (người + Claude)" for r in rows
                       if r.get("nguồn nhãn") == "người + Claude" and r["chỉ số"].startswith("P@")))
    return P.tables / "metrics_main.csv"
