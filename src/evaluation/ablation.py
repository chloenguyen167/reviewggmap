"""Ablation (D5) và độ nhạy tham số (D7).

Ablation: xếp hạng lại TOÀN BỘ gold set (nhóm A + B) theo từng cấu hình:
  1–4. từng module riêng lẻ (gồm M2a burst trong quán, M2b phối hợp liên quán)   5. M1   6. M1+M2
  7a/7b/7c. Full với 3 cách gộp: trung bình / Fisher / max (7b là cấu hình chính, chốt trước khi có nhãn)
  9/10. Biến thể vị trí tín hiệu phối hợp liên quán: bỏ hẳn / gộp trong M1a như thiết kế cũ
  8. Full, trọng số học bằng Logistic Regression (dự đoán out-of-fold, k-fold CV → không rò rỉ nhãn)
Module không áp dụng được cho một review → điểm 0 cho cấu hình đơn module (có báo cáo độ phủ).

Lưu ý thiên lệch chọn mẫu: nhóm A được chọn theo chính điểm của từng module, nên so sánh
giữa các cấu hình chỉ mang tính tương đối — cần nêu trong phần Limitations.

Độ nhạy: chạy lại module liên quan với từng giá trị tham số → fusion → P@50 và PR-AUC trên gold.
Đầu ra: reports/tables/ablation.csv, sensitivity.csv, reports/figures/fig_ablation.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.evaluation.metrics import bootstrap_ci, ranking_metrics
from src.fusion.score import MODULES, load_scores, suspicion
from src.utils.common import get_logger, write_table
from src.utils.config import Paths, set_by_dotted

FULL = ["dup", "content", "burst", "coord", "behav"]
CONFIGS = [  # (tên, module, cách gộp; None = dùng fusion.method trong config)
    ("1. Chỉ M1a (trùng lặp văn bản)", ["dup"], "mean"),
    ("2. Chỉ M1b (nội dung)", ["content"], "mean"),
    ("3. Chỉ M2a (burst trong quán)", ["burst"], "mean"),
    ("3b. Chỉ M2b (phối hợp liên quán)", ["coord"], "mean"),
    ("4. Chỉ M3 (hành vi)", ["behav"], "mean"),
    ("5. M1 (a+b)", ["dup", "content"], None),
    ("6. M1 + M2 (a+b)", ["dup", "content", "burst", "coord"], None),
    ("7a. Full — gộp trung bình", FULL, "mean"),
    ("7b. Full — gộp Fisher  [CẤU HÌNH CHÍNH, chốt trước]", FULL, "fisher"),
    ("7c. Full — gộp max", FULL, "max"),
    # Biến thể vị trí tín hiệu phối hợp liên quán (xem m2_temporal.py, M2b) — so trên cùng gold set
    ("9. Biến thể: bỏ hẳn tín hiệu phối hợp (Full − M2b)", ["dup", "content", "burst", "behav"], None),
    ("10. Biến thể: phối hợp gộp trong M1a (thiết kế cũ)", ["dup_coord", "content", "burst", "behav"], None),
]


def _add_variant_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Biến thể 10: điểm M1a kiểu cũ = s_dup_raw × (1 + mức bất ngờ phối hợp), xếp hạng trên toàn corpus."""
    from src.utils.common import pct_rank
    if {"s_dup_raw", "s_coord_raw"} <= set(scores.columns):
        scores["s_dup_coord"] = pct_rank(scores["s_dup_raw"] * (1 + scores["s_coord_raw"].fillna(0)))
    else:
        scores["s_dup_coord"] = scores.get("s_dup", np.nan)
    return scores


def _gold_frame(P: Paths, scores: pd.DataFrame) -> pd.DataFrame:
    gold = pd.read_csv(P.labels / "gold_final.csv", encoding="utf-8-sig")
    g = gold.merge(scores, on="review_id", how="left", suffixes=("", "_s"))
    g = g[g["label"].isin(["Spam", "Genuine"])].copy()
    g["y"] = (g["label"] == "Spam").astype(int)
    return g


def run_ablation(cfg: dict) -> Path:
    import matplotlib.pyplot as plt
    from src.utils import plotstyle as ps
    P = Paths(cfg).ensure()
    log = get_logger("ablation", P.logs)
    e = cfg["evaluation"]
    seed = cfg["project"]["seed"]
    g = _gold_frame(P, _add_variant_scores(load_scores(P)))
    y = g["y"].to_numpy()
    rows = []
    default_method = cfg["fusion"].get("method", "mean")
    for name, mods, method in CONFIGS:
        s = suspicion(g, mods, cfg["fusion"]["weights"], method or default_method).fillna(0).to_numpy()
        m = ranking_metrics(y, s, e["k_list"])
        ci = bootstrap_ci(y, s, e["k_list"], e["bootstrap"], seed)
        cov = float(g[[f"s_{x}" for x in mods]].notna().any(axis=1).mean())
        rows.append({"cấu hình": name, **m, "PR-AUC_lo": ci["PR-AUC"][0], "PR-AUC_hi": ci["PR-AUC"][1],
                     "độ phủ": cov})

    # 8. Trọng số học bằng LR (out-of-fold)
    mods = list(MODULES)
    X = pd.concat([g[[f"s_{m}" for m in mods]].fillna(0),
                   g[[f"s_{m}" for m in mods]].isna().astype(int).add_suffix("_missing")], axis=1)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    folds = min(e["cv_folds"], n_pos, n_neg)
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        s = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
        m = ranking_metrics(y, s, e["k_list"])
        ci = bootstrap_ci(y, s, e["k_list"], e["bootstrap"], seed)
        coef = lr.fit(X, y).coef_[0][: len(mods)]
        rows.append({"cấu hình": f"8. Full, trọng số LR ({folds}-fold, out-of-fold)", **m,
                     "PR-AUC_lo": ci["PR-AUC"][0], "PR-AUC_hi": ci["PR-AUC"][1], "độ phủ": 1.0,
                     "hệ số LR": ", ".join(f"{a}={b:.2f}" for a, b in zip(mods, coef))})
    ab = pd.DataFrame(rows)
    write_table(ab.round(4), P.tables / "ablation.csv")

    ps.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    order = ab.iloc[::-1]
    colors = [ps.SERIES[0] if n.startswith(("7", "8")) else ps.SEQ[2] for n in order["cấu hình"]]
    yy = np.arange(len(order))
    ax.barh(yy, order["PR-AUC"], color=colors, height=0.6)
    ax.errorbar(order["PR-AUC"], yy, xerr=[order["PR-AUC"] - order["PR-AUC_lo"], order["PR-AUC_hi"] - order["PR-AUC"]],
                fmt="none", ecolor=ps.INK_2, elinewidth=1, capsize=2)
    for yi, v in zip(yy, order["PR-AUC"]):
        ax.text(0.01, yi, f"{v:.2f}", va="center", ha="left", fontsize=8, color=ps.SURFACE, fontweight="bold")
    ax.set_yticks(yy, order["cấu hình"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC trên gold set (thanh sai số: CI 95% bootstrap)")
    ax.set_title("Ablation: đóng góp của từng tín hiệu")
    ax.grid(axis="y", visible=False)
    ps.save(fig, P.figures / "fig_ablation.png")
    log.info("Ablation:\n%s", ab[["cấu hình", "PR-AUC", f"P@{e['k_list'][1]}"]].round(3).to_string(index=False))
    return P.tables / "ablation.csv"


def run_sensitivity(cfg: dict) -> Path:
    """Chạy lại module bị ảnh hưởng với từng giá trị tham số, ghi kết quả vào thư mục tạm."""
    import shutil
    import tempfile
    from src.fusion import score as fusion
    from src.modules import m1a_duplicate, m2_temporal
    P = Paths(cfg).ensure()
    log = get_logger("sensitivity", P.logs)
    e = cfg["evaluation"]
    k = e["k_list"][1]
    rows = []
    for dotted, values in e["sensitivity"].items():
        for v in values:
            c2 = set_by_dotted(cfg, dotted, v)
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                # Sao chép processed hiện tại, rồi chỉ chạy lại module bị ảnh hưởng
                shutil.copytree(P.processed, tmp / "processed")
                c2 = set_by_dotted(c2, "paths.processed", str(tmp / "processed"))
                c2 = set_by_dotted(c2, "paths.reports", str(tmp / "reports"))   # không ghi đè hình/bảng chính
                if dotted.startswith("m1a"):
                    m1a_duplicate.run(c2)
                    m2_temporal.run(c2)          # M2b dùng cụm trùng của M1a → chạy lại theo
                elif dotted.startswith("m2"):
                    m2_temporal.run(c2)
                P2 = Paths(c2)
                sc = fusion.load_scores(P2)
                sc["suspicion"] = fusion.suspicion(sc, list(MODULES), cfg["fusion"]["weights"],
                                                   cfg["fusion"].get("method", "mean"))
                g = _gold_frame(P, sc)
                m = ranking_metrics(g["y"].to_numpy(), g["suspicion"].fillna(0).to_numpy(), [k])
                rows.append({"tham số": dotted, "giá trị": v, f"P@{k}": m[f"P@{k}"], "PR-AUC": m["PR-AUC"],
                             "mặc định": v == _get(cfg, dotted)})
    out = pd.DataFrame(rows)
    write_table(out.round(4), P.tables / "sensitivity.csv")
    log.info("Độ nhạy:\n%s", out.round(3).to_string(index=False))
    return P.tables / "sensitivity.csv"


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node
