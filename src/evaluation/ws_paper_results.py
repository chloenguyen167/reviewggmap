"""Toàn bộ số liệu + hình cho paper/slide của phương pháp chính (WS + Dawid–Skene).

Cần chạy trước: --stages ws  và  python -m src.evaluation.exp_pu_ws (điểm PU Spy).
Đầu ra: reports/tables/paper_*.csv, reports/figures/paper_*.pdf|png
Chạy: python -m src.evaluation.ws_paper_results
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.evaluation import labeltool_eval as E
from src.fusion.ws_label_model import dawid_skene
from src.utils import plotstyle as ps
from src.utils.config import Paths, load_config

SEED = 42
OURS = "WS + Dawid–Skene (ours)"
NAMES = {  # tên trong paper → nguồn điểm
    OURS: "ours", "WS majority vote": "mv", "Score fusion (Fisher)": "HỆ THỐNG (Fisher, chốt trước)",
    "Rule checklist": "B: Checklist 4 luật", "Account newness": "B: Tài khoản mới (−số review)",
    "PU learning (Spy)": "pu", "Supervised transfer (ViSpamReviews)": "B: TF-IDF+LR ViSpamReviews",
    "Isolation Forest": "B: Isolation Forest thô",
}
# màu đã kiểm định (validate_palette.js, light): ours = teal, rồi blue, amber, violet; còn lại xám
COL = {OURS: "#1CA893", "Score fusion (Fisher)": "#2E75B6", "Rule checklist": "#B46E0A",
       "PU learning (Spy)": "#7A4FA0"}
GREY = "#9AA1A9"


def inclusion_prob(s: np.ndarray, budget: int) -> np.ndarray:
    """Xác suất mỗi review nằm trong top-`budget` khi phá thế hòa ngẫu nhiên."""
    t = np.sort(s)[::-1][budget - 1]
    p = (s > t).astype(float)
    tie = s == t
    p[tie] = (budget - (s > t).sum()) / tie.sum()
    return p


def main():
    cfg = load_config()
    P = Paths(cfg).ensure()
    D, S = E.load(P)
    S = S.reset_index(drop=True)
    W = pd.read_parquet(P.processed / "ws_scores.parquet")
    X = pd.read_parquet(P.processed / "exp_pu_ws_scores.parquet")
    lf_cols = [c for c in W.columns if c.startswith("LF_")]
    Lm = W[lf_cols].to_numpy()
    W["mv"] = (Lm == 1).sum(1) - (Lm == 0).sum(1)
    W["pu"] = X["PU Spy (P = bằng chứng mạnh)"].to_numpy()
    W["ours"] = W["ws_score"]
    D = D.merge(W, on="review_id", how="left")
    base = E.baseline_scores(D, S, P, cfg)
    sc_all = {k: (D[v].to_numpy(float) if v in D else base[v]) for k, v in NAMES.items()}

    # ---------------------------------------------------------------- A. dữ liệu
    places = pd.read_parquet(P.interim / "places.parquet") if (P.interim / "places.parquet").exists() else None
    stats = {"reviews": len(S), "places": S["place_key"].nunique(), "accounts": S["reviewer_hash"].nunique(),
             "with_text": S["has_text"].mean(), "star_only": 1 - S["has_text"].mean(),
             "first_day": str(pd.to_datetime(S["review_day"]).min().date()),
             "last_day": str(pd.to_datetime(S["review_day"]).max().date()),
             "lf_coverage_any": (Lm != -1).any(1).mean(), "pi_hat": None}
    q, pi, theta = dawid_skene(Lm)
    stats["pi_hat"] = pi
    pd.Series(stats).to_csv(P.tables / "paper_data_stats.csv", header=["value"], encoding="utf-8-sig")
    lab = D.groupby("place_key").agg(n=("label", "size"), spam=("label", lambda s: (s == "Spam").sum()),
                                     genuine=("label", lambda s: (s == "Genuine").sum()),
                                     uncertain=("label", lambda s: (s == "Uncertain").sum()),
                                     annotator=("annotator", "first"))
    if places is not None and "place_name" in places:
        lab = lab.join(places.drop_duplicates("place_key").set_index("place_key")[["place_name"]])
    lab.to_csv(P.tables / "paper_label_stats.csv", encoding="utf-8-sig")
    print(pd.Series(stats), "\n", lab)

    # ---------------------------------------------------------------- B. so sánh chính (3 cách xử lý Uncertain)
    rng = np.random.default_rng(SEED)
    res = []
    for pol, unc in E.UNC.items():
        yv = D["label"].map({"Spam": 1, "Genuine": 0, "Uncertain": unc})
        m = yv.notna().to_numpy()
        y = yv[m].astype(int).to_numpy()
        t = E.compare(y, {k: v[m] for k, v in sc_all.items()}, OURS, rng)
        t.insert(1, "uncertain_policy", pol)
        t.insert(2, "n", len(y))
        t.insert(3, "n_spam", int(y.sum()))
        res.append(t)
    main_t = pd.concat(res, ignore_index=True)
    main_t.round(4).to_csv(P.tables / "paper_main_comparison.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(main_t[["uncertain_policy", "phương pháp", "ROC-AUC", "ROC-AUC CI", "PR-AUC", "PR-AUC CI", "P@50",
                  "p DeLong (Holm)", "p boot ΔPR-AUC (Holm)"]].round(3).to_string(index=False))

    # tập chính: loại Uncertain
    yv = D["label"].map({"Spam": 1, "Genuine": 0})
    m = yv.notna().to_numpy()
    y = yv[m].astype(int).to_numpy()
    sc = {k: v[m] for k, v in sc_all.items()}
    Dm = D[m].reset_index(drop=True)

    # ---------------------------------------------------------------- C. theo quán
    rows = []
    for pk, g in Dm.groupby("place_key"):
        yy = y[g.index]
        r = {"place_key": pk, "n": len(yy), "spam": int(yy.sum())}
        for k in (OURS, "Score fusion (Fisher)", "Rule checklist", "PU learning (Spy)"):
            r[k] = roc_auc_score(yy, sc[k][g.index]) if 0 < yy.sum() < len(yy) else np.nan
        rows.append(r)
    pd.DataFrame(rows).round(3).to_csv(P.tables / "paper_per_place.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- D. bảng LF: tham số học được vs độ chính xác thực nghiệm
    L_lab = Dm[lf_cols].to_numpy()
    summ = pd.read_csv(P.tables / "ws_lf_summary.csv", encoding="utf-8-sig")
    emp = []
    for j, c in enumerate(lf_cols):
        f = L_lab[:, j] != -1
        pol = 1 if (L_lab[f, j] == 1).all() and f.any() else 0
        emp.append({"LF": c, "fires_labeled": int(f.sum()),
                    "empirical_precision": (y[f] == pol).mean() if f.any() else np.nan,
                    "model_precision": None})
    emp = pd.DataFrame(emp)
    # độ chính xác theo mô hình: P(y = cực của LF | LF bỏ phiếu) = π·θ1 / (π·θ1 + (1−π)·θ0)
    for i, c in enumerate(lf_cols):
        r = summ.set_index("LF").loc[c]
        if r["votes"] == "spam":
            a, b = pi * r["P(vote spam | spam)"], (1 - pi) * r["P(vote spam | genuine)"]
        else:
            a, b = (1 - pi) * r["P(vote genuine | genuine)"], pi * r["P(vote genuine | spam)"]
        emp.loc[i, "model_precision"] = a / (a + b)
    # ablation bỏ từng LF
    full_auc, full_ap = roc_auc_score(y, sc[OURS]), average_precision_score(y, sc[OURS])
    abl = []
    for j, c in enumerate(lf_cols):
        keep = [k for k in range(len(lf_cols)) if k != j]
        qq, _, _ = dawid_skene(Lm[:, keep])
        s_ = pd.Series(qq, index=W["review_id"]).loc[Dm["review_id"]].to_numpy()
        abl.append({"LF": c, "AUC_without": roc_auc_score(y, s_), "AP_without": average_precision_score(y, s_)})
    abl = pd.DataFrame(abl)
    abl["dAUC"] = abl["AUC_without"] - full_auc
    abl["dAP"] = abl["AP_without"] - full_ap
    lf_tab = summ.merge(emp, on="LF").merge(abl, on="LF")
    lf_tab.round(4).to_csv(P.tables / "paper_lf_table.csv", index=False, encoding="utf-8-sig")
    print(lf_tab[["LF", "votes", "coverage", "conflict", "fires_labeled", "empirical_precision",
                  "model_precision", "dAUC", "dAP"]].round(3).to_string(index=False))

    # ---------------------------------------------------------------- E. AUC trong từng tầng cờ người gán (kiểm tra tính vòng tròn)
    strat = []
    for f in ("f4_tai_khoan_it_review_cung_muc_sao", "f6_trong_cum_dang_bat_thuong"):
        for v in (0, 1):
            g = (Dm[f] == v).to_numpy()
            yy = y[g]
            r = {"flag": f[:2], "value": v, "n": int(g.sum()), "spam": int(yy.sum())}
            for k in (OURS, "Score fusion (Fisher)", "Rule checklist", "Account newness"):
                r[k] = roc_auc_score(yy, sc[k][g]) if 0 < yy.sum() < len(yy) else np.nan
            strat.append(r)
    pd.DataFrame(strat).round(3).to_csv(P.tables / "paper_within_stratum.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(strat).round(3).to_string(index=False))

    # ---------------------------------------------------------------- F. điểm vận hành theo ngân sách kiểm tra (toàn corpus)
    idx = pd.Index(S["review_id"]).get_indexer(Dm["review_id"])
    corpus = {OURS: W.set_index("review_id")["ws_score"].loc[S["review_id"]].to_numpy(),
              "Score fusion (Fisher)": S["suspicion"].fillna(0).to_numpy(),
              "Rule checklist": None, "Account newness": -S["reviewer_n_reviews"].fillna(S["reviewer_n_reviews"].median()).to_numpy()}
    lg = S["reviewer_local_guide"].fillna(False).astype(bool)
    corpus["Rule checklist"] = ((S["reviewer_n_reviews"] <= 3).astype(int) + S["rating"].isin([1, 5]).astype(int)
                                + (~lg).astype(int) + (S["n_words"].fillna(0) < 8).astype(int)).to_numpy(float)
    op = []
    rb = np.random.default_rng(SEED)
    for frac in (0.05, 0.10):
        B = int(round(frac * len(S)))
        for k, s in corpus.items():
            p = inclusion_prob(s, B)[idx]
            tp, fp = (p * y).sum(), (p * (1 - y)).sum()
            boots = []
            for _ in range(2000):
                ix = rb.integers(0, len(y), len(y))
                t_, f_ = (p[ix] * y[ix]).sum(), (p[ix] * (1 - y[ix])).sum()
                boots.append((t_ / max(t_ + f_, 1e-9), t_ / max(y[ix].sum(), 1)))
            boots = np.array(boots)
            op.append({"budget": f"top {frac:.0%}", "method": k, "precision": tp / (tp + fp), "recall": tp / y.sum(),
                       "precision_CI": "[%.2f, %.2f]" % tuple(np.percentile(boots[:, 0], [2.5, 97.5])),
                       "recall_CI": "[%.2f, %.2f]" % tuple(np.percentile(boots[:, 1], [2.5, 97.5])),
                       "labeled_flagged": p.sum()})
    op = pd.DataFrame(op)
    op["F1"] = 2 * op.precision * op.recall / (op.precision + op.recall)
    op.round(4).to_csv(P.tables / "paper_operating_points.csv", index=False, encoding="utf-8-sig")
    print(op.round(3).to_string(index=False))

    # ---------------------------------------------------------------- G. hình
    import matplotlib.pyplot as plt
    ps.apply()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    show = [OURS, "Score fusion (Fisher)", "Rule checklist", "PU learning (Spy)"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for k in show:
        s = sc[k]
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        lw = 2.2 if k == OURS else 1.4
        lab = f"{k}: AUC {roc_auc_score(y, s):.3f}, AP {average_precision_score(y, s):.3f}"
        axs[0].plot(fpr, tpr, color=COL[k], lw=lw, label=lab)
        axs[1].plot(rc, pr, color=COL[k], lw=lw, drawstyle="steps-post")
    axs[0].plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    axs[1].axhline(y.mean(), color=GREY, lw=0.8, ls="--")
    axs[1].annotate(f"random classifier (AP = {y.mean():.3f})", (0.02, y.mean()), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7, color="#52514e")
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC curve", xlim=(0, 1), ylim=(0, 1.01))
    axs[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision–recall curve", xlim=(0, 1), ylim=(0, 1.01))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(P.figures / f"paper_roc_pr.{ext}", dpi=300)
    plt.close(fig)

    # forest plot AUC & AP với CI (loại Uncertain)
    t = res[0].set_index("phương pháp")
    order = list(NAMES)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for a, met in zip(axs, ("ROC-AUC", "PR-AUC")):
        for i, k in enumerate(order):
            v = t.loc[k, met]
            lo, hi = [float(x) for x in t.loc[k, f"{met} CI"].strip("[]").split(",")]
            c = COL.get(k, GREY) if k == OURS else "#52514e"
            a.plot([lo, hi], [i, i], color=COL[OURS] if k == OURS else GREY, lw=2.0 if k == OURS else 1.4,
                   solid_capstyle="round")
            a.plot([v], [i], "o", ms=5 if k == OURS else 4, color=COL[OURS] if k == OURS else "#52514e",
                   mec="white", mew=0.8)
            a.annotate(f"{v:.3f}", (hi, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                       color="#0b0b0b" if k == OURS else "#52514e", fontweight="bold" if k == OURS else "normal",
                       bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ref = 0.5 if met == "ROC-AUC" else y.mean()
        a.axvline(ref, color=GREY, lw=0.8, ls="--")
        a.set(xlabel=met, xlim=(0, 1.08), title=f"({'a' if met == 'ROC-AUC' else 'b'}) {met}, 95% bootstrap CI")
        a.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(order)), order)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(P.figures / f"paper_forest.{ext}", dpi=300)
    plt.close(fig)
    print("Xong:", P.figures)


if __name__ == "__main__":
    main()
