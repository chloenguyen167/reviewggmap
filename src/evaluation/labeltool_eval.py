"""Đánh giá hệ thống trên nhãn theo quán (công cụ labeltool: data/labels/nhan_review_*.csv).

Khác gold set (nhóm A chọn theo điểm hệ thống): ở đây người gán gán TOÀN BỘ review có chữ/ảnh của
từng quán, nên mẫu không bị chọn theo điểm → precision/recall/AUC đọc trực tiếp, không cần hiệu chỉnh.

So sánh (điểm lớn = đáng ngờ; NaN → 0):
  - HỆ THỐNG: suspicion (Fisher, cấu hình chính đã chốt trước) + 2 cách gộp khác (mean, max)
  - Từng module riêng lẻ (ablation)
  - Baseline ngây thơ: ngẫu nhiên, tài khoản mới, review ngắn, sao cực đoan, checklist 4 luật
  - Baseline "phương pháp khác": Isolation Forest thuần trên đặc trưng thô (không định hướng),
    TF-IDF + LR học trên ViSpamReviews (giám sát, khác miền)
  - Tham chiếu: LR học trọng số module (leave-one-place-out CV, dùng nhãn — không phải đối thủ công bằng)

Thống kê:
  - ROC-AUC, PR-AUC (AP), P@K (xử lý điểm bằng nhau theo kỳ vọng), CI 95% bootstrap phân tầng theo lớp
  - So với ngẫu nhiên: kiểm định hoán vị (AP) và Mann–Whitney (≡ AUC)
  - So hệ thống với từng baseline: DeLong (ΔAUC), bootstrap ghép cặp (ΔAP, ΔP@K), hiệu chỉnh Holm
  - Điểm vận hành: vùng review+auto_flag (top 10% corpus) so với top-n cùng ngân sách của baseline, McNemar
  - Độ vững: 3 cách xử lý Uncertain, theo quán, theo người gán, có chữ / chỉ ảnh
Chạy: python -m src.evaluation.labeltool_eval
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.fusion.score import MODULES, suspicion
from src.utils.config import Paths, load_config

SEED = 42
N_BOOT = 2000
N_PERM = 5000
K_LIST = [25, 50, 85]
UNC = {"loại Uncertain": None, "Uncertain = Spam": 1, "Uncertain = Genuine": 0}


# ----------------------------------------------------------------------------- chỉ số
def p_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """P@K kỳ vọng khi phá thế hòa ngẫu nhiên (nhiều điểm module = 0 bằng nhau)."""
    if k > len(y):
        return float("nan")
    t = np.sort(s)[::-1][k - 1]
    above, tie = s > t, s == t
    need = k - above.sum()
    return float((y[above].sum() + need * y[tie].mean()) / k)


def metrics(y, s) -> dict:
    out = {"ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s)}
    out.update({f"P@{k}": p_at_k(y, s, k) for k in K_LIST})
    return out


def delong(y, s1, s2) -> tuple[float, float]:
    """DeLong (Sun & Xu 2014) cho 2 AUC ghép cặp → (ΔAUC, p hai phía)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(s):
        X, Y = s[pos], s[neg]
        V10 = ((X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])).mean(axis=1)
        V01 = ((X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])).mean(axis=0)
        return V10.mean(), V10, V01

    a1, v10_1, v01_1 = comps(s1)
    a2, v10_2, v01_2 = comps(s2)
    S10 = np.cov(np.vstack([v10_1, v10_2]))
    S01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    if var <= 0:
        return a1 - a2, float("nan")
    z = (a1 - a2) / np.sqrt(var)
    return a1 - a2, float(2 * stats.norm.sf(abs(z)))


def strat_boot_idx(y, rng):
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])


def holm(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    order = p.sort_values().index
    m, adj, run = len(p), {}, 0.0
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[ix]))
        adj[ix] = run
    return pd.Series(adj)[p.index]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------------------- dữ liệu + điểm
def load(P: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    fs = sorted(glob.glob(str(P.labels / "nhan_review_*.csv")))
    L = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in fs], ignore_index=True)
    L = L.drop_duplicates("review_id", keep="last")
    S = pd.read_parquet(P.processed / "scored_reviews.parquet")
    keep = ["annotator", "review_id", "label", "label_theo_luat", "has_image"] + \
           [c for c in L.columns if c[:1] == "f" and c[1:2].isdigit()]
    return L[keep].merge(S, on="review_id", how="left", validate="1:1"), S


def baseline_scores(D: pd.DataFrame, S: pd.DataFrame, P: Paths, cfg: dict) -> dict[str, np.ndarray]:
    mods = list(MODULES)
    sc = {
        "HỆ THỐNG (Fisher, chốt trước)": D["suspicion"],
        "Gộp trung bình (mean)": suspicion(D, mods, cfg["fusion"]["weights"], "mean"),
        "Gộp max": suspicion(D, mods, cfg["fusion"]["weights"], "max"),
        "Chỉ M1a trùng lặp": D["s_dup"], "Chỉ M1b nội dung (luật)": D["s_content"],
        "Chỉ M2a burst": D["s_burst"], "Chỉ M2b phối hợp": D["s_coord"], "Chỉ M3 hành vi": D["s_behav"],
        "B: Tài khoản mới (−số review)": -D["reviewer_n_reviews"].fillna(D["reviewer_n_reviews"].median()),
        "B: Review ngắn (−số từ)": -D["n_words"].fillna(0),
        "B: Sao cực đoan |sao−3|": (D["rating"] - 3).abs(),
        "B: Checklist 4 luật": ((D["reviewer_n_reviews"] <= 3).astype(int) + D["rating"].isin([1, 5]).astype(int)
                                + (~D["reviewer_local_guide"].fillna(False).astype(bool)).astype(int)
                                + (D["n_words"].fillna(0) < 8).astype(int)),
    }
    # Isolation Forest "thuần": đặc trưng thô, không định hướng/cắt một phía, học trên toàn corpus
    raw = ["reviewer_n_reviews", "reviewer_n_photos", "reviewer_local_guide", "rating", "n_chars", "n_words",
           "n_photos", "n_urls", "n_phones", "n_emojis", "n_exclaim", "upper_ratio", "has_owner_response"]
    X = S[raw].astype(float)
    X = X.fillna(X.median())
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X)
    s_if = pd.Series(-iso.score_samples(X), index=S["review_id"])
    sc["B: Isolation Forest thô"] = D["review_id"].map(s_if)
    # Mô hình giám sát học trên ViSpamReviews (TMĐT) — đúng kiểu "dùng benchmark có sẵn"
    mf = P.models / cfg["m1b"]["model_file"]
    if mf.exists():
        import joblib
        model = joblib.load(mf)
        p = np.zeros(len(D))
        t = D["text_norm"].fillna("").astype(str)
        has = t.str.len() > 0
        p[has.to_numpy()] = model.predict_proba(t[has])[:, 1]
        sc["B: TF-IDF+LR ViSpamReviews"] = p
    return {k: np.nan_to_num(np.asarray(v, dtype=float), nan=0.0) for k, v in sc.items()}


def lopo_lr(D: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Tham chiếu có giám sát: LR trên 5 điểm module, leave-one-place-out (không rò rỉ nhãn trong quán)."""
    X = D[[f"s_{m}" for m in MODULES]].fillna(0).to_numpy()
    out = np.zeros(len(D))
    for pk in D["place_key"].unique():
        te = (D["place_key"] == pk).to_numpy()
        if len(set(y[~te])) < 2:
            continue
        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[~te], y[~te])
        out[te] = lr.predict_proba(X[te])[:, 1]
    return out


# ----------------------------------------------------------------------------- chạy
def compare(y, scores: dict, main: str, rng) -> pd.DataFrame:
    base = {k: metrics(y, s) for k, s in scores.items()}
    boots = {k: [] for k in scores}
    for _ in range(N_BOOT):
        ix = strat_boot_idx(y, rng)
        for k, s in scores.items():
            boots[k].append(metrics(y[ix], s[ix]))
    rows = []
    for k in scores:
        b = pd.DataFrame(boots[k])
        r = {"phương pháp": k}
        for c in base[k]:
            r[c] = base[k][c]
            r[f"{c} CI"] = f"[{b[c].quantile(.025):.3f}, {b[c].quantile(.975):.3f}]"
        if k != main:
            d, p = delong(y, scores[main], scores[k])
            r["ΔAUC (HT−PP)"], r["p DeLong"] = d, p
            bm, bk = pd.DataFrame(boots[main]), b
            for c in ("PR-AUC", "P@50"):
                diff = bm[c] - bk[c]
                r[f"Δ{c}"] = base[main][c] - base[k][c]
                r[f"Δ{c} CI"] = f"[{diff.quantile(.025):+.3f}, {diff.quantile(.975):+.3f}]"
                r[f"p boot Δ{c}"] = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
        rows.append(r)
    t = pd.DataFrame(rows)
    for c in ("p DeLong", "p boot ΔPR-AUC", "p boot ΔP@50"):
        m = t[c].notna()
        t.loc[m, c + " (Holm)"] = holm(t.loc[m, c]).values
    return t


def vs_random(y, s, rng) -> dict:
    ap = average_precision_score(y, s)
    perm = np.array([average_precision_score(rng.permutation(y), s) for _ in range(N_PERM)])
    u = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
    return {"AP": ap, "AP kỳ vọng ngẫu nhiên": y.mean(), "p hoán vị (AP)": (1 + (perm >= ap).sum()) / (1 + N_PERM),
            "AUC": roc_auc_score(y, s), "p Mann–Whitney (AUC>0.5)": u.pvalue}


def operating_point(D, y, scores, main) -> pd.DataFrame:
    """Vùng review+auto_flag của hệ thống (top 10% corpus) vs top-n cùng ngân sách của từng phương pháp."""
    flag = D["zone"].isin(["review", "auto_flag"]).to_numpy()
    n = int(flag.sum())
    rows = []
    for k, s in scores.items():
        pred = flag if k == main else np.isin(np.arange(len(s)), np.argsort(-s, kind="stable")[:n])
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        r = {"phương pháp": k, "n gắn cờ": n, "TP": tp, "FP": fp, "FN": fn,
             "Precision": tp / max(tp + fp, 1), "Precision CI": "[%.3f, %.3f]" % wilson(tp, tp + fp),
             "Recall": tp / max(tp + fn, 1), "Recall CI": "[%.3f, %.3f]" % wilson(tp, tp + fn)}
        r["F1"] = 2 * r["Precision"] * r["Recall"] / max(r["Precision"] + r["Recall"], 1e-9)
        if k != main:
            c_main, c_k = flag == (y == 1), pred == (y == 1)
            b, c = int((c_main & ~c_k).sum()), int((~c_main & c_k).sum())
            r["McNemar b/c"] = f"{b}/{c}"
            r["p McNemar (exact)"] = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append(r)
    return pd.DataFrame(rows)


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    rng = np.random.default_rng(SEED)
    D, S = load(P)
    main = "HỆ THỐNG (Fisher, chốt trước)"
    allsc = baseline_scores(D, S, P, cfg)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40)

    print(f"Nhãn: {len(D)} review, {D['place_key'].nunique()} quán, người gán: "
          f"{D['annotator'].value_counts().to_dict()} | nhãn: {D['label'].value_counts().to_dict()}")
    print(f"label == label_theo_luat: {(D['label'] == D['label_theo_luat']).mean():.1%}")

    out = {}
    for pol, unc in UNC.items():
        yv = D["label"].map({"Spam": 1, "Genuine": 0, "Uncertain": unc})
        m = yv.notna().to_numpy()
        y = yv[m].astype(int).to_numpy()
        sc = {k: v[m] for k, v in allsc.items()}
        sc["Tham chiếu: LR học trọng số module (LOPO-CV, có nhãn)"] = lopo_lr(D[m].reset_index(drop=True), y)
        t = compare(y, sc, main, rng)
        t.insert(1, "Uncertain", pol)
        t.insert(2, "n / n_spam", f"{len(y)} / {int(y.sum())}")
        out[pol] = t
        if pol == "loại Uncertain":
            print("\n=== So với ngẫu nhiên (loại Uncertain) ===")
            print(pd.Series(vs_random(y, sc[main], rng)).round(5).to_string())
            op = operating_point(D[m].reset_index(drop=True), y, sc, main)
            print("\n=== Điểm vận hành: vùng review+auto_flag (top 10% corpus) vs top-n cùng ngân sách ===")
            print(op.round(3).to_string(index=False))
            op.round(4).to_csv(P.tables / "labeltool_eval_operating_point.csv", index=False, encoding="utf-8-sig")
            # Độ vững theo quán / người gán / loại review
            grp = []
            Dm = D[m].reset_index(drop=True)
            for name, key in (("quán", Dm["place_name"] if "place_name" in Dm else Dm["place_key"]),
                              ("người gán", Dm["annotator"]),
                              ("loại", np.where(Dm["has_text"].astype(bool), "có chữ", "chỉ ảnh"))):
                for g in pd.unique(key):
                    gm = (key == g)
                    gm = np.asarray(gm)
                    yy = y[gm]
                    r = {"nhóm": name, "giá trị": g, "n": int(gm.sum()), "n_spam": int(yy.sum())}
                    if 0 < yy.sum() < len(yy):
                        for k in (main, "B: Checklist 4 luật", "B: Isolation Forest thô", "B: TF-IDF+LR ViSpamReviews"):
                            if k in sc:
                                r[f"AUC {k[:22]}"] = roc_auc_score(yy, sc[k][gm])
                        r["AP hệ thống"] = average_precision_score(yy, sc[main][gm])
                        r["AP ngẫu nhiên"] = yy.mean()
                    grp.append(r)
            gt = pd.DataFrame(grp)
            print("\n=== Theo nhóm (loại Uncertain) ===")
            print(gt.round(3).to_string(index=False))
            gt.round(4).to_csv(P.tables / "labeltool_eval_by_group.csv", index=False, encoding="utf-8-sig")

    allt = pd.concat(out.values(), ignore_index=True)
    allt.round(4).to_csv(P.tables / "labeltool_eval_compare.csv", index=False, encoding="utf-8-sig")
    show = ["phương pháp", "ROC-AUC", "ROC-AUC CI", "PR-AUC", "PR-AUC CI", "P@50", "ΔAUC (HT−PP)",
            "p DeLong (Holm)", "ΔPR-AUC", "p boot ΔPR-AUC (Holm)"]
    for pol, t in out.items():
        print(f"\n=== So sánh phương pháp — {pol} (n / n_spam = {t['n / n_spam'].iloc[0]}) ===")
        print(t[show].round(3).to_string(index=False))

    # Ảnh hưởng của từng cờ lên nhãn (để bàn về tính vòng tròn: cờ f4/f6 ≈ đặc trưng M3/M2)
    yv = D["label"].map({"Spam": 1, "Genuine": 0})
    m = yv.notna()
    fl = [c for c in D.columns if c[:1] == "f" and c[1:2].isdigit()]
    ft = pd.DataFrame([{"cờ": c, "tỷ lệ bật": D.loc[m, c].mean(),
                        "P(Spam | cờ=1)": yv[m][D.loc[m, c] == 1].mean(),
                        "P(Spam | cờ=0)": yv[m][D.loc[m, c] == 0].mean()} for c in fl])
    print("\n=== Cờ của người gán vs nhãn (loại Uncertain) ===")
    print(ft.round(3).to_string(index=False))
    return P.tables / "labeltool_eval_compare.csv"


if __name__ == "__main__":
    run(load_config())
