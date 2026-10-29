"""Thí nghiệm: PU learning và weak supervision trên toàn corpus, đánh giá trên nhãn labeltool.

Không phương pháp nào dùng nhãn người khi huấn luyện; nhãn chỉ dùng để đánh giá (labeltool_eval).
Mọi biến thể được khai báo TRƯỚC khi chạy và đều được báo cáo (không chọn lọc kết quả).

Đặc trưng X (dùng chung cho mọi bộ phân loại): TF-IDF từ (1–2 gram, văn bản đã tách từ) + hồ sơ tài khoản
và review (số review, số ảnh, Local Guide, số sao, độ dài, ảnh, phản hồi chủ quán, emoji, …).
KHÔNG có đặc trưng thời gian / trùng lặp / phối hợp — các tín hiệu đó chỉ dùng để tạo tập P hoặc LF.

PU (positive–unlabeled; P = review có bằng chứng mạnh, U = phần còn lại):
  - PU Elkan–Noto: LR phân biệt P với U, dự đoán out-of-fold (thứ hạng ≡ P(y=1|x) của Elkan & Noto 2008)
  - PU Spy (Liu et al. 2002; Li et al. 2014): cài 15% P làm "gián điệp" vào U, lấy tập âm tin cậy RN
    (U có điểm thấp hơn phân vị 5% của gián điệp), rồi LR phân biệt P với RN
Weak supervision (Ratner et al. 2016, data programming):
  - 8 labeling function (LF) một chiều → label model Dawid–Skene (EM) → xác suất spam  [src/fusion/ws_label_model.py]
  - End model: LR trên X học từ nhãn mềm của label model (out-of-fold)
Chạy: python -m src.evaluation.exp_pu_ws
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.evaluation import labeltool_eval as E
from src.fusion.score import MODULES, suspicion
from src.fusion.ws_label_model import dawid_skene, labeling_functions
from src.utils.common import pct_rank
from src.utils.config import Paths, load_config

SEED = 42
FOLDS = 5


# ----------------------------------------------------------------------------- đặc trưng
def build_X(S: pd.DataFrame) -> sp.csr_matrix:
    text = S["text_seg"].fillna("").astype(str).str.lower()
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                         token_pattern=r"(?u)\b\w+\b")
    Xt = tf.fit_transform(text)
    nrev = S["reviewer_n_reviews"]
    lg = S["reviewer_local_guide"].fillna(False).astype(float)
    place_mean = S.groupby("place_key")["rating"].transform("mean")
    M = pd.DataFrame({
        "log_nrev": np.log1p(nrev.fillna(nrev.median())), "nrev_na": nrev.isna().astype(float),
        "log_nphoto_acc": np.log1p(S["reviewer_n_photos"].fillna(0)), "local_guide": lg,
        **{f"star{k}": (S["rating"] == k).astype(float) for k in range(1, 6)},
        "rating_dev": S["rating"] - place_mean,
        "has_text": S["has_text"].astype(float), "log_nwords": np.log1p(S["n_words"].fillna(0)),
        "log_nphoto": np.log1p(S["n_photos"].fillna(0)), "owner_resp": S["has_owner_response"].fillna(False).astype(float),
        "emoji": (S["n_emojis"].fillna(0) > 0).astype(float), "exclaim": (S["n_exclaim"].fillna(0) > 0).astype(float),
        "upper": S["upper_ratio"].fillna(0), "edited": S["is_edited"].fillna(False).astype(float),
    })
    Xm = StandardScaler().fit_transform(M.to_numpy(float))
    return sp.hstack([Xt, sp.csr_matrix(Xm)]).tocsr()


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")


# ----------------------------------------------------------------------------- tín hiệu mạnh / LF
def strong_signals(S: pd.DataFrame) -> pd.DataFrame:
    ext = S["rating"].isin([1, 5])
    return pd.DataFrame({
        "dup": (S["dup_cross_user"].fillna(0) >= 1) & (S["n_words"] >= 8),      # trùng gần y hệt với TK khác
        "ad": S["content_rule"].fillna(0) > 0,                                   # link/SĐT/từ khóa quảng cáo
        "coord": S["s_coord_raw"].fillna(0) > 0,                                 # cùng nội dung ở quán khác trong ±14 ngày
        "burst": (S["burst_z"].fillna(0) >= 3) & (S["burst_step_n"].fillna(0) >= 5) & ext,  # đợt dồn mạnh, cùng chiều
    })


# ----------------------------------------------------------------------------- PU / end model (out-of-fold)
def oof(X, fit_fn, n: int) -> np.ndarray:
    out = np.zeros(n)
    for tr, te in KFold(FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)):
        out[te] = fit_fn(tr, te)
    return out


def pu_elkan_noto(X, s: np.ndarray) -> np.ndarray:
    def f(tr, te):
        return lr().fit(X[tr], s[tr]).predict_proba(X[te])[:, 1]
    return oof(X, f, len(s))


def pu_spy(X, s: np.ndarray, spy_frac=0.15, noise_q=0.05) -> np.ndarray:
    rng = np.random.default_rng(SEED)

    def f(tr, te):
        str_ = s[tr]
        P_idx, U_idx = tr[str_ == 1], tr[str_ == 0]
        spies = rng.choice(P_idx, int(spy_frac * len(P_idx)), replace=False)
        y1 = np.zeros(len(tr), int)
        y1[np.isin(tr, np.setdiff1d(P_idx, spies))] = 1
        m1 = lr().fit(X[tr], y1)
        thr = np.quantile(m1.predict_proba(X[spies])[:, 1], noise_q)
        RN = U_idx[m1.predict_proba(X[U_idx])[:, 1] < thr]
        idx = np.concatenate([P_idx, RN])
        y2 = np.r_[np.ones(len(P_idx)), np.zeros(len(RN))]
        return lr().fit(X[idx], y2).predict_proba(X[te])[:, 1]
    return oof(X, f, len(s))


def ws_end_model(X, q: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """LR trên nhãn mềm: mỗi mục xuất hiện 2 lần (y=1 trọng số q, y=0 trọng số 1−q)."""
    def f(tr, te):
        tr = tr[covered[tr]]
        Xd = sp.vstack([X[tr], X[tr]])
        yd = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        wd = np.r_[q[tr], 1 - q[tr]]
        m = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(Xd, yd, sample_weight=wd)
        return m.predict_proba(X[te])[:, 1]
    return oof(X, f, len(q))


# ----------------------------------------------------------------------------- chạy
def run(cfg: dict):
    P = Paths(cfg).ensure()
    D, S = E.load(P)
    S = S.reset_index(drop=True)
    print(f"Corpus {len(S)} review; tạo đặc trưng…")
    X = build_X(S)
    print("X:", X.shape)

    st = strong_signals(S)
    print("Tín hiệu mạnh (tập P):", st.sum().to_dict(), "| hợp:", int(st.any(axis=1).sum()))
    s_pu = st.any(axis=1).to_numpy().astype(int)
    s_sys = S["zone"].isin(["review", "auto_flag"]).to_numpy().astype(int)

    cand = {}
    print("PU Elkan–Noto…")
    cand["PU Elkan–Noto (P = bằng chứng mạnh)"] = pu_elkan_noto(X, s_pu)
    print("PU Spy…")
    cand["PU Spy (P = bằng chứng mạnh)"] = pu_spy(X, s_pu)
    print("PU Elkan–Noto (P = top 10% hệ thống)…")
    cand["PU Elkan–Noto (P = top 10% hệ thống)"] = pu_elkan_noto(X, s_sys)

    Lf = labeling_functions(S)
    Lm = Lf.to_numpy()
    print("LF coverage:", (Lm != -1).mean(0).round(3).tolist(), "| phủ ≥1 LF:", round(((Lm != -1).any(1)).mean(), 3))
    q, pi, acc = dawid_skene(Lm)
    acc.index = Lf.columns
    print(f"Label model: π(spam) = {pi:.3f}\n{acc.round(3)}")
    cand["WS label model (Dawid–Skene)"] = q
    cand["WS bỏ phiếu đa số (#spam − #genuine)"] = (Lm == 1).sum(1) - (Lm == 0).sum(1) + 0.0
    print("WS end model…")
    cand["WS end model (LR trên nhãn mềm)"] = ws_end_model(X, q, (Lm != -1).any(1))

    # Hệ thống Fisher với M3 sửa (trung bình hạng đặc trưng định hướng), như lượt trước
    feats = ["b_newness", "b_not_local_guide", "b_few_photos_acc", "b_extreme", "b_shortness", "b_same_day_places"]
    S2 = S.copy()
    S2["s_behav"] = pct_rank(pd.concat([S[f].astype(float).rank(pct=True) for f in feats], axis=1).mean(axis=1))
    cand["HT Fisher + M3 sửa (post-hoc)"] = suspicion(S2, list(MODULES), None, "fisher").to_numpy()

    m = pd.DataFrame(cand)
    m["review_id"] = S["review_id"]
    m.to_parquet(P.processed / "exp_pu_ws_scores.parquet", index=False)

    # Đánh giá: gộp với baseline của labeltool_eval
    Dm = D.merge(m, on="review_id", how="left")
    base = E.baseline_scores(D, S, P, cfg)
    keep = ["HỆ THỐNG (Fisher, chốt trước)", "B: Checklist 4 luật", "B: Tài khoản mới (−số review)",
            "B: TF-IDF+LR ViSpamReviews", "B: Isolation Forest thô", "Chỉ M2a burst"]
    scores = {k: base[k] for k in keep}
    scores.update({k: Dm[k].fillna(0).to_numpy() for k in cand})
    main = "WS label model (Dawid–Skene)"
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 42)
    res = []
    for pol, unc in E.UNC.items():
        yv = D["label"].map({"Spam": 1, "Genuine": 0, "Uncertain": unc})
        mk = yv.notna().to_numpy()
        y = yv[mk].astype(int).to_numpy()
        t = E.compare(y, {k: v[mk] for k, v in scores.items()}, main, rng)
        t.insert(1, "Uncertain", pol)
        res.append(t)
        print(f"\n=== {pol} (n={len(y)}, spam={y.sum()}) — Δ = WS label model − phương pháp ===")
        print(t[["phương pháp", "ROC-AUC", "ROC-AUC CI", "PR-AUC", "PR-AUC CI", "P@50",
                 "ΔAUC (HT−PP)", "p DeLong (Holm)", "ΔPR-AUC", "p boot ΔPR-AUC (Holm)"]].round(3).to_string(index=False))
        if pol == "loại Uncertain":
            Dk = D[mk].reset_index(drop=True)
            rows = []
            for pk, g in Dk.groupby("place_key"):
                yy = y[g.index]
                if 0 < yy.sum() < len(yy):
                    rows.append({"quán": pk, "n": len(yy), "spam": int(yy.sum()),
                                 **{k[:26]: E.roc_auc_score(yy, scores[k][mk][g.index]) for k in scores}})
            print("\nAUC theo quán:\n", pd.DataFrame(rows).round(3).T.to_string())
            # Điểm vận hành theo luật quyết định TỰ NHIÊN của từng phương pháp (không cần hạn ngạch)
            dec = {"WS label model: P(spam) ≥ 0,5": scores[main][mk] >= 0.5,
                   "Hệ thống: vùng review+auto_flag": Dk["zone"].isin(["review", "auto_flag"]).to_numpy(),
                   "Checklist: ≥ 3/4 luật": scores["B: Checklist 4 luật"][mk] >= 3,
                   "Checklist: 4/4 luật": scores["B: Checklist 4 luật"][mk] >= 4}
            ref = next(iter(dec.values())) == (y == 1)
            rows = []
            for k, pred in dec.items():
                tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
                pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                ok = pred == (y == 1)
                b, c = int((ref & ~ok).sum()), int((~ref & ok).sum())
                rows.append({"luật quyết định": k, "gắn cờ": int(pred.sum()), "TP": tp, "FP": fp, "FN": fn,
                             "Precision": pr, "P CI": "[%.2f, %.2f]" % E.wilson(tp, tp + fp),
                             "Recall": rc, "R CI": "[%.2f, %.2f]" % E.wilson(tp, tp + fn),
                             "F1": 2 * pr * rc / max(pr + rc, 1e-9), "Accuracy": ok.mean(),
                             "McNemar vs WS (b/c, p)": "" if k.startswith("WS") else
                             f"{b}/{c}, p={E.stats.binomtest(b, b + c, .5).pvalue if b + c else 1:.4f}"})
            print("\nĐiểm vận hành:\n", pd.DataFrame(rows).round(3).to_string(index=False))
    pd.concat(res).round(4).to_csv(P.tables / "exp_pu_ws_compare.csv", index=False, encoding="utf-8-sig")
    return scores, D


if __name__ == "__main__":
    run(load_config())
