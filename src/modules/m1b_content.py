"""Module 1b — Bộ phân loại nội dung spam (có giám sát, chuyển miền).

Huấn luyện: ViSpamReviews v2 (e-commerce, tiếng Việt) → TF-IDF (từ 1–2 gram + ký tự 3–5 gram)
            + Logistic Regression; chọn C trên tập dev.
Áp dụng   : review Google Maps tiếng Việt có chữ → s_content = P(spam).
Không có benchmark: dùng điểm luật (URL/SĐT/từ khóa quảng cáo/emoji/in hoa) và ghi rõ trong paper.

Lưu ý chuyển miền: "spam" trong ViSpamReviews chủ yếu là review KHÔNG LIÊN QUAN / quảng cáo
trên sàn TMĐT, không hoàn toàn trùng với "review không có trải nghiệm thật" trên Google Maps.
Đầu ra: processed/features_m1b.parquet, reports/tables/m1b_benchmark.csv, m1b_gmaps_spotcheck.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.preprocess.normalize_vi import load_teencode, normalize
from src.utils.common import get_logger, write_table
from src.utils.config import Paths

SPLIT_NAMES = {"train": ["train"], "dev": ["dev", "val", "valid", "validation"], "test": ["test"]}


# ---------------------------------------------------------------------------
# Nạp benchmark (tự dò tên file / cột vì cấu trúc chưa được công bố)
# ---------------------------------------------------------------------------
def _read_any(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".json", ".jsonl"):
        return pd.read_json(p, lines=p.suffix == ".jsonl")
    return pd.read_csv(p)


def _detect_cols(df: pd.DataFrame, text_col, label_col) -> tuple[str, str]:
    if text_col is None:
        obj = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    if label_col is None:
        pref = [c for c in df.columns if c.lower() in ("label", "labels", "is_spam", "spam")]
        cand = pref or [c for c in df.columns if c != text_col and df[c].nunique() == 2]
        if not cand:
            raise ValueError(f"Không dò được cột nhãn nhị phân trong {list(df.columns)} — đặt m1b.label_col")
        label_col = cand[0]
    return text_col, label_col


def load_benchmark(P: Paths, cfg: dict) -> dict[str, pd.DataFrame]:
    c = cfg["m1b"]
    d = P.benchmark / c["benchmark_dir"]
    files = [p for p in d.rglob("*") if p.suffix in (".csv", ".parquet", ".json", ".jsonl")] if d.exists() else []
    if not files:
        return {}
    splits = {}
    for name, keys in SPLIT_NAMES.items():
        hit = [p for p in files if any(k in re.split(r"[-_. ]", p.stem.lower()) for k in keys)]
        if hit:
            splits[name] = pd.concat([_read_any(p) for p in hit], ignore_index=True)
    if "train" not in splits:                               # một file duy nhất → tự chia 80/10/10
        full = pd.concat([_read_any(p) for p in files], ignore_index=True)
        tc, lc = _detect_cols(full, c["text_col"], c["label_col"])
        tr, rest = train_test_split(full, test_size=0.2, random_state=cfg["project"]["seed"], stratify=full[lc])
        dv, te = train_test_split(rest, test_size=0.5, random_state=cfg["project"]["seed"], stratify=rest[lc])
        splits = {"train": tr, "dev": dv, "test": te}
    tc, lc = _detect_cols(splits["train"], c["text_col"], c["label_col"])
    pos = {str(v) for v in c["spam_positive_values"]}
    drop = {str(v) for v in c.get("drop_values") or []}
    if drop:   # bỏ hẳn các loại spam không áp dụng cho Google Maps (khỏi cả train lẫn test)
        splits = {k: v[~v[lc].astype(str).isin(drop)] for k, v in splits.items()}
    teen = load_teencode(cfg["preprocess"]["teencode_file"])
    out = {}
    for k, v in splits.items():
        out[k] = pd.DataFrame({"text": [normalize(str(t), teen) for t in v[tc].fillna("")],
                               "y": v[lc].astype(str).isin(pos).astype(int)})
    get_logger("m1b").info("Benchmark: cột văn bản='%s', nhãn='%s', kích thước %s", tc, lc,
                           {k: len(v) for k, v in out.items()})
    return out


def build_model(C: float, seed: int) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                 max_features=300_000)),
    ])
    return Pipeline([("tfidf", feats),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))])


def _metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "precision_spam": precision_score(y, pred, zero_division=0),
            "recall_spam": recall_score(y, pred, zero_division=0), "f1_spam": f1_score(y, pred, zero_division=0)}


def train(P: Paths, cfg: dict) -> Pipeline | None:
    log = get_logger("m1b", P.logs)
    data = load_benchmark(P, cfg)
    if not data:
        log.warning("Không có benchmark trong %s → M1b dùng chế độ LUẬT.", P.benchmark / cfg["m1b"]["benchmark_dir"])
        return None
    seed = cfg["project"]["seed"]
    best, best_f1, rows = None, -1, []
    for C in cfg["m1b"]["C_grid"]:
        m = build_model(C, seed).fit(data["train"]["text"], data["train"]["y"])
        f1 = f1_score(data["dev"]["y"], m.predict(data["dev"]["text"]), average="macro")
        rows.append({"split": "dev", "C": C, "macro_f1": f1})
        if f1 > best_f1:
            best, best_f1 = (C, m), f1
    C, model = best
    # Huấn luyện lại trên train+dev với C tốt nhất, báo cáo trên test (chưa từng dùng để chọn)
    trdv = pd.concat([data["train"], data["dev"]])
    model = build_model(C, seed).fit(trdv["text"], trdv["y"])
    pred = model.predict(data["test"]["text"])
    res = {"split": "test", "C": C, **_metrics(data["test"]["y"], pred)}
    tn, fp, fn, tp = confusion_matrix(data["test"]["y"], pred, labels=[0, 1]).ravel()
    res.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    write_table(pd.DataFrame(rows + [res]), P.tables / "m1b_benchmark.csv")
    joblib.dump(model, P.models / cfg["m1b"]["model_file"])
    log.info("M1b benchmark (test): macro-F1=%.3f, P/R spam=%.3f/%.3f, C=%s",
             res["macro_f1"], res["precision_spam"], res["recall_spam"], C)
    return model


# ---------------------------------------------------------------------------
# Điểm luật (dùng khi chưa có benchmark, và luôn xuất làm lý do giải thích)
# ---------------------------------------------------------------------------
def rule_score(df: pd.DataFrame, ad_keywords: list[str]) -> tuple[np.ndarray, list[str]]:
    kw = [k.lower() for k in ad_keywords]
    text = df["text_norm"].fillna("")
    n_kw = np.array([sum(k in t for k in kw) for t in text])
    parts = {
        "có link": (df["n_urls"] > 0).to_numpy() * 0.5,
        "có SĐT": (df["n_phones"] > 0).to_numpy() * 0.5,
        "từ khóa quảng cáo": np.minimum(n_kw, 3) / 3 * 0.4,
        "nhiều emoji": (df["n_emojis"] >= 5).to_numpy() * 0.15,
        "viết hoa nhiều": ((df["upper_ratio"] > 0.6) & (df["n_chars"] > 15)).to_numpy() * 0.15,
    }
    score = np.clip(sum(parts.values()), 0, 1)
    reasons = [", ".join(k for k, v in parts.items() if v[i] > 0) for i in range(len(df))]
    return score, reasons


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("m1b", P.logs)
    mode = cfg["m1b"]["mode"]
    model_path = P.models / cfg["m1b"]["model_file"]
    model = None
    if mode in ("auto", "model"):
        model = joblib.load(model_path) if model_path.exists() else train(P, cfg)
        if model is None and mode == "model":
            raise RuntimeError("m1b.mode=model nhưng không có benchmark/model")
    clean = P.interim / "reviews_clean.parquet"
    if not clean.exists():      # cho phép train + kiểm định benchmark trước khi có dữ liệu Google Maps
        log.warning("Chưa có %s → chỉ huấn luyện/kiểm định benchmark, chưa chấm review Google Maps "
                    "(chạy --stages all sau khi thu thập xong).", clean.name)
        return model_path if model is not None else clean
    df = pd.read_parquet(clean)

    elig = df["has_text"] & (df["lang"] == "vi")
    rs, reasons = rule_score(df, cfg["m1b"]["ad_keywords"])
    feat = pd.DataFrame({"review_id": df["review_id"], "content_rule": rs,
                         "reason_content": [f"nội dung: {r}" if r else "" for r in reasons]})
    if model is not None:
        prob = np.full(len(df), np.nan)
        prob[elig.to_numpy()] = model.predict_proba(df.loc[elig, "text_norm"])[:, 1]
        feat["content_model"] = prob
        feat["s_content_raw"] = prob
        feat["content_mode"] = "model"
    else:
        feat["s_content_raw"] = np.where(df["has_text"], rs, np.nan)
        feat["content_mode"] = "rules"
    feat.loc[~df["has_text"].to_numpy(), "s_content_raw"] = np.nan

    out = P.processed / "features_m1b.parquet"
    feat.to_parquet(out, index=False)
    # 30 dự đoán ngẫu nhiên để kiểm tra tay hiện tượng lệch miền
    spot = df.loc[elig, ["review_id", "rating", "text"]].merge(feat[["review_id", "s_content_raw"]], on="review_id")
    spot = spot.sample(min(30, len(spot)), random_state=cfg["project"]["seed"]).sort_values("s_content_raw", ascending=False)
    write_table(spot.assign(nhan_xet=""), P.tables / "m1b_gmaps_spotcheck.csv")
    log.info("M1b (%s): %d review được chấm, trung bình=%.3f", feat["content_mode"].iloc[0],
             feat["s_content_raw"].notna().sum(), np.nanmean(feat["s_content_raw"]))
    return out
