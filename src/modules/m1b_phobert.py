"""Module 1b — phiên bản PhoBERT (TÙY CHỌN, chạy trên Colab/máy có GPU).

!!! Script này CHƯA được chạy thử trong môi trường phát triển (không tải được model từ
HuggingFace). Hãy chạy thử trên Colab trước khi dùng kết quả cho paper. !!!

Cách dùng (sau khi đã chạy các bước parse, clean):
    pip install transformers torch
    python run_pipeline.py --stages m1b_phobert,fusion
Kết quả ghi đè processed/features_m1b.parquet với content_mode = "phobert"
và bảng reports/tables/m1b_benchmark_phobert.csv.

PhoBERT yêu cầu văn bản ĐÃ TÁCH TỪ (dùng cột text_seg / pyvi) và tối đa 256 token.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.modules.m1b_content import _metrics, load_benchmark, rule_score
from src.preprocess.normalize_vi import segment
from src.utils.common import get_logger, write_table
from src.utils.config import Paths

MODEL = "vinai/phobert-base"


def run(cfg: dict, epochs: int = 3, batch_size: int = 32, lr: float = 2e-5, max_len: int = 160) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    P = Paths(cfg).ensure()
    log = get_logger("m1b_phobert", P.logs)
    torch.manual_seed(cfg["project"]["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_benchmark(P, cfg)
    if not data:
        raise RuntimeError("Cần benchmark ViSpamReviews trong data/benchmark/ để huấn luyện PhoBERT.")
    for k in data:                                           # PhoBERT cần văn bản đã tách từ
        data[k]["text"] = [segment(t) for t in data[k]["text"]]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)

    def loader(texts, labels=None, shuffle=False):
        enc = tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        items = [enc["input_ids"], enc["attention_mask"]]
        if labels is not None:
            items.append(torch.tensor(np.asarray(labels), dtype=torch.long))
        ds = torch.utils.data.TensorDataset(*items)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def predict(dl):
        model.eval()
        out = []
        with torch.no_grad():
            for b in dl:
                logits = model(input_ids=b[0].to(dev), attention_mask=b[1].to(dev)).logits
                out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    tr = loader(data["train"]["text"], data["train"]["y"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(tr) * epochs), len(tr) * epochs)
    w = torch.tensor([1.0, (1 - data["train"]["y"].mean()) / max(data["train"]["y"].mean(), 1e-6)], device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    best_f1, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for b in tr:
            opt.zero_grad()
            logits = model(input_ids=b[0].to(dev), attention_mask=b[1].to(dev)).logits
            loss = lossf(logits, b[2].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        pdev = predict(loader(data["dev"]["text"]))
        f1 = _metrics(data["dev"]["y"], (pdev >= 0.5).astype(int))["macro_f1"]
        log.info("epoch %d: dev macro-F1=%.4f", ep + 1, f1)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ptest = predict(loader(data["test"]["text"]))
    res = {"model": MODEL, "best_dev_macro_f1": best_f1, **_metrics(data["test"]["y"], (ptest >= 0.5).astype(int))}
    write_table(pd.DataFrame([res]), P.tables / "m1b_benchmark_phobert.csv")

    df = pd.read_parquet(P.interim / "reviews_clean.parquet")
    elig = (df["has_text"] & (df["lang"] == "vi")).to_numpy()
    prob = np.full(len(df), np.nan)
    prob[elig] = predict(loader(df.loc[elig, "text_seg"]))
    rs, reasons = rule_score(df, cfg["m1b"]["ad_keywords"])
    feat = pd.DataFrame({"review_id": df["review_id"], "content_rule": rs, "content_model": prob,
                         "s_content_raw": prob, "content_mode": "phobert",
                         "reason_content": [f"nội dung: {r}" if r else "" for r in reasons]})
    out = P.processed / "features_m1b.parquet"
    feat.to_parquet(out, index=False)
    log.info("PhoBERT test macro-F1=%.4f; đã chấm %d review Google Maps", res["macro_f1"], int(elig.sum()))
    return out
