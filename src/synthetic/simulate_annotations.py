"""Giả lập 2 người gán nhãn cho gold set DEMO (dùng nhãn giả lập làm "sự thật" + nhiễu).

!!! CHỈ DÙNG CHO DEMO — với dữ liệu thật, hai người gán nhãn điền tay annotator_A/B.csv !!!
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.sample_gold import FLAG_COLS
from src.utils.common import get_logger, write_table
from src.utils.config import Paths

TYPE_FLAGS = {"copy_paste": [0, 2, 3], "burst_5star": [0, 3, 5], "bomb_1star": [3, 5, 0],
              "ad_spam": [1, 3, 0], "account_farm": [3, 4, 5]}


def _annotate(truth: pd.DataFrame, rng: np.random.Generator, acc: float) -> pd.DataFrame:
    rows = []
    for t in truth.itertuples():
        flags = np.zeros(6, dtype=int)
        if t.is_spam:
            for j in TYPE_FLAGS[t.spam_type]:
                flags[j] = int(rng.random() < 0.85)
            label = "Spam" if rng.random() < acc else rng.choice(["Uncertain", "Genuine"], p=[0.7, 0.3])
        else:
            if rng.random() < 0.15:
                flags[rng.integers(6)] = 1
            label = "Genuine" if rng.random() < acc else rng.choice(["Uncertain", "Spam"], p=[0.7, 0.3])
        rows.append({"item_id": t.item_id, **dict(zip(FLAG_COLS, flags)), "label": label, "ghi_chu": "DEMO"})
    return pd.DataFrame(rows)


def run(cfg: dict):
    P = Paths(cfg).ensure()
    log = get_logger("demo_annot", P.logs)
    key = pd.read_csv(P.labels / "gold_key.csv", encoding="utf-8-sig")
    truth = pd.read_parquet(P.root / "_truth" / "truth.parquet")
    t = key.merge(truth, on="review_id", how="left")
    t["is_spam"] = t["is_spam"].fillna(False).astype(bool)
    rng = np.random.default_rng(cfg["project"]["seed"])
    th = t[t["annotator"] == "human"] if "annotator" in t else t     # người: Nhóm B + human_a_n mục Nhóm A
    A, B = _annotate(th, rng, 0.92), _annotate(th, rng, 0.88)
    write_table(A, P.labels / "annotator_A.csv")
    write_table(B, P.labels / "annotator_B.csv")
    if "for_claude" in t:                                            # Claude giả lập: toàn bộ Nhóm A
        write_table(_annotate(t[t["for_claude"]], rng, 0.90), P.labels / "annotator_C.csv")
    # "Họp hòa giải": với mục bất đồng, chọn theo sự thật (mô phỏng thảo luận)
    dis = A.merge(B, on="item_id", suffixes=("_a", "_b"))
    dis = dis[dis["label_a"] != dis["label_b"]][["item_id"]].merge(t[["item_id", "is_spam"]], on="item_id")
    dis["label"] = np.where(dis["is_spam"], "Spam", "Genuine")
    write_table(dis[["item_id", "label"]], P.labels / "adjudication.csv")
    log.info("DEMO: đã giả lập nhãn cho %d mục (%d bất đồng được hòa giải)", len(t), len(dis))
