"""Độ đồng thuận giữa người gán nhãn (A, B) và Claude (C) + hợp nhất nhãn (D3).

- A, B gán toàn bộ Nhóm B + human_a_n mục Nhóm A; C (Claude trên claude.ai, xem llm_label.py) gán toàn bộ
  Nhóm A — phần chung với người dùng để kiểm chứng; nhãn NGƯỜI là nguồn chính của mọi con số kết quả.
- Dùng nhãn C cho phần người chưa gán CHỈ KHI κ(người hợp nhất–C) ≥ kappa_min và ≥ κ(A–B) − llm_max_kappa_gap
  (trên ≥ 30 mục); gold_final có cột label_source = human | llm để báo cáo tách riêng.

- Cohen's Kappa cho kết luận 3 lớp (Genuine / Uncertain / Spam) và cho từng cờ f1..f6.
- Tính riêng cho PILOT (pilot_n mục người gán đầu tiên theo item_id — thứ tự đã xáo trộn nên là mẫu ngẫu nhiên).
- Kiểm tra nhất quán với luật rubric: 0–1 cờ → Genuine, 2 cờ → Uncertain,
  ≥3 cờ hoặc có cờ 2/3 → Spam (người gán vẫn được quyết khác, nhưng sẽ bị liệt kê để xem lại).
- Hợp nhất: 2 người đồng ý → lấy luôn; bất đồng → lấy từ adjudication.csv (item_id,label) nếu có,
  ngược lại ghi vào disagreements.csv và loại khỏi gold_final cho tới khi được hòa giải.
Đầu ra: labels/gold_final.csv, reports/tables/agreement.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.evaluation.sample_gold import FLAG_COLS
from src.utils.common import get_logger, write_table
from src.utils.config import Paths

LABELS = {"genuine": "Genuine", "g": "Genuine", "that": "Genuine", "uncertain": "Uncertain", "u": "Uncertain",
          "khong_chac": "Uncertain", "spam": "Spam", "s": "Spam", "gia": "Spam"}


def _norm_label(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return LABELS.get(str(x).strip().lower().replace(" ", "_"), None)


def rule_label(flags: pd.Series) -> str:
    n = int(flags.sum())
    if n >= 3 or flags.get("f2_quang_cao", 0) == 1 or flags.get("f3_trung_noi_dung", 0) == 1:
        return "Spam"
    return "Uncertain" if n == 2 else "Genuine"


def _load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, encoding="utf-8-sig")
    d["label"] = d["label"].map(_norm_label)
    for f in FLAG_COLS:
        d[f] = pd.to_numeric(d[f], errors="coerce").fillna(0).astype(int)
    return d


def kappa(a, b) -> float:
    if len(set(a) | set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def _kappa_rows(scope: str, pair: str, x: pd.DataFrame, a: str, b: str) -> list[dict]:
    rows = [{"phạm vi": scope, "cặp": pair, "mục": "kết luận (3 lớp)", "n": len(x),
             "kappa": kappa(x["label" + a], x["label" + b]),
             "đồng ý %": 100 * (x["label" + a] == x["label" + b]).mean()}]
    for f in FLAG_COLS:
        rows.append({"phạm vi": scope, "cặp": pair, "mục": f, "n": len(x), "kappa": kappa(x[f + a], x[f + b]),
                     "đồng ý %": 100 * (x[f + a] == x[f + b]).mean()})
    return rows


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("agreement", P.logs)
    e = cfg["evaluation"]
    A, B = _load(P.labels / "annotator_A.csv"), _load(P.labels / "annotator_B.csv")
    m = A.merge(B, on="item_id", suffixes=("_a", "_b"))
    done = m[m["label_a"].notna() & m["label_b"].notna()].copy()
    if done.empty:
        raise RuntimeError("Chưa có mục nào được CẢ HAI người gán nhãn (cột 'label').")
    c_path = P.labels / "annotator_C.csv"
    C = _load(c_path).add_suffix("_c").rename(columns={"item_id_c": "item_id"}) if c_path.exists() else None

    # 1) Người – người (trần của độ đồng thuận)
    rows = []
    pilot_ids = sorted(A["item_id"])[:e["pilot_n"]]
    for name, sub in (("pilot", done[done["item_id"].isin(pilot_ids)]), ("người gán", done)):
        if len(sub):
            rows += _kappa_rows(name, "A–B", sub, "_a", "_b")

    # 2) Nhất quán với luật rubric (A, B, C)
    incons = []
    whos = [("a", done)] + [("b", done)] + ([("c", C[C["label_c"].notna()])] if C is not None else [])
    for who, d in whos:
        flags = d[[f + "_" + who for f in FLAG_COLS]].rename(columns=lambda c: c[:-2])
        rl = flags.apply(rule_label, axis=1)
        bad = d[rl != d["label_" + who]]
        incons += [{"item_id": i, "annotator": who.upper(), "label": l, "rule_label": r}
                   for i, l, r in zip(bad["item_id"], bad["label_" + who], rl[bad.index])]
    write_table(pd.DataFrame(incons), P.tables / "label_vs_rubric_inconsistencies.csv")

    # 3) Hợp nhất nhãn NGƯỜI (đồng ý → lấy; bất đồng → adjudication.csv)
    done["label"] = np.where(done["label_a"] == done["label_b"], done["label_a"], None)
    adj_p = P.labels / "adjudication.csv"
    if adj_p.exists():
        adj = pd.read_csv(adj_p, encoding="utf-8-sig")
        adj["label"] = adj["label"].map(_norm_label)
        done["label"] = done["label"].fillna(done["item_id"].map(adj.set_index("item_id")["label"]))
    dis = done[done["label"].isna()]
    write_table(dis[["item_id", "label_a", "label_b"]], P.labels / "disagreements.csv")
    for f in FLAG_COLS:
        done[f] = ((done[f + "_a"] + done[f + "_b"]) >= 1).astype(int)   # cờ: ít nhất 1 người đánh
    human = done[done["label"].notna()][["item_id", "label"] + FLAG_COLS].assign(label_source="human")

    # 4) Người – Claude: quyết định có dùng nhãn Claude cho phần còn lại không
    use_llm, verdict = False, "chưa có annotator_C.csv → chỉ dùng nhãn người"
    llm = pd.DataFrame()
    if C is not None:
        hc = done.merge(C, on="item_id")
        rows += _kappa_rows("người gán", "A–C", hc, "_a", "_c") + _kappa_rows("người gán", "B–C", hc, "_b", "_c")
        hf = human.merge(C, on="item_id").rename(columns={"label": "label_h", **{f: f + "_h" for f in FLAG_COLS}})
        rows += _kappa_rows("người gán", "người (hợp nhất)–C", hf, "_h", "_c")
        k_ab = kappa(done["label_a"], done["label_b"])
        k_hc = kappa(hf["label_h"], hf["label_c"])
        checks = {
            f"κ(A–B) ≥ {e['kappa_min']}": k_ab >= e["kappa_min"],
            "đã hòa giải hết bất đồng của người": len(dis) == 0,
            "≥ 30 mục chung": len(hf) >= 30,
            f"κ(người–Claude) ≥ {e['kappa_min']}": k_hc >= e["kappa_min"],
            f"κ(người–Claude) ≥ κ(A–B) − {e['llm_max_kappa_gap']}": k_hc >= k_ab - e["llm_max_kappa_gap"],
        }
        use_llm = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        verdict = (f"κ(A–B)={k_ab:.2f}, κ(người–Claude)={k_hc:.2f} trên {len(hf)} mục → "
                   + ("DÙNG nhãn Claude cho các mục người chưa gán" if use_llm else
                      "KHÔNG dùng nhãn Claude — chưa đạt: " + "; ".join(failed)))
        cm = pd.crosstab(hf["label_h"].rename("người"), hf["label_c"].rename("Claude"))
        write_table(cm.reset_index(), P.tables / "human_vs_claude_confusion.csv")
        if use_llm:
            rest = C[~C["item_id"].isin(done["item_id"]) & C["label_c"].notna()]
            llm = rest.rename(columns={"label_c": "label", **{f + "_c": f for f in FLAG_COLS}})[
                ["item_id", "label"] + FLAG_COLS].assign(label_source="llm")
    ag = pd.DataFrame(rows)
    write_table(ag.round(3), P.tables / "agreement.csv")
    write_table(pd.DataFrame([{"kết luận": verdict}]), P.tables / "llm_validation.csv")
    log.info(verdict)
    k_pilot = ag[(ag["phạm vi"] == "pilot") & (ag["mục"] == "kết luận (3 lớp)")]["kappa"]
    if len(k_pilot) and k_pilot.iloc[0] < e["kappa_min"]:
        log.warning("Kappa pilot = %.2f < %.2f → nên họp chỉnh rubric trước khi gán phần còn lại.",
                    k_pilot.iloc[0], e["kappa_min"])

    key = pd.read_csv(P.labels / "gold_key.csv", encoding="utf-8-sig")
    final = pd.concat([human, llm], ignore_index=True).merge(key, on="item_id")
    out = P.labels / "gold_final.csv"
    write_table(final, out)
    log.info("gold_final: %d mục (người=%d, Claude=%d), còn %d bất đồng chưa hòa giải | %s",
             len(final), len(human), len(llm), len(dis), final["label"].value_counts().to_dict())
    return out
