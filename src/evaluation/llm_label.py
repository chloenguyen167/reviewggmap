"""Gán nhãn gold set bằng Claude trên claude.ai (không dùng API) → "người gán C".

Bước 1 — `--stages llm_export` tạo thư mục labels/claude_ai/:
  - PROMPT.md       : instruction (gồm nguyên văn docs/huong_dan_gan_nhan.md + định dạng đầu ra) — dán vào
                      Project instructions trên claude.ai (hoặc đầu mỗi cuộc chat)
  - batch_01.csv ...: mỗi file ~batch_size mục; giống gold_sample_blind.csv + cột `review_giong_nhat`
                      (K review giống nhất trong corpus, tính sẵn bằng TF-IDF — thay cho Ctrl+F)
  - ket_qua/        : nơi lưu CSV Claude trả về (batch_01_result.csv ...)
Bước 2 — mỗi batch: 1 cuộc chat mới, upload batch_XX.csv, gửi "Gán nhãn file này", tải CSV kết quả về ket_qua/.
Bước 3 — `--stages llm_import`: kiểm tra định dạng/đủ mục, gộp → labels/annotator_C.csv.
Claude chỉ thấy đúng dữ liệu người gán thấy (không có điểm module, không biết nhóm A/B, không thấy nhãn người).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.evaluation.sample_gold import FLAG_COLS
from src.utils.common import get_logger, write_table
from src.utils.config import PROJECT_ROOT, Paths

GUIDE = PROJECT_ROOT / "docs" / "huong_dan_gan_nhan.md"
LABELS = ["Genuine", "Uncertain", "Spam"]
OUT_COLS = ["item_id"] + FLAG_COLS + ["label", "ghi_chu"]

PROMPT_TEMPLATE = """# Nhiệm vụ: gán nhãn review Google Maps nghi vấn (nghiên cứu khoa học)

Bạn là một người gán nhãn trong nghiên cứu phát hiện đánh giá giả mạo trên Google Maps tại Việt Nam.
Bạn gán nhãn ĐỘC LẬP theo đúng hướng dẫn bên dưới — giống hệt hướng dẫn hai người gán nhãn con người đang dùng.
Nhãn của bạn sẽ được so với nhãn của người, nên hãy bám sát định nghĩa và ngưỡng trong hướng dẫn, không dùng
tiêu chí riêng. Bạn không xem được trang Google Maps hay ảnh; chỉ dựa trên dữ liệu trong file.

## Dữ liệu trong mỗi file batch_XX.csv
Mỗi dòng là một mục, gồm các cột mô tả trong mục 2 của hướng dẫn, cộng thêm:
- `review_giong_nhat`: JSON danh sách tối đa {k} review giống nhất trong TOÀN BỘ dữ liệu (đã loại chính review này),
  mỗi phần tử có `do_giong` (0–1), `quan`, `sao`, `ngay`, `noi_dung`. Dùng cột này THAY CHO thao tác Ctrl+F trong
  corpus_lookup.csv khi xét cờ f3. Danh sách rỗng = không có review nào giống đáng kể (hoặc review không có chữ).

## Cách làm
1. Đọc toàn bộ file. Gán nhãn TỪNG dòng, không bỏ dòng nào, không gộp dòng.
2. Với mỗi dòng: xét 6 cờ trước, rồi kết luận theo mục 4 của hướng dẫn.
3. Nếu file dài, cứ làm hết; không tóm tắt, không hỏi lại giữa chừng.

## Định dạng trả về (bắt buộc)
Trả về MỘT file CSV (UTF-8, dấu phẩy, có dòng tiêu đề) tên `batch_XX_result.csv` (XX = số của file đầu vào), đúng các cột:
`{cols}`
- `item_id`: giữ nguyên từ file đầu vào; đủ tất cả item_id, mỗi item_id đúng 1 dòng
- 6 cờ: chỉ 0 hoặc 1
- `label`: chỉ một trong `Genuine`, `Uncertain`, `Spam`
- `ghi_chu`: 1 câu tiếng Việt nêu bằng chứng chính (bắt buộc nêu lý do nếu label khác luật tham khảo); đặt trong
  dấu ngoặc kép nếu có dấu phẩy
Nếu không tạo được file, in toàn bộ CSV trong một khối ```csv``` duy nhất.

---

<huong_dan_gan_nhan>
{guide}
</huong_dan_gan_nhan>
"""


def similar_reviews(blind: pd.DataFrame, own_ids: dict[int, str], corpus: pd.DataFrame,
                    k: int) -> dict[int, list[dict]]:
    """K review giống nhất trong corpus (trừ chính nó) cho mỗi mục có chữ."""
    corpus = corpus[corpus["text"].fillna("").str.strip() != ""].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(corpus["text"].str.lower())
    out: dict[int, list[dict]] = {}
    for it, txt in zip(blind["item_id"], blind["text"].fillna("")):
        out[it] = []
        if not txt.strip():
            continue
        sims = linear_kernel(vec.transform([txt.lower()]), X).ravel()
        for j in np.argsort(-sims):
            if len(out[it]) >= k or sims[j] < 0.2:
                break
            if corpus.at[j, "id_tra_cuu"] == own_ids.get(it):
                continue
            out[it].append({"do_giong": round(float(sims[j]), 2), "quan": corpus.at[j, "place_name"],
                            "sao": corpus.at[j, "rating"], "ngay": str(corpus.at[j, "review_day"])[:10],
                            "noi_dung": corpus.at[j, "text"][:400]})
    return out


def run_export(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("llm_label", P.logs)
    c = cfg["evaluation"]["llm"]
    blind = pd.read_csv(P.labels / "gold_sample_blind.csv", encoding="utf-8-sig")
    corpus = pd.read_csv(P.labels / "corpus_lookup.csv", encoding="utf-8-sig")
    key = pd.read_csv(P.labels / "gold_key.csv", encoding="utf-8-sig")
    if "for_claude" in key:                      # Claude gán Nhóm A (gồm phần chung với người để kiểm chứng)
        blind = blind[blind["item_id"].isin(key.loc[key["for_claude"], "item_id"])]
    sims = similar_reviews(blind, dict(zip(key["item_id"], key["review_id"])), corpus, c["similar_k"])

    d = P.labels / "claude_ai"
    (d / "ket_qua").mkdir(parents=True, exist_ok=True)
    out = blind.drop(columns=["place_url"], errors="ignore").copy()
    out["review_giong_nhat"] = [json.dumps(sims[i], ensure_ascii=False, default=str) for i in out["item_id"]]
    n = 0
    for n, start in enumerate(range(0, len(out), c["batch_size"]), 1):
        write_table(out.iloc[start:start + c["batch_size"]], d / f"batch_{n:02d}.csv")
    prompt = PROMPT_TEMPLATE.format(k=c["similar_k"], cols=",".join(OUT_COLS),
                                    guide=GUIDE.read_text(encoding="utf-8"))
    (d / "PROMPT.md").write_text(prompt, encoding="utf-8")
    log.info("Đã xuất %d batch (%d mục) + PROMPT.md vào %s. Lưu CSV kết quả của Claude vào %s",
             n, len(out), d, d / "ket_qua")
    return d


def _read_result(p: Path) -> pd.DataFrame:
    text = p.read_text(encoding="utf-8-sig")
    if "```" in text:                                  # dán nguyên câu trả lời có khối ```csv
        text = text.split("```")[1].removeprefix("csv").lstrip("\n")
    from io import StringIO
    return pd.read_csv(StringIO(text))


def run_import(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("llm_label", P.logs)
    blind = pd.read_csv(P.labels / "gold_sample_blind.csv", encoding="utf-8-sig")
    key = pd.read_csv(P.labels / "gold_key.csv", encoding="utf-8-sig")
    if "for_claude" in key:
        blind = blind[blind["item_id"].isin(key.loc[key["for_claude"], "item_id"])]
    files = sorted((P.labels / "claude_ai" / "ket_qua").glob("*.csv")) + \
        sorted((P.labels / "claude_ai" / "ket_qua").glob("*.txt"))
    if not files:
        raise SystemExit("Chưa có file kết quả trong labels/claude_ai/ket_qua/")
    parts, problems = [], []
    for f in files:
        try:
            r = _read_result(f)
        except Exception as e:
            problems.append(f"{f.name}: không đọc được CSV ({e})")
            continue
        r.columns = [str(x).strip() for x in r.columns]
        miss = [x for x in OUT_COLS if x not in r.columns]
        if miss:
            problems.append(f"{f.name}: thiếu cột {miss}")
            continue
        parts.append(r[OUT_COLS].assign(_file=f.name))
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUT_COLS + ["_file"])
    res["item_id"] = pd.to_numeric(res["item_id"], errors="coerce")
    res["label"] = res["label"].astype(str).str.strip().str.capitalize()

    bad_label = res[~res["label"].isin(LABELS)]
    bad_flag = res[~res[FLAG_COLS].isin([0, 1, "0", "1"]).all(axis=1)]
    dup = res[res["item_id"].duplicated(keep=False)]
    missing = sorted(set(blind["item_id"]) - set(res["item_id"].dropna().astype(int)))
    extra = sorted(set(res["item_id"].dropna().astype(int)) - set(blind["item_id"]))
    problems += [f"label không hợp lệ ở item_id {bad_label['item_id'].tolist()}"] * (len(bad_label) > 0)
    problems += [f"cờ không phải 0/1 ở item_id {bad_flag['item_id'].tolist()}"] * (len(bad_flag) > 0)
    problems += [f"item_id lặp: {sorted(set(dup['item_id'].tolist()))}"] * (len(dup) > 0)
    problems += [f"item_id không có trong gold set: {extra}"] * (len(extra) > 0)
    problems += [f"THIẾU {len(missing)} mục: {missing[:30]}{' …' if len(missing) > 30 else ''}"] * (len(missing) > 0)
    for p_ in problems:
        log.warning(p_)

    ok = res[res["label"].isin(LABELS) & res.index.isin(res.index.difference(bad_flag.index))]
    ok = ok.drop_duplicates("item_id", keep="first").sort_values("item_id")
    ok[FLAG_COLS] = ok[FLAG_COLS].astype(int)
    p = P.labels / "annotator_C.csv"
    write_table(ok[OUT_COLS], p)
    log.info("annotator_C.csv: %d/%d mục hợp lệ %s", len(ok), len(blind), ok["label"].value_counts().to_dict())
    return p
