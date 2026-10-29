"""Đóng gói công cụ gán nhãn thành MỘT file HTML (mở bằng trình duyệt, không cần cài gì).

File chứa toàn bộ review của mọi quán (nén gzip + base64, giải nén bằng DecompressionStream của
trình duyệt) cùng các bằng chứng tính sẵn:
  - nội dung trùng (cụm gần trùng của M1a, kể cả ở quán khác)
  - ảnh trùng (features_img.parquet của img_hash; thiếu thì bỏ qua)
  - các review khác của cùng tài khoản (reviewer_hash — không có tên / id thật)
Ảnh hiển thị trực tiếp từ máy chủ ảnh của Google nên khi gán nhãn cần có mạng.
KHÔNG đưa điểm nghi vấn / zone của hệ thống vào file để nhãn không bị ảnh hưởng bởi hệ thống.

Đầu ra: <paths.labels>/<labeltool.output> (mặc định data/labels/labeltool/gan_nhan_review.html)
"""
from __future__ import annotations

import base64
import gzip
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.sample_gold import FLAG_COLS
from src.utils.common import get_logger, read_jsonl
from src.utils.config import Paths

TEMPLATE = Path(__file__).with_name("template.html")
HOST = "https://lh3.googleusercontent.com/"
FLAG_KEYS = FLAG_COLS + ["f7_trung_anh"]
FLAGS = [   # (cột, nhãn ngắn, mô tả) — theo docs/huong_dan_gan_nhan.md, thêm f7 cho ảnh
    (FLAG_KEYS[0], "Chung chung", "Có thể dán y nguyên cho hầu hết quán cùng loại; không có chi tiết cụ thể nào "
                                  "(tên món, giá, sự việc, thời điểm, người)."),
    (FLAG_KEYS[1], "Quảng cáo", "Link, SĐT, Zalo, mã giảm giá, mời liên hệ; quảng bá nơi khác; quán tự viết; "
                                "nói rõ được tặng quà / giảm giá để đổi lấy review."),
    (FLAG_KEYS[2], "Trùng nội dung", "Có review khác (tài khoản khác hoặc quán khác) gần như y hệt, dài ≥ 8 từ."),
    (FLAG_KEYS[3], "TK mỏng, sao cực đoan", "≤ 3 review, chấm 1★/5★, không phải Local Guide; hoặc mọi review khác "
                                            "của tài khoản đều cùng 1★/5★ (≤ 10 review, gần như không có ảnh)."),
    (FLAG_KEYS[4], "Nhiều quán cùng lúc", "Cùng tài khoản review ≥ 3 quán trong 1 ngày, hoặc 2 quán khác tỉnh/thành "
                                          "trong 1 ngày."),
    (FLAG_KEYS[5], "Đợt dồn review", "Số review của quán trong ±3 ngày ≥ 5 và ≥ 3 lần mức bình thường; review cùng "
                                     "chiều với đợt."),
    (FLAG_KEYS[6], "Trùng ảnh", "Ảnh đính kèm giống hệt ảnh của review khác (tài khoản khác hoặc quán khác): "
                                "ảnh lấy lại / ảnh mạng."),
]
STRONG = [FLAG_KEYS[1], FLAG_KEYS[2], FLAG_KEYS[6]]      # có 1 cờ này → gợi ý Spam


OWNER_HEAD = re.compile(r"^\s*(?:Thông tin phản hồi từ chủ sở hữu|Response from the owner)\s*(.*)$", re.I)
OWNER_JUNK = re.compile(r"^\s*(?:Bản dịch của Google|Translated by Google)\b.*$", re.I)


def _clean_owner(text: str) -> str:
    """Bỏ phần chữ giao diện: dòng tiêu đề (giữ thời điểm), "Bản dịch của Google ・ Xem bản gốc", "… Xem thêm"."""
    lines, when = [], ""
    for i, ln in enumerate(text.splitlines()):
        m = OWNER_HEAD.match(ln) if i == 0 else None
        if m:
            when = m.group(1).strip()
        elif not OWNER_JUNK.match(ln):
            lines.append(re.sub(r"\s*… ?Xem thêm\s*$", " …", ln))
    body = "\n".join(lines).strip()
    return f"{when}\n{body}" if body else ""


def _owner_responses(P: Paths) -> dict[str, str]:
    """review_id → "<thời điểm>\\n<nội dung>" (dòng đầu có thể rỗng)."""
    out = {}
    for p in sorted((P.raw / "reviews").glob("*.jsonl")):
        for r in read_jsonl(p):
            if r.get("owner_response") and r.get("review_id"):
                out[r["review_id"]] = _clean_owner(r["owner_response"])
    return out


def _nullable(s: pd.Series, cast=int) -> list:
    return [None if pd.isna(x) else cast(x) for x in s]


def build_payload(cfg: dict, P: Paths, log) -> dict:
    df = pd.read_parquet(P.interim / "reviews_clean.parquet")
    m1a = pd.read_parquet(P.processed / "features_m1a.parquet", columns=["review_id", "dup_cluster_id"])
    df = df.merge(m1a, on="review_id", how="left")
    places = pd.read_parquet(P.interim / "places.parquet")
    places = places[places["place_key"].isin(df["place_key"])].drop_duplicates("place_key")
    places = places.sort_values(["query", "name"]).reset_index(drop=True)
    pidx = {k: i for i, k in enumerate(places["place_key"])}
    df["p"] = df["place_key"].map(pidx)
    df = df.sort_values(["p", "review_day"], ascending=[True, False]).reset_index(drop=True)

    media_path = P.processed / "features_img.parquet"
    if media_path.exists():
        media = pd.read_parquet(media_path)
        vids = pd.read_parquet(P.interim / "review_media.parquet", columns=["review_id", "idx", "video"])
        media = media.merge(vids, on=["review_id", "idx"], how="left")
    elif (P.interim / "review_media.parquet").exists():
        log.warning("Chưa có features_img.parquet (bước img_hash) → không có thông tin ảnh trùng")
        media = pd.read_parquet(P.interim / "review_media.parquet").assign(img_group=-1)
    else:
        log.warning("Chưa có review_media.parquet (bước media) → không hiển thị ảnh")
        media = pd.DataFrame(columns=["review_id", "idx", "photo", "video", "img_group"])
    strip = lambda u: u[len(HOST):] if isinstance(u, str) and u.startswith(HOST) else (u if isinstance(u, str) else 0)
    media = media.sort_values(["review_id", "idx"])
    mlist = {rid: [[strip(r.photo), strip(r.video), int(r.img_group)] for r in g.itertuples()]
             for rid, g in media.groupby("review_id", sort=False)}

    owner = _owner_responses(P)
    users = df["reviewer_hash"].fillna("").astype(str)
    uniq = sorted(set(users) - {""})
    uidx = {h: i for i, h in enumerate(uniq)}
    epoch = pd.Timestamp("1970-01-01")
    day = ((pd.to_datetime(df["review_day"]) - epoch).dt.days)

    R = {
        "id": df["review_id"].tolist(),
        "p": df["p"].astype(int).tolist(),
        "u": [uidx.get(h, -1) for h in users],
        "r": _nullable(df["rating"]),
        "t": df["text"].fillna("").tolist(),
        "tr": df["text_truncated"].astype(int).tolist(),
        "dt": df["date_text"].fillna("").tolist(),
        "dy": _nullable(day),
        "ed": df["is_edited"].astype(int).tolist(),
        "lg": _nullable(df["reviewer_local_guide"]),
        "nr": _nullable(df["reviewer_n_reviews"]),
        "na": _nullable(df["reviewer_n_photos"]),
        "ow": [owner.get(r, "") for r in df["review_id"]],
        "m": [mlist.get(r, 0) for r in df["review_id"]],
        "dc": _nullable(df["dup_cluster_id"].where(df["dup_cluster_id"] >= 0)),
    }
    P_ = {
        "k": places["place_key"].tolist(),
        "n": places["name"].fillna(places["place_key"]).tolist(),
        "u": places["url"].fillna("").tolist(),
        "r": _nullable(places["rating"], float),
        "c": _nullable(places["user_rating_count"]),
        "q": places["query"].fillna("").tolist(),
    }
    return {"places": P_, "reviews": R, "users": [h[:8] for h in uniq]}


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("labeltool", P.logs)
    lt = cfg.get("labeltool", {})
    payload = build_payload(cfg, P, log)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "host": HOST,
        "flags": [{"key": k, "short": s, "desc": d} for k, s, d in FLAGS],
        "strong": STRONG,
        "burst": {"min_count": 5, "ratio": 3.0, **lt.get("burst", {})},
        "has_img_hash": (P.processed / "features_img.parquet").exists(),
    }
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DATA__", packed))
    out = P.labels / lt.get("output", "labeltool/gan_nhan_review.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    R = payload["reviews"]
    n_media = sum(1 for m in R["m"] if m)
    log.info("Công cụ gán nhãn: %d quán, %d review (%d có ảnh/video) | JSON %.1f MB → file %.1f MB: %s",
             len(payload["places"]["k"]), len(R["id"]), n_media, len(raw) / 1e6, out.stat().st_size / 1e6, out)
    return out
