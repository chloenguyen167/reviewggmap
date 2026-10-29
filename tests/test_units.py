"""Unit test cho các hàm cốt lõi (không cần mạng, chạy vài giây)."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.collect.parse_raw import clean_pw_text, parse_rating_label, parse_reviewer_meta
from src.collect.place_url import contributor_id_from_url, parse_place_url
from src.collect.relative_date import parse_relative_date
from src.collect.rpc_timestamps import extract_from_body
from src.evaluation.agreement import rule_label
from src.evaluation.metrics import precision_at_k, ranking_metrics, wilson
from src.fusion.score import suspicion
from src.modules.m1a_duplicate import jaccard, shingles
from src.modules.m2_temporal import robust_z
from src.preprocess.normalize_vi import detect_lang, normalize, raw_features
from src.utils.common import one_sided, pct_rank

NOW = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)


# ---------- ngày tương đối ----------
@pytest.mark.parametrize("text,days_lo,days_hi,prec", [
    ("3 tháng trước", 4 * 30.44, 3 * 30.44, 30.44),
    ("một tuần trước", 14, 7, 7),
    ("2 ngày trước", 3, 2, 1),
    ("hôm qua", 2, 1, 1),
    ("a year ago", 2 * 365.25, 365.25, 365.25),
    ("5 năm trước trên Google", 6 * 365.25, 5 * 365.25, 365.25),
])
def test_relative_date_interval(text, days_lo, days_hi, prec):
    r = parse_relative_date(text, NOW)
    assert r.parsed and r.precision_days == pytest.approx(prec)
    assert (NOW - r.date_lo).total_seconds() / 86400 == pytest.approx(days_lo)
    assert (NOW - r.date_hi).total_seconds() / 86400 == pytest.approx(days_hi)
    assert r.date_lo <= r.date_est <= r.date_hi


def test_relative_date_edited_and_garbage():
    r = parse_relative_date("Đã chỉnh sửa 2 tuần trước", NOW)
    assert r.is_edited and r.parsed
    r = parse_relative_date("Thời gian chỉnh sửa: 3 năm trước", NOW)   # giao diện 2026
    assert r.is_edited and r.parsed and r.precision_days > 300
    assert not parse_relative_date("không rõ", NOW).parsed
    assert not parse_relative_date(None, NOW).parsed


# ---------- RPC ----------
def test_rpc_extracts_min_timestamp_and_ignores_place_id():
    rid1, rid2 = "ChZDSUhNMG9nS0VJQ0FnSUR1111111", "ChZDSUhNMG9nS0VJQ0FnSUR2222222"
    t1, t2 = 1_727_000_000_000_000, 1_700_000_000_000_000
    body = ")]}'\n" + json.dumps([None, [[rid1, [t1 + 5_000_000, [t1]]], [rid2, [[t2]], "ChIJabcdefghijklmnopqrstu"]],
                                   "ChIJabcdefghijklmnopqrstu"])
    out = extract_from_body(body)
    assert set(out) == {rid1, rid2}
    assert out[rid1] == datetime.fromtimestamp(t1 / 1e6, tz=timezone.utc)
    # với danh sách id đã biết: chỉ nhận id trong danh sách
    assert set(extract_from_body(body, known_ids={rid2})) == {rid2}


def test_rpc_bad_body():
    assert extract_from_body("not json") == {}


# ---------- parse nhỏ ----------
def test_rpc_batchexecute_format():
    """2026: review tải qua batchexecute — chunk "<độ dài>\n<json>", payload là chuỗi JSON lồng."""
    rid = "ChZDSUhNMG9nS0VJQ0FnSUMya2NHU2NnEAE"
    inner = json.dumps([None, "tok", [[[rid, ["0x0:0x1", None, 1650699733203088, 1650699946495817]]],
                                      [["ChZDSUhNMG9nS0VJQ0FnSUN4eHh4eHh4eEAE", ["0x0:0x1", None, 1700000000000000]]]]])
    chunk = json.dumps([["wrb.fr", "qv9Egd", inner, None, None, None, "generic"]])
    body = ")]}'\n\n" + f"{len(chunk)}\n{chunk}\n25\n" + '[["e",4,null,null,123]]\n'
    out = extract_from_body(body)
    assert out[rid] == datetime.fromtimestamp(1650699733.203088, tz=timezone.utc)
    assert len(out) == 2


def test_clean_pw_text():
    assert clean_pw_text("Quán ngon\nLoại đơn đặt món\nĂn tại chỗ\nĐồ ăn: 5") == ("Quán ngon", False)
    assert clean_pw_text("ngồi ở quầy bar … Xem thêm") == ("ngồi ở quầy bar", True)
    assert clean_pw_text("Matcha ngon. Xem thêm") == ("Matcha ngon.", False)
    # review chỉ chấm sao: selector cũ lấy nhầm phản hồi của chủ quán
    assert clean_pw_text("Xin lỗi khách iu", "Phản hồi từ chủ sở hữu 2 năm trước\nXin lỗi khách iu") == ("", False)
    assert clean_pw_text(None) == ("", False)


def test_rating_and_meta():
    assert parse_rating_label("5 sao") == 5
    assert parse_rating_label("Xếp hạng 4,0 trên 5") == 4
    assert parse_rating_label("2/5") == 2
    m = parse_reviewer_meta("Local Guide · 1.132 bài đánh giá · 1.024 ảnh")
    assert m == {"reviewer_local_guide": True, "reviewer_n_reviews": 1132, "reviewer_n_photos": 1024}
    assert parse_reviewer_meta("12 reviews")["reviewer_n_reviews"] == 12


def test_place_url():
    u = ("https://www.google.com/maps/place/X/@10.7,106.6,17z/data=!4m7!3m6!1s0x31752f4b:0x5a8b"
         "!8m2!3d10.7769!4d106.7009!16s%2Fg%2F11!19sChIJyby6M0svdTER")
    m = parse_place_url(u)
    assert m["place_id"] == "ChIJyby6M0svdTER" and m["data_id"] == "0x31752f4b:0x5a8b"
    assert m["lat"] == pytest.approx(10.7769) and m["lng"] == pytest.approx(106.7009)
    assert contributor_id_from_url("https://www.google.com/maps/contrib/1234567/reviews") == "1234567"


# ---------- văn bản ----------
def test_normalize_and_features():
    t = "Quán NGONNNN lắm!!! ko chê dc, gọi 0901 234 567 hoặc www.abc.vn 😍😍"
    f = raw_features(t)
    assert f["n_phones"] == 1 and f["n_urls"] == 1 and f["n_emojis"] == 2
    n = normalize(t, {"ko": "không", "dc": "được"})
    assert "không" in n and "được" in n and "ngon" in n and "0901" not in n and "abc" not in n
    assert normalize("giá 1000k", {}) == "giá 1000k"          # không làm hỏng số


def test_detect_lang():
    assert detect_lang("Đồ ăn ngon, nhân viên dễ thương") == "vi"
    assert detect_lang("quan ngon nhan vien rat nhiet tinh") == "vi"
    assert detect_lang("The food was great and the staff very friendly") == "en"
    assert detect_lang("") == "none"


# ---------- M1a ----------
def test_shingles_jaccard():
    a = shingles("a b c d e".split(), 3)
    b = shingles("a b c d x".split(), 3)
    assert a == {"a b c", "b c d", "c d e"}
    assert jaccard(a, b) == pytest.approx(2 / 4)
    assert jaccard(set(), a) == 0


# ---------- M2 ----------
def test_robust_z_sparse_series_has_floor():
    s = pd.Series([0.0] * 40 + [1.0] + [0.0] * 5 + [10.0])
    z, med = robust_z(s, 28)
    assert np.isfinite(z.iloc[40]) and z.iloc[40] == pytest.approx(1.0)   # 1 review lẻ: không bùng nổ
    assert z.iloc[-1] >= 3                                               # 10 review: bị gắn cờ


# ---------- tiện ích ----------
def test_pct_rank_zero_means_no_signal():
    s = pd.Series([0, 0, 0, 1, 2, np.nan])
    r = pct_rank(s)
    assert (r[:3] == 0).all() and r[4] == 1.0 and np.isnan(r[5])


def test_one_sided_clips_below_median():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    assert one_sided(X)["a"].tolist() == [3, 3, 3, 4, 100]


# ---------- Fusion ----------
def test_suspicion_methods_and_nan_reweighting():
    df = pd.DataFrame({"s_a": [0.9, 0.99, np.nan, 0.5], "s_b": [0.1, 0.5, np.nan, 0.5]})
    mean = suspicion(df, ["a", "b"], method="mean")
    assert mean[0] == pytest.approx(0.5) and np.isnan(mean[2])
    one = suspicion(pd.DataFrame({"s_a": [0.8], "s_b": [np.nan]}), ["a", "b"], method="mean")
    assert one[0] == pytest.approx(0.8)                     # NaN bị bỏ, không điền 0
    fis = suspicion(df, ["a", "b"], method="fisher")
    assert fis[1] > fis[0] > fis[3]                          # một tín hiệu cực mạnh được ưu tiên
    assert suspicion(df, ["a", "b"], method="max")[1] == pytest.approx(0.99)


# ---------- Đánh giá ----------
def test_precision_at_k_and_metrics():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, s, 1) == 1.0
    assert precision_at_k(y, s, 2) == 0.5
    assert np.isnan(precision_at_k(y, s, 10))
    m = ranking_metrics(y, s, [2])
    assert 0 <= m["PR-AUC"] <= 1 and m["ROC-AUC"] == pytest.approx(5 / 6)
    lo, hi = wilson(5, 50)
    assert lo < 0.1 < hi


def test_rule_label():
    f = pd.Series({"f1_chung_chung": 1, "f2_quang_cao": 0, "f3_trung_noi_dung": 0,
                   "f4_tai_khoan_it_review_cung_muc_sao": 0, "f5_nhieu_quan_xa_nhau_cung_luc": 0,
                   "f6_trong_cum_dang_bat_thuong": 0})
    assert rule_label(f) == "Genuine"
    f["f4_tai_khoan_it_review_cung_muc_sao"] = 1
    assert rule_label(f) == "Uncertain"
    f["f2_quang_cao"] = 1
    assert rule_label(f) == "Spam"


def test_read_claude_ai_result(tmp_path):
    from src.evaluation.llm_label import OUT_COLS, _read_result
    csv = ",".join(OUT_COLS) + '\n1,0,0,1,0,0,0,Spam,"trùng 3 quán, khác tài khoản"\n'
    p = tmp_path / "a.txt"
    p.write_text("Kết quả:\n```csv\n" + csv + "```\nXong.", encoding="utf-8")
    r = _read_result(p)
    assert list(r.columns) == OUT_COLS and r.loc[0, "label"] == "Spam" and r.loc[0, "f3_trung_noi_dung"] == 1
    p2 = tmp_path / "b.csv"
    p2.write_text(csv, encoding="utf-8")
    assert len(_read_result(p2)) == 1


# ---------- công cụ gán nhãn (src/labeltool) ----------
def test_media_extract_photos_and_video():
    from src.labeltool.media import _records
    H = "https://lh3.googleusercontent.com/"
    rec = lambda rid, items: [rid, None, [None, None, items]]
    photo = lambda pid, vid=None: ["m", [None] * 6 + [[H + "grass-cs/" + pid + "=k-no"]]
                                        + ([None] * 14 + [[None, None, [None] * 10 + [[[0, 1, 2, H + "gpms-cs-s/" + vid + "=m18"]]]]]
                                           if vid else [])]
    body = [[rec("ChAAAAAAAAAAAAAAAAAAAAAA", [photo("p1"), photo("p2", "v2")]),
             rec("ChBBBBBBBBBBBBBBBBBBBBBB", []),
             [H + "a/avatar", H + "local-reviews/badge=s120"]]]
    out = {}
    _records(body, {"ChAAAAAAAAAAAAAAAAAAAAAA", "ChBBBBBBBBBBBBBBBBBBBBBB"}, out)
    assert out["ChAAAAAAAAAAAAAAAAAAAAAA"] == [(H + "grass-cs/p1", None), (H + "grass-cs/p2", H + "gpms-cs-s/v2=m18")]
    assert "ChBBBBBBBBBBBBBBBBBBBBBB" not in out


def test_image_hash_groups_recompressed_copies():
    from io import BytesIO

    from PIL import Image

    from src.labeltool.image_hash import _signed, dhash, group_images
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).resize((256, 256)) for _ in range(3)]
    buf = BytesIO()
    imgs[0].resize((180, 180)).save(buf, "JPEG", quality=60)       # bản tải lại: đổi cỡ + nén
    hs = [dhash(im) for im in imgs + [Image.open(BytesIO(buf.getvalue()))]]
    h = np.array([_signed(a) for a, _ in hs], dtype=np.int64)
    v = np.array([_signed(b) for _, b in hs], dtype=np.int64)
    g = group_images(h, v, max_dist=6, max_bucket=100)
    assert g[0] == g[3] >= 0 and g[1] == -1 and g[2] == -1


def test_clean_owner_response():
    from src.labeltool.build import _clean_owner
    raw = "Thông tin phản hồi từ chủ sở hữu 2 năm trước\nCảm ơn bạn! … Xem thêm\nBản dịch của Google ・ Xem bản gốc (Tiếng Anh)"
    assert _clean_owner(raw) == "2 năm trước\nCảm ơn bạn! …"
    assert _clean_owner("Thông tin phản hồi từ chủ sở hữu 1 năm trước") == ""
