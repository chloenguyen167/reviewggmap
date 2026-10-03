"""Sinh dữ liệu DEMO giả lập để chạy thử toàn bộ pipeline khi chưa có dữ liệu thật.

!!! SỐ LIỆU TỪ DỮ LIỆU NÀY KHÔNG ĐƯỢC ĐƯA VÀO PAPER !!!

Dữ liệu được ghi ĐÚNG ĐỊNH DẠNG THÔ của scraper Playwright (raw/reviews/pw_*.jsonl,
raw/network/<place>/*.txt, *.done.json) để kiểm thử luôn bước parse, gồm:
  - ngày tương đối ("3 tháng trước"), ~85% review có timestamp trong gói RPC giả lập,
    ~3% review "Đã chỉnh sửa", ~2% timestamp RPC sai (để kiểm tra cơ chế loại bỏ)
  - review thật: tỷ lệ theo ngày có mùa vụ cuối tuần + Tết, 2 quán có đợt khai trương THẬT
    (burst hợp lệ → ca dương tính giả tiềm năng cho Module 2)
  - 5 kiểu chiến dịch spam: copy_paste, burst_5star, bomb_1star, ad_spam, account_farm
Sự thật (nhãn giả lập) ghi vào <root>/_truth/truth.parquet — pipeline KHÔNG đọc file này.
Thêm một benchmark kiểu ViSpamReviews giả lập (TMĐT) để chạy thử Module 1b.
"""
from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger
from src.utils.config import Paths

SCRAPED_AT = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)

CATS = {
    "cafe": {"n": "Cà phê", "items": ["cà phê sữa đá", "bạc xỉu", "trà đào cam sả", "cold brew", "matcha latte",
                                      "sinh tố bơ", "cà phê muối", "trà vải"]},
    "food": {"n": "Quán", "items": ["bún bò", "cơm tấm sườn", "lẩu thái", "phở bò", "bánh xèo", "gỏi cuốn",
                                    "mì quảng", "hủ tiếu nam vang", "cơm gà"]},
    "spa": {"n": "Spa", "items": ["gói massage body", "chăm sóc da mặt", "gội đầu dưỡng sinh", "tẩy tế bào chết",
                                  "massage chân"]},
    "hotel": {"n": "Khách sạn", "items": ["phòng deluxe", "bữa sáng buffet", "hồ bơi", "phòng đôi", "dịch vụ đưa đón"]},
}
PLACE_WORDS = ["Sài Gòn", "Mộc", "An Nhiên", "Góc Phố", "Hạnh", "Bến Thành", "Hoa Sữa", "Xưa", "Nhà Mình", "Tre Xanh",
               "Phố Cũ", "Mây", "Ông Bảy", "Cô Ba", "Hẻm 51", "Lá", "Nắng", "Vườn", "Chill", "Thanh Đa"]
GOOD_ADJ = ["ngon", "rất ngon", "đậm đà", "vừa miệng", "tươi", "chất lượng", "ổn áp", "xuất sắc"]
SPACE_ADJ = ["thoáng mát", "sạch sẽ", "ấm cúng", "hơi chật", "decor đẹp", "yên tĩnh", "khá ồn giờ trưa", "rộng rãi"]
STAFF_ADJ = ["thân thiện", "nhiệt tình", "hơi chậm", "chu đáo", "dễ thương", "phục vụ nhanh"]
PRICE_ADJ = ["hợp lý", "hơi cao", "bình dân", "đáng tiền", "hơi mắc so với chất lượng"]
WHO = ["đi với gia đình", "đi cùng bạn bè", "đi một mình", "đi với đồng nghiệp", "dẫn người yêu đi"]
NEG = ["{i} hôm nay hơi mặn, không như lần trước", "chờ gần {m} phút mới có {i}", "nhân viên thái độ không tốt",
       "{i} nguội, trình bày sơ sài", "bàn ghế dơ, không ai dọn", "giá {p}k mà chất lượng không tương xứng"]
DETAIL = ["mình ghé lúc {h}h thứ {d}, {w}", "gửi xe mất {m} phút vì hết chỗ", "lần thứ {n} ghé rồi, {x}",
          "nhóm {n} người, gọi thêm {i}", "{x}, chờ khoảng {m} phút", "hôm đó trời mưa nên {x}",
          "đặt bàn trước {m} phút vẫn có chỗ, {x}", "dịp sinh nhật bạn, {w}", "tầm {h}h hơi đông, {x}",
          "ở gần công ty nên hay ghé, {x}", "lần trước gọi {i} thấy ổn hơn", "được bạn giới thiệu, {x}"]
EXTRA = ["wifi hơi yếu", "nhạc hơi to", "có chỗ ngồi ngoài trời", "view đẹp buổi tối", "máy lạnh mát",
         "toilet sạch", "có ổ cắm sạc", "bãi xe rộng", "hơi khó tìm đường vào", "có menu chay",
         "phần ăn hơi ít", "ly to", "đồ uống ra nhanh", "chủ quán vui tính", "hay có khuyến mãi cuối tuần",
         "không gian hợp làm việc", "trẻ em có chỗ chơi", "gần chợ nên hơi ồn"]
EN = ["Great {i}, friendly staff.", "The {i} was good but service was slow.", "Nice place, fair prices.",
      "Not bad, too crowded though.", "Loved the {i}.", "Decent {i}, average vibe.", "Solid spot for {i}.",
      "Friendly owner, tasty {i}.", "Would come back for the {i}.", "Okay experience overall."]

EN_DETAIL = ["We came around {h}pm with {n} friends.", "Waited about {m} minutes for the {i}.",
             "My {n}th visit here, still consistent.", "Tried the {i} this time, would order again.",
             "Parking took {m} minutes, a bit annoying.", "Came here after work at {h}pm, quite busy.",
             "Group of {n}, they found us a table quickly.", "The {i} came out in about {m} minutes."]
EXTRA_EN = ["", "Wifi was decent.", "Music a bit loud.", "Nice view at night.", "Clean toilets.",
            "Hard to find the entrance.", "Good for working.", "Kids friendly.", "Small portions though."]
COPY_TEMPLATES = [
    "Quán rất tuyệt vời, đồ ăn ngon, nhân viên thân thiện nhiệt tình, không gian sạch sẽ thoáng mát, giá cả hợp lý, mọi người nên đến thử nhé",
    "Mình đã trải nghiệm và rất hài lòng, chất lượng dịch vụ tốt, phục vụ chuyên nghiệp, sẽ giới thiệu cho bạn bè người thân cùng đến",
    "Địa điểm lý tưởng cho gia đình và bạn bè, không gian đẹp, sản phẩm chất lượng, giá cả phải chăng, nhất định sẽ quay lại nhiều lần nữa",
    "Dịch vụ quá tuyệt vời luôn, nhân viên tư vấn tận tình chu đáo, cơ sở vật chất hiện đại sạch sẽ, rất đáng để trải nghiệm mọi người ơi",
    "Tuyệt vời không có gì để chê, từ không gian đến chất lượng đều xuất sắc, phục vụ nhanh nhẹn lịch sự, xứng đáng năm sao",
    "Lần đầu ghé mà ấn tượng quá, mọi thứ đều hoàn hảo, nhân viên dễ thương, giá tốt, ai chưa đến thì nên đến thử ngay",
]
GENERIC_POS = ["Tuyệt vời", "Rất tốt", "Quán ngon, phục vụ tốt, sẽ ủng hộ dài dài", "Tuyệt vời, 10 điểm",
               "Dịch vụ tốt, sẽ quay lại", "Quá ok luôn mọi người", "Chất lượng, đáng tiền", "Rất hài lòng",
               "Nhân viên nhiệt tình, không gian đẹp", "Số 1 khu vực này"]
GENERIC_NEG = ["Lừa đảo, đừng đến", "Tệ nhất từng thấy", "Phục vụ quá tệ, không bao giờ quay lại",
               "Đồ ăn dở, giá cắt cổ", "Mọi người tránh xa quán này", "Thất vọng", "Chủ quán thái độ lồi lõm"]
AD_TEMPLATES = ["Bên mình nhận thiết kế decor quán giá rẻ, liên hệ zalo 09{d} để được tư vấn",
                "Ai cần mua {i} giá sỉ inbox hoặc gọi 0{d} nha, ship toàn quốc",
                "Ghé shop mình ở shopee link shp.ee/{w} đang sale 50% mọi người ơi",
                "Khuyến mãi lớn tháng này, follow page để nhận voucher, hotline 0{d}"]


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _rel_text(age: timedelta) -> str:
    d = age.total_seconds() / 86400
    if d < 1:
        h = max(int(d * 24), 1)
        return "một giờ trước" if h == 1 else f"{h} giờ trước"
    if d < 7:
        n = int(d)
        return "một ngày trước" if n == 1 else f"{n} ngày trước"
    if d < 30:
        n = int(d // 7)
        return "một tuần trước" if n == 1 else f"{n} tuần trước"
    if d < 365:
        n = int(d // 30.44)
        return "một tháng trước" if n == 1 else f"{n} tháng trước"
    n = int(d // 365.25)
    return "một năm trước" if n == 1 else f"{n} năm trước"


class Gen:
    def __init__(self, cfg: dict):
        dc = cfg.get("demo", {})
        self.rng = np.random.default_rng(dc.get("seed", 42))
        self.n_places = dc.get("n_places", 50)
        self.days = dc.get("days", 730)
        self.n_acc = dc.get("n_genuine_accounts", 6000)
        self.reviews: list[dict] = []
        self.acc_id = 10 ** 20

    def pick(self, seq):
        return seq[self.rng.integers(len(seq))]

    # ---- tài khoản ------------------------------------------------------------
    def new_account(self, kind: str) -> dict:
        self.acc_id += int(self.rng.integers(1, 10 ** 6))
        if kind == "genuine":
            n = int(max(1, self.rng.lognormal(2.6, 1.3)))
            lg = self.rng.random() < 0.35 and n > 5
            ph = int(n * self.rng.uniform(0, 3)) if self.rng.random() < 0.7 else 0
        else:                                                  # tài khoản ảo
            n = int(self.rng.integers(1, 5))
            lg, ph = False, int(self.rng.random() < 0.15)
        return {"cid": str(self.acc_id), "name": f"User {self.acc_id % 100000}", "n": n, "lg": lg, "ph": ph}

    # ---- văn bản thật ---------------------------------------------------------
    def genuine_text(self, cat: str, rating: int) -> str:
        r = self.rng
        if r.random() < 0.30:
            return ""                                         # chỉ chấm sao
        if r.random() < 0.07:                                 # tiếng Anh, cũng có chi tiết riêng
            det = [self.pick(EN_DETAIL).format(h=int(r.integers(6, 22)), n=int(r.integers(2, 9)),
                                               m=int(r.integers(5, 45)), i=self.pick(CATS[cat]["items"]))
                   for _ in range(2)]
            return " ".join([self.pick(EN).format(i=self.pick(CATS[cat]["items"]))] + det + [self.pick(EXTRA_EN)])
        it = self.pick(CATS[cat]["items"])
        parts = []
        if rating >= 4:
            pool = [f"{it} {self.pick(GOOD_ADJ)}", f"không gian {self.pick(SPACE_ADJ)}",
                    f"nhân viên {self.pick(STAFF_ADJ)}", f"giá {self.pick(PRICE_ADJ)}",
                    f"hôm nay {self.pick(WHO)}, gọi {it} với {self.pick(CATS[cat]['items'])}",
                    f"hết khoảng {int(r.integers(4, 60)) * 10}k cho {int(r.integers(1, 5))} người"]
        else:
            pool = [t.format(i=it, m=int(r.integers(15, 60)), p=int(r.integers(5, 50)) * 10) for t in NEG] + \
                   [f"không gian {self.pick(SPACE_ADJ)}", f"giá {self.pick(PRICE_ADJ)}"]
        k = int(r.integers(2, 5))
        idx = r.choice(len(pool), size=min(k, len(pool)), replace=False)
        parts = [pool[i] for i in idx]
        if r.random() < 0.8:                                 # chi tiết cá nhân → review thật hiếm khi trùng nhau
            parts.append(self.pick(DETAIL).format(
                h=int(r.integers(6, 22)), d=int(r.integers(2, 8)), n=int(r.integers(2, 9)),
                m=int(r.integers(5, 45)), i=self.pick(CATS[cat]["items"]), w=self.pick(WHO),
                x=self.pick(EXTRA)))
        tail = self.pick(["", "Sẽ quay lại.", "Recommend nha.", "Lần sau ghé tiếp.", ""]) if rating >= 4 else \
            self.pick(["", "Chắc không quay lại.", "Hơi thất vọng."])
        txt = (", ".join(parts)).capitalize() + ". " + tail
        if r.random() < 0.12:                                # teencode
            txt = txt.replace("không", "ko").replace("được", "dc").replace("rồi", "r")
        if r.random() < 0.08:
            txt = _strip_accents(txt)
        return txt.strip()

    # ---- ghi một review -------------------------------------------------------
    def add(self, place, acc, rating, text, when, spam=None, campaign=None):
        rid = "ChZDSUhNMG9nS0VJQ0FnSU" + "".join(self.pick(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))
                                           for _ in range(14))
        self.reviews.append({"place_key": place["place_key"], "review_id": rid, "acc": acc, "rating": int(rating),
                             "text": text, "true_date": when, "is_spam": spam is not None, "spam_type": spam,
                             "campaign_id": campaign, "category": place["category"]})

    def rand_time(self, day0: datetime) -> datetime:
        return day0 + timedelta(hours=float(self.rng.uniform(7, 22)))

    # ---- sinh toàn bộ ---------------------------------------------------------
    def build(self):
        r = self.rng
        start = (SCRAPED_AT - timedelta(days=self.days)).replace(hour=0, minute=0, second=0, microsecond=0)
        cats = list(CATS)
        places = []
        for i in range(self.n_places):
            cat = cats[i % len(cats)] if i < 40 else self.pick(cats)
            name = f"{CATS[cat]['n']} {self.pick(PLACE_WORDS)} {i + 1}"
            lat, lng = 10.78 + r.normal(0, 0.06), 106.69 + r.normal(0, 0.06)
            pid = "ChIJ" + "".join(self.pick(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")) for _ in range(23))
            url = (f"https://www.google.com/maps/place/{name.replace(' ', '+')}/data=!4m7!3m6!1s0x31752f{i:010x}:0x{i:08x}"
                   f"!8m2!3d{lat:.6f}!4d{lng:.6f}!16s%2Fg%2F11demo{i}!19s{pid}")
            places.append({"place_key": pid, "place_id": pid, "name": name, "url": url, "category": cat,
                           "lat": lat, "lng": lng, "base_rate": float(r.uniform(0.15, 2.2)),
                           "quality": float(r.uniform(3.6, 4.7))})
        accounts = [self.new_account("genuine") for _ in range(self.n_acc)]
        tet = {datetime(2025, 1, 29, tzinfo=timezone.utc), datetime(2026, 2, 17, tzinfo=timezone.utc)}

        # 1) Review thật
        for p in places:
            opening = p is places[3] or p is places[17]            # 2 quán có đợt khai trương thật
            open_day = start + timedelta(days=int(r.integers(200, 600)))
            for d in range(self.days):
                day0 = start + timedelta(days=d)
                lam = p["base_rate"] * (1.35 if day0.weekday() >= 5 else 1.0) * (1 + 0.3 * d / self.days)
                if any(abs((day0 - t).days) <= 4 for t in tet):
                    lam *= 1.8
                if opening and 0 <= (day0 - open_day).days < 3:
                    lam += 9.0                                      # khai trương: nhiều review thật
                for _ in range(r.poisson(lam)):
                    q = p["quality"] + (0.4 if opening and 0 <= (day0 - open_day).days < 3 else 0)
                    rating = int(np.clip(round(r.normal(q, 1.0)), 1, 5))
                    acc = accounts[int(r.integers(len(accounts)))]
                    self.add(p, acc, rating, self.genuine_text(p["category"], rating), self.rand_time(day0))

        def rand_day(lo=60):
            return start + timedelta(days=int(r.integers(lo, self.days - 5)))

        cid = 0
        # 2) copy_paste: 1 mẫu văn bản, sửa nhẹ, rải nhiều quán trong vài ngày
        for tpl in COPY_TEMPLATES:
            cid += 1
            d0 = rand_day()
            targets = r.choice(len(places), size=int(r.integers(4, 9)), replace=False)
            for t in targets:
                for _ in range(int(r.integers(1, 3))):
                    words = tpl.split()
                    if r.random() < 0.5:
                        words.pop(int(r.integers(len(words))))
                    if r.random() < 0.3:
                        j = int(r.integers(len(words) - 1))
                        words[j], words[j + 1] = words[j + 1], words[j]
                    self.add(places[t], self.new_account("fake"), 5, " ".join(words) + self.pick(["", "!", " 5 sao!", " ❤️❤️"]),
                             self.rand_time(d0 + timedelta(days=int(r.integers(0, 4)))), "copy_paste", f"cp{cid}")
        # 3) burst_5star
        for t in r.choice(len(places), size=5, replace=False):
            cid += 1
            d0 = rand_day()
            for _ in range(int(r.integers(15, 36))):
                txt = "" if r.random() < 0.4 else self.pick(GENERIC_POS)
                self.add(places[t], self.new_account("fake"), 5, txt,
                         self.rand_time(d0 + timedelta(days=int(r.integers(0, 3)))), "burst_5star", f"b{cid}")
        # 4) bomb_1star
        for t in r.choice(len(places), size=3, replace=False):
            cid += 1
            d0 = rand_day()
            for _ in range(int(r.integers(10, 21))):
                txt = "" if r.random() < 0.35 else self.pick(GENERIC_NEG)
                self.add(places[t], self.new_account("fake"), int(self.pick([1, 1, 1, 2])), txt,
                         self.rand_time(d0 + timedelta(days=int(r.integers(0, 2)))), "bomb_1star", f"n{cid}")
        # 5) ad_spam
        for _ in range(60):
            p = self.pick(places)
            txt = self.pick(AD_TEMPLATES).format(d=int(r.integers(10 ** 7, 10 ** 8)), i=self.pick(CATS[p["category"]]["items"]),
                                                 w=int(r.integers(10 ** 5, 10 ** 6)))
            self.add(p, self.new_account("fake"), int(self.pick([5, 5, 1])), txt, self.rand_time(rand_day(5)),
                     "ad_spam", "ad")
        # 6) account_farm: 1 tài khoản, nhiều quán xa nhau, cùng ngày
        far = sorted(places, key=lambda p: p["lat"] + p["lng"])
        for _ in range(15):
            cid += 1
            acc = self.new_account("fake")
            d0 = rand_day()
            k = int(r.integers(4, 9))
            chosen = list(far[:10]) + list(far[-10:])
            for t in r.choice(len(chosen), size=k, replace=False):
                txt = "" if r.random() < 0.3 else self.pick(GENERIC_POS)
                self.add(chosen[t], acc, 5, txt, self.rand_time(d0), "account_farm", f"f{cid}")
        return places

    # ---- ghi ra định dạng thô Playwright ---------------------------------------
    def write(self, P: Paths, places: list[dict]):
        r = self.rng
        rev = pd.DataFrame(self.reviews)
        rev = rev[rev["true_date"] < SCRAPED_AT]
        (P.raw / "places").mkdir(parents=True, exist_ok=True)
        cnt = rev.groupby("place_key").size()
        pl = pd.DataFrame(places)
        pl["rating"] = pl["place_key"].map(rev.groupby("place_key")["rating"].mean().round(1))
        pl["user_rating_count"] = pl["place_key"].map(cnt).fillna(0).astype(int)
        pl[["place_key", "place_id", "name", "url", "category", "lat", "lng", "rating", "user_rating_count"]].to_csv(
            P.raw / "places" / "places_selected.csv", index=False, encoding="utf-8-sig")

        truth = []
        url_of = pl.set_index("place_key")["url"].to_dict()
        for key, g in rev.groupby("place_key"):
            recs, net = [], []
            for x in g.sort_values("true_date", ascending=False).itertuples():
                acc = x.acc
                shown = x.true_date
                edited = r.random() < 0.03
                if edited:
                    shown = min(x.true_date + timedelta(days=float(r.uniform(20, 200))), SCRAPED_AT - timedelta(hours=2))
                date_text = ("Đã chỉnh sửa " if edited else "") + _rel_text(SCRAPED_AT - shown)
                meta = ("Local Guide · " if acc["lg"] else "") + f"{acc['n']} bài đánh giá" + (
                    f" · {acc['ph']} ảnh" if acc["ph"] else "")
                owner = "Cảm ơn bạn đã ủng hộ quán!" if r.random() < 0.15 else None
                recs.append({"source": "playwright", "place_key": key, "place_url": url_of[key],
                             "scraped_at": SCRAPED_AT.isoformat(), "review_id": x.review_id,
                             "reviewer_name": acc["name"], "reviewer_meta": meta, "rating_label": f"{x.rating} sao",
                             "date_text": date_text, "text": x.text, "owner_response": owner,
                             "n_photos": int(r.random() < 0.2) * int(r.integers(1, 4)),
                             "profile_url": f"https://www.google.com/maps/contrib/{acc['cid']}/reviews",
                             "contributor_id": acc["cid"]})
                if r.random() < 0.85:                                  # có gói RPC cho review này
                    ts = int(x.true_date.timestamp() * 1e6)
                    if r.random() < 0.02:
                        ts -= int(400 * 86400 * 1e6)                   # timestamp sai → phải bị loại
                    rec = [x.review_id, [acc["name"], None, [ts]], x.text]
                    if owner:
                        rec.append(["owner", [ts + int(3 * 86400 * 1e6)]])
                    net.append(rec)
                truth.append({"review_id": x.review_id, "is_spam": x.is_spam, "spam_type": x.spam_type,
                              "campaign_id": x.campaign_id, "true_date": x.true_date})
            with open(P.raw / "reviews" / f"pw_{key}.jsonl", "w", encoding="utf-8") as f:
                for rc in recs:
                    f.write(json.dumps(rc, ensure_ascii=False) + "\n")
            nd = P.raw / "network" / key
            nd.mkdir(parents=True, exist_ok=True)
            for j in range(0, len(net), 10):
                body = ")]}'\n" + json.dumps([None, net[j:j + 10], key], ensure_ascii=False)
                (nd / f"{j // 10:05d}.txt").write_text(body, encoding="utf-8")
            (P.raw / "reviews" / f"pw_{key}.done.json").write_text(json.dumps(
                {"place_key": key, "expected": int(len(g) * 1.02), "fetched": len(g)}), encoding="utf-8")
        tdir = P.root / "_truth"
        tdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(truth).to_parquet(tdir / "truth.parquet", index=False)
        return rev


def make_benchmark(P: Paths, seed: int = 7) -> None:
    """Benchmark kiểu ViSpamReviews (TMĐT) GIẢ LẬP — chỉ để chạy thử Module 1b."""
    r = np.random.default_rng(seed)
    prods = ["áo thun", "tai nghe", "ốp lưng", "son môi", "giày thể thao", "bình giữ nhiệt", "sạc dự phòng", "váy"]
    good = ["{p} đẹp, đúng mô tả, giao hàng nhanh", "chất lượng {p} ổn so với giá", "{p} dùng tốt, đóng gói cẩn thận",
            "shop tư vấn nhiệt tình, {p} y hình", "{p} hơi khác màu nhưng vẫn ok", "{p} bị lỗi nhẹ, shop đổi nhanh"]
    bad = ["{p} kém chất lượng, không giống hình", "giao sai màu {p}, nhắn shop không trả lời", "{p} dùng 2 ngày đã hỏng"]
    spam = ["đánh giá để nhận xu", "shop ơi cho mình xin mã giảm giá", "mn ghé shop mình nha link shp.ee/{n}",
            "abc xyz test đánh giá", "hôm nay trời đẹp quá", "follow mình mình follow lại nha",
            "liên hệ zalo 09{n} mua sỉ", "kkkkkkkk", "được tặng voucher nên đánh giá 5 sao"]
    rows = []
    for _ in range(3000):
        p = prods[r.integers(len(prods))]
        if r.random() < 0.3:
            t, y = spam[r.integers(len(spam))].format(n=int(r.integers(10 ** 6, 10 ** 8))), 1
        else:
            t, y = (good if r.random() < 0.8 else bad)[r.integers(3)].format(p=p), 0
            if r.random() < 0.5:
                t += ", " + good[r.integers(len(good))].format(p=prods[r.integers(len(prods))])
        rows.append({"Comment": t, "Label": y})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed)
    d = P.benchmark / "vispamreviews_v2"
    d.mkdir(parents=True, exist_ok=True)
    n = len(df)
    df.iloc[: int(0.8 * n)].to_csv(d / "train.csv", index=False)
    df.iloc[int(0.8 * n): int(0.9 * n)].to_csv(d / "dev.csv", index=False)
    df.iloc[int(0.9 * n):].to_csv(d / "test.csv", index=False)


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("demo", P.logs)
    if not str(P.root).rstrip("/").endswith("demo/data"):
        raise RuntimeError("Chỉ sinh dữ liệu demo khi dùng config.demo.yaml (paths.root = demo/data).")
    g = Gen(cfg)
    places = g.build()
    rev = g.write(P, places)
    make_benchmark(P)
    log.info("DEMO: %d quán, %d review (%d spam, %s)", len(places), len(rev), int(rev["is_spam"].sum()),
             rev["spam_type"].value_counts().to_dict())
    return P.raw
