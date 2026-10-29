# Phát hiện đánh giá giả mạo trên Google Maps tại Việt Nam

Đồ án môn **Kỹ thuật Xử lý Dữ liệu** (nhóm 12). Kho mã này là một **pipeline xử lý dữ liệu** từ đầu đến cuối: thu thập review công khai trên Google Maps, kiểm tra chất lượng và làm sạch, tính nhiều tín hiệu nghi vấn, rồi **gộp chúng bằng weak supervision**: 8 luật đơn giản bỏ phiếu và mô hình **Dawid–Skene** tự học độ tin cậy của từng luật, **không cần nhãn để huấn luyện**. Nhãn người chỉ dùng để đánh giá.

> **Điểm quan trọng nhất cần nhớ:** hệ thống **không kết luận** review nào là giả. Nó chấm một điểm để xếp hạng "review nào nên được xem trước". Kết luận cuối cùng cần con người đọc. Vì vậy mỗi review đều kèm **phiếu của từng luật** (luật nào cho là spam, luật nào cho là thật) để người đọc kiểm tra lại được.

## Mục lục

1. [Bài toán và ý tưởng](#1-bài-toán-và-ý-tưởng)
2. [Sơ đồ pipeline](#2-sơ-đồ-pipeline)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt](#4-cài-đặt)
5. [Chạy thử bằng dữ liệu demo](#5-chạy-thử-bằng-dữ-liệu-demo)
6. [Quy trình với dữ liệu thật (từng bước)](#6-quy-trình-với-dữ-liệu-thật-từng-bước)
7. [Dữ liệu: nguồn, lược đồ và cách xử lý](#7-dữ-liệu-nguồn-lược-đồ-và-cách-xử-lý)
8. [Các module chấm điểm](#8-các-module-chấm-điểm)
9. [Gán nhãn: người + Claude](#9-gán-nhãn-người--claude)
10. [Đánh giá](#10-đánh-giá)
11. [Quyền riêng tư và đạo đức](#11-quyền-riêng-tư-và-đạo-đức)
12. [Cấu hình (`config.yaml`)](#12-cấu-hình-configyaml)
13. [Bảng tất cả các lệnh](#13-bảng-tất-cả-các-lệnh)
14. [Đầu ra và mục tương ứng trong paper](#14-đầu-ra-và-mục-tương-ứng-trong-paper)
15. [Xử lý sự cố](#15-xử-lý-sự-cố)
16. [Hạn chế đã biết](#16-hạn-chế-đã-biết)
17. [Tài liệu liên quan](#17-tài-liệu-liên-quan)

---

## 1. Bài toán và ý tưởng

**Vấn đề.** Google Maps không có cơ chế "đã mua hàng" như sàn thương mại điện tử, nên ai cũng đăng review được. Review giả (do dịch vụ bán review, quán tự viết, hoặc đối thủ dìm điểm) làm sai lệch quyết định của người dùng.

**Khó khăn.** (a) Chưa có bộ dữ liệu công khai nào gán nhãn thật/giả cho review Google Maps tiếng Việt; (b) review giả ngày càng giống review thật, đọc từng review riêng lẻ gần như không phân biệt được; (c) dữ liệu Google Maps mỏng: mỗi tài khoản thường chỉ có 1–2 review quan sát được.

**Ý tưởng.** Không dựa vào một dấu hiệu, mà kết hợp nhiều nhóm tín hiệu. Các module dưới đây tính bằng chứng thô:

| Tín hiệu | Câu hỏi nó trả lời | Ví dụ dấu hiệu |
|---|---|---|
| **M1a** Văn bản gần trùng | Nội dung này có bị sao chép/biến tấu nhẹ ở review khác (nhất là ở quán khác) không? | Một câu khen xuất hiện y hệt ở 6 quán |
| **M1b** Nội dung quảng cáo | Review có phải quảng cáo không? | Có link, số điện thoại, Zalo, "liên hệ", "voucher" |
| **M2a** Bất thường thời gian trong quán | Có đợt review dồn cục bất thường không? | Quán thường 0 review/ngày bỗng có 49 review 5★ trong một ngày |
| **M2b** Phối hợp liên quán | Các bản sao của cùng một nội dung có được đăng dồn ở nhiều quán trong vài ngày không? | Một câu xuất hiện ở 8 quán khác nhau trong cùng một tuần |
| **M3** Hành vi tài khoản | Tài khoản này trông có tự nhiên không? | Tài khoản mới, 1 review, không ảnh, luôn chấm 1★ hoặc 5★ |

**Phương pháp chính: weak supervision + Dawid–Skene** (mục 8.6). Bằng chứng của các module được viết thành **8 luật** một dòng (*labeling function*), mỗi luật bỏ phiếu *Spam*, *Thật* hoặc bỏ phiếu trắng. Mô hình **Dawid–Skene** nhìn các luật đồng ý/bất đồng với nhau để ước lượng độ tin cậy của từng luật (thuật toán EM, không dùng nhãn), rồi tính **P(spam)** cho từng review. Review được xếp theo P(spam); đội kiểm duyệt đọc top 5% hoặc 10%.

Cách gộp cũ (điểm Fisher, 3 vùng `auto_flag`/`review`/`safe`, mục 8.5) vẫn còn trong code và được dùng làm **baseline**.

**Vì sao không huấn luyện một mô hình phân loại?** Vì không có nhãn: chưa có bộ review Google Maps tiếng Việt nào gán nhãn sẵn, và mô hình học trên ViSpamReviews (thương mại điện tử) gắn cờ 42,8% review Google Maps. Weak supervision dùng kiến thức chuyên môn (luật) thay cho nhãn; tập nhãn nhỏ chỉ dùng để **đo** hệ thống đúng bao nhiêu.


---

> **Trạng thái (29/10/2026):** đã thu thập 119 địa điểm (114.542 review sau làm sạch, 100% có ngày chính xác). Phương pháp chính đã chốt là **weak supervision + Dawid–Skene** (`--stages ws`). Đánh giá trên 568 review gán nhãn toàn bộ theo 4 quán: **ROC-AUC 0,899, PR-AUC 0,715** (đoán mò 0,205), hơn có ý nghĩa thống kê các baseline luật đơn giản, học chuyển miền và Isolation Forest; khi đọc 10% review tìm được 73% spam đã gán nhãn (pipeline Fisher cũ: 48%). Chi tiết và hạn chế ở mục 10.2 và 16. Gold set nhóm A/B (mục 9) chưa gán nhãn.

## 2. Sơ đồ pipeline

```
 A. THU THẬP            Playwright (Chrome tự động) ──► raw/reviews/*.jsonl  (review hiển thị trên trang)
                                                    └─► raw/network/*/*.txt (gói RPC của trang: chứa ngày giờ chính xác)
                                    │
 B. TIỀN XỬ LÝ          parse ──► clean ──► quality (6 chiều) ──► eda
                        (1 schema, ghép timestamp, ẩn danh, gỡ chữ giao diện)
                                    │
 C. CHẤM ĐIỂM           M1a trùng lặp · M1b quảng cáo · M2a burst trong quán · M2b phối hợp liên quán · M3 hành vi
                                    │
                        ws: 8 luật bỏ phiếu ──► Dawid–Skene (EM, không nhãn) ──► P(spam) + phiếu từng luật   [PHƯƠNG PHÁP CHÍNH]
                        fusion (Fisher) ──► Điểm Nghi Vấn + 3 vùng + lý do                                   [baseline]
                                    │
 D. ĐÁNH GIÁ            labeltool (gán nhãn toàn bộ theo quán) ──► labeltool_eval / exp_pu_ws / ws_paper_results
                        sample (gold set) ──► gán nhãn (người A, B + Claude C)
                        ──► agreement (Kappa) ──► metrics (P@K, PR-AUC) ──► ablation ──► sensitivity
```

---

## 3. Cấu trúc thư mục

```
reviewggmap/
├── run_pipeline.py            # Điểm vào duy nhất: chạy từng bước bằng --stages
├── config.yaml                # TOÀN BỘ ngưỡng, đường dẫn, selector (không hard-code trong mã)
├── config.demo.yaml           # Cấu hình cho dữ liệu giả lập (kế thừa config.yaml)
├── requirements.txt
├── .env.example               # Mẫu biến môi trường (copy thành .env, KHÔNG commit .env)
├── resources/teencode_vi.txt  # Từ điển teencode/viết tắt tiếng Việt
├── docs/
│   └── huong_dan_gan_nhan.md          # Tiêu chí gán nhãn (llm_export đọc file này)
├── src/
│   ├── collect/        # Thu thập: scrape_playwright, serpapi_client, parse_raw, rpc_timestamps,
│   │                   #           relative_date, place_url, login, download_benchmark
│   ├── preprocess/     # clean, normalize_vi (chuẩn hóa tiếng Việt), quality (6 chiều), eda
│   ├── modules/        # m1a_duplicate, m1b_content, m1b_phobert, m2_temporal, m3_behavior
│   ├── fusion/         # ws_label_model.py: 8 luật + Dawid–Skene (PHƯƠNG PHÁP CHÍNH)
│   │                   # score.py: gộp điểm Fisher, chia 3 vùng (baseline)
│   ├── evaluation/     # sample_gold, llm_label, agreement, metrics, ablation (gold set)
│   │                   # labeltool_eval (đánh giá trên nhãn theo quán), exp_pu_ws (PU learning vs WS)
│   ├── labeltool/      # công cụ gán nhãn theo quán: build (HTML), media, image_hash
│   ├── synthetic/      # make_demo_data, simulate_annotations (chỉ để chạy thử)
│   └── utils/          # config, common, plotstyle
├── tests/              # 26 test unit + test scraper chạy trên trang Google Maps giả lập
│
│  ── Các thư mục dưới đây do pipeline SINH RA, đã nằm trong .gitignore (không push) ──
├── data/
│   ├── raw/            # reviews/, network/, places/, contributors/  (có tên reviewer: KHÔNG chia sẻ)
│   ├── interim/        # reviews_parsed/clean.parquet, private_reviewer_map.parquet (id thật: KHÔNG chia sẻ)
│   ├── processed/      # features_m1a..m3.parquet, scored_reviews.parquet, ws_scores.parquet
│   ├── labels/         # gold_sample_blind.csv, annotator_A/B/C.csv, gold_final.csv, claude_ai/,
│   │                   # labeltool/ và nhan_review_<người gán>_<thời điểm>.csv (nhãn theo quán)
│   ├── benchmark/      # ViSpamReviews v2 (chỉ dùng cho thí nghiệm lệch miền)
│   └── models/         # mô hình đã huấn luyện (.joblib)
├── reports/            # tables/ (CSV) và figures/ (PNG) cho paper
└── logs/               # pipeline.log, reviews_run.log
```

---

## 4. Cài đặt

**Yêu cầu:** Python 3.10+ (đã chạy trên 3.12), Google Chrome cài trên máy, kết nối mạng ổn định cho bước thu thập. Đã kiểm thử trên macOS; Windows/Linux cần đổi đường dẫn Chrome trong `config.yaml`.

```bash
# 1. Tạo môi trường ảo và cài thư viện
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium  # Chromium dự phòng (bước thu thập thật dùng Chrome thật, xem mục 6.1)

# 2. Tạo file biến môi trường
cp .env.example .env
```

Mở `.env` và điền:

| Biến | Bắt buộc | Ý nghĩa |
|---|---|---|
| `REVIEWER_SALT` | **Có** | Chuỗi ngẫu nhiên dài, dùng để băm tên/ID reviewer. **Đặt một lần rồi giữ nguyên**: đổi salt giữa chừng thì cùng một reviewer sẽ ra hash khác, làm hỏng M1a/M3 |
| `SERPAPI_KEY` | Không | Chỉ cần nếu dùng SerpApi thay Playwright (`collect.source: serpapi`) |
| `HF_TOKEN` | Không | Chỉ cần để tải ViSpamReviews v2 (thí nghiệm lệch miền, mục 8.2) |

Kiểm tra cài đặt:

```bash
python -m pytest -q tests/test_units.py     # 23 test, vài giây, không cần mạng
```

> ⚠️ `tests/test_scraper_mock.py` mở một Chrome thật bằng cùng profile với scraper. **Đừng chạy toàn bộ `pytest tests` khi scraper thật đang chạy**, vì hai Chrome không dùng chung được một profile (test này sẽ lỗi `ProcessSingleton`). Dùng lệnh `tests/test_units.py` ở trên.

---

## 5. Chạy thử bằng dữ liệu demo

Không cần mạng, khoảng 5 phút. Dùng để kiểm tra code chạy đúng và làm quen với các file đầu ra.

```bash
python run_pipeline.py --config config.demo.yaml --stages demo,all,demo_labels,eval
```

Lệnh này: (1) sinh khoảng 55.000 review giả lập cho 50 quán, có cài sẵn 5 kiểu chiến dịch spam (copy-paste, dồn 5★, bão 1★, quảng cáo, nông trại tài khoản) và 2 đợt khai trương *thật*; (2) chạy toàn bộ pipeline; (3) giả lập hai người gán nhãn; (4) tính các chỉ số. Kết quả ở thư mục `demo/`.

> ⚠️ **Số liệu từ demo KHÔNG được đưa vào paper.** Dữ liệu giả lập do chính mã sinh ra, nên kết quả đẹp không chứng minh điều gì về dữ liệu thật.

---

## 6. Quy trình với dữ liệu thật (từng bước)

### 6.1. Chuẩn bị Chrome đã đăng nhập (làm một lần)

Khi chưa đăng nhập, Google Maps hiện **"chế độ bị hạn chế"**: không có tab Bài đánh giá. Vì vậy scraper dùng một profile Chrome đã đăng nhập (nên dùng **tài khoản phụ**, không dùng tài khoản chính).

```bash
python -m src.collect.login --profile .chrome_profile_gc
```

Một cửa sổ Chrome mở ra; đăng nhập Google, đợi script báo `Maps chế độ hạn chế: False` rồi đóng. Sau đó trong `config.yaml → collect.playwright` đặt:

```yaml
user_data_dir: .chrome_profile_gc
executable_path: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"   # đường dẫn Chrome thật của bạn
headless: true
```

Profile tạo bằng Chrome thật chỉ dùng được với **chính Chrome đó**, nên hai dòng trên phải đi cùng nhau. Thư mục profile chứa cookie đăng nhập, **tuyệt đối không push** (đã nằm trong `.gitignore`).

### 6.2. Thử scraper trên 1 quán trước khi chạy cả loạt

Tạo file `data/raw/places/places_selected.csv` có một dòng (cột `name,url`), đặt `headless: false` rồi chạy:

```bash
python run_pipeline.py --stages reviews
```

Quan sát Chrome: scraper phải bấm tab *Bài đánh giá*, chọn sắp xếp *Mới nhất*, rồi cuộn dần. Sau đó kiểm tra bằng:

```bash
python run_pipeline.py --stages parse
# Dòng log phải có: "timestamp RPC dùng=N" với N ≈ số review. Nếu =0, xem mục 15.
```

### 6.3. Chọn địa điểm

```bash
python run_pipeline.py --stages search
```

Tìm theo các truy vấn trong `collect.search_queries` (cà phê, nhà hàng, quán ăn, spa, khách sạn ở một số quận TP.HCM). Kết quả:

- `data/raw/places/places_candidates.csv`: mọi ứng viên tìm được;
- `data/raw/places/places_selected.csv`: đã lọc theo `place_filter` (100–3.000 review). **Hãy mở và chỉnh tay** file này để chọn đúng danh sách quán muốn thu thập (loại quán không phù hợp, cân bằng các ngành).

Ngưỡng 100 là để đủ dữ liệu dựng chuỗi thời gian; ngưỡng 3.000 là để thời gian thu thập không quá dài.

### 6.4. Thu thập review

```bash
python run_pipeline.py --stages reviews
```

- Chạy **tuần tự từng quán**, nghỉ ngẫu nhiên 20–45 giây giữa hai quán (tránh bị Google chặn).
- **Chạy lại được:** quán nào đã có file `*.done.json` sẽ được bỏ qua. Nếu bị ngắt giữa chừng, cứ chạy lại lệnh cũ.
- Tốc độ tham khảo: khoảng **9.000 review/giờ**; 120 quán (~114.000 review) mất cỡ 12–14 giờ.
- Chạy nền, không để máy ngủ (macOS):

```bash
nohup caffeinate -i python run_pipeline.py --stages reviews >> logs/reviews_run.log 2>&1 &
tail -f logs/reviews_run.log          # theo dõi
```

Mỗi quán ghi một dòng như: `Tên quán: lấy 805 review (Google báo 805.0) | gói RPC lưu=168, lỗi=0`. Hai con số đầu phải **khớp nhau** (độ phủ ≈ 100%). Nếu `lỗi` khác 0 hoặc số review lấy được thấp hơn nhiều so với Google báo, xem mục 15.

> ⚠️ Chỉ chạy **một** tiến trình scraper tại một thời điểm, và không mở Chrome thủ công bằng cùng profile trong lúc đó.

Muốn dùng **SerpApi** (trả phí) thay Playwright: đặt `collect.source: serpapi` và `SERPAPI_KEY` trong `.env`. Playwright là nguồn chính của đề tài; SerpApi chỉ là dự phòng.

### 6.5. Xử lý và chấm điểm

```bash
python run_pipeline.py --stages all
```

`all` = `parse → clean → quality → eda → m1a → m1b → m2 → m3 → fusion → sample`. Mất khoảng 1–2 phút cho vài chục nghìn review. Mỗi bước có thể chạy riêng (`--stages m2`, ...) và đọc/ghi file trung gian nên chạy lại nhanh.

Sau đó chạy **phương pháp chính** (dưới 2 giây):

```bash
python run_pipeline.py --stages ws      # 8 luật → Dawid–Skene → data/processed/ws_scores.parquet
```

### 6.6. (Tùy chọn) Lịch sử reviewer cho M3

```bash
python run_pipeline.py --stages contributors   # thu lịch sử 300 reviewer nghi vấn nhất
python run_pipeline.py --stages all            # chạy lại để M3 dùng thêm đặc trưng
```

Bước này mở trang cá nhân của từng reviewer nên chậm; chỉ làm cho nhóm nghi vấn nhất (`collect.contributors.top_n`).

### 6.7. Gán nhãn và đánh giá

Xem [mục 9](#9-gán-nhãn-người--claude) và [mục 10](#10-đánh-giá).

---

## 7. Dữ liệu: nguồn, lược đồ và cách xử lý

### 7.1. Nguồn

Review **công khai** trên Google Maps của khoảng 120 địa điểm ở TP.HCM, lấy bằng Chrome tự động. Không dùng Google Places API vì API này chỉ trả tối đa **5 review/địa điểm** và không phân trang, nên không dựng được chuỗi thời gian.

### 7.2. Vấn đề "ngày tương đối" và cách giải quyết

Trang web chỉ hiện *"3 năm trước"*, nên review càng cũ thì ngày càng mơ hồ (khoảng sai số cả năm). Giải pháp: khi cuộn danh sách, trang tự gọi các endpoint nội bộ (`batchexecute`) trả về **timestamp chính xác tới micro-giây**. Scraper chèn một đoạn script vào trang để giữ lại nội dung các gói này; bước `parse` rút `review_id → timestamp` rồi **chỉ nhận** timestamp nằm trong khoảng ngày suy ra từ ngày tương đối (kiểm tra chéo, loại timestamp sai). Nhờ đó ngày đăng chính xác tới từng ngày, điều kiện bắt buộc cho M2.

### 7.3. Lược đồ sau khi parse (`interim/reviews_parsed.parquet`)

| Nhóm | Trường | Ghi chú |
|---|---|---|
| Review | `review_id`, `rating` (1–5), `text`, `review_date`, `is_edited`, `n_photos`, `has_owner_response`, `text_truncated` | `review_date` lấy từ timestamp RPC; `text` đã gỡ chữ giao diện |
| Tài khoản | `reviewer_hash`, `reviewer_n_reviews`, `reviewer_n_photos`, `reviewer_local_guide` | Tên/ID thật được **băm có salt**; bản đối chiếu ID thật nằm riêng ở `private_reviewer_map.parquet` |
| Địa điểm | `place_key` | Tên, toạ độ, rating Google báo nằm ở `places.parquet` |
| Ngày | `date_text`, `date_lo`, `date_hi`, `date_precision_days`, `date_source` | `date_source = rpc` nghĩa là ngày chính xác |

Sau `clean` có thêm: `lang`, `text_norm`, `text_seg`, `n_words`, `n_urls`, `n_phones`, `n_emojis`, `upper_ratio`, `review_day`, `review_week`.

### 7.4. Các lỗi dữ liệu đã gặp và cách xử lý

| Vấn đề | Biểu hiện | Xử lý |
|---|---|---|
| Lẫn chữ giao diện | Nội dung dính "Xem thêm", khối "Loại đơn đặt món / Đồ ăn: 5" | Selector lấy đúng phần tử nội dung; `parse` gỡ phần còn sót |
| Phản hồi chủ quán bị lấy nhầm | Review **chỉ chấm sao** nhưng `text` là câu trả lời của chủ quán → sinh ra cụm "trùng lặp" giả | Loại các `text` trùng với `owner_response`; selector bỏ phần tử nằm trong khối phản hồi |
| Cắt cụt | Nội dung kết thúc bằng "…" (~10%) | Gắn cờ `text_truncated`; toàn văn có trong các gói RPC đã lưu |
| Bản dịch tự động | Review tiếng nước ngoài hiển thị đã dịch sang tiếng Việt (tỷ lệ tiếng Anh ≈ 0%) | Cần đánh dấu, tránh để M1a coi câu dịch máy là trùng lặp (đang làm) |
| Mất gói RPC | Một số review không có ngày chính xác | Đọc nội dung gói ngay trong trang thay vì qua CDP; log báo số gói lưu/lỗi mỗi quán |

### 7.5. Kiểm tra chất lượng 6 chiều (`reports/tables/data_quality.csv`)

| Chiều | Phép kiểm tra |
|---|---|
| Completeness | % review chỉ chấm sao; % thiếu ngày; % thiếu số review tài khoản; độ phủ thu thập (lấy được / Google báo) |
| Uniqueness | Số `review_id` trùng; số nội dung trùng y hệt ≥ 8 từ (**giữ lại**, vì là tín hiệu spam) |
| Validity | Rating ngoài 1–5; ngày trong tương lai |
| Consistency | Chênh lệch giữa điểm trung bình Google báo và điểm tính lại từ review |
| Timeliness | % review trong 24 tháng gần nhất; % có ngày chính xác tới ngày |
| Accuracy | Đối chiếu tay 20 review ngẫu nhiên với Google Maps (điền cột `khớp` trong `accuracy_check_sample.csv`) |

### 7.6. Làm sạch và chuẩn hóa tiếng Việt

Chuẩn hóa Unicode (NFC), chữ thường, thay teencode/viết tắt theo `resources/teencode_vi.txt` (ví dụ "nc" → "nước"), rút gọn ký tự lặp ("ngonnnn"), tách từ bằng `pyvi`, nhận diện ngôn ngữ, trích đặc trưng thô (link, số điện thoại, emoji, tỷ lệ in hoa). **Không xóa** review trùng nội dung hay review ngắn, vì đó có thể chính là tín hiệu.

---

## 8. Các module chấm điểm

Mỗi module trả về một điểm thô cho từng review; `fusion` đổi thành thứ hạng phần trăm rồi gộp.

### 8.1. M1a — Văn bản gần trùng (`src/modules/m1a_duplicate.py`)

**Ý tưởng:** một mẫu văn bản bị dán hoặc biến tấu nhẹ cho nhiều quán/tài khoản là dấu hiệu spam rõ nhất.

1. Mỗi review (≥ 8 từ) được cắt thành các *shingle* 3 từ liên tiếp.
2. **MinHash (128 hoán vị) + LSH** làm bước *blocking* trên **toàn bộ corpus**: chỉ so những cặp rơi cùng "xô", tránh so mọi cặp (O(N²)). Không chia theo quán, vì spam hay rải ở nhiều quán.
3. Cặp ứng viên được tính Jaccard thật; coi là gần trùng nếu ≥ 0,6.
4. Điểm = độ giống lớn nhất × (số quán khác có bản trùng) × (số tài khoản khác có bản trùng), theo thang log.

**M1a chỉ đo trùng lặp văn bản.** Mọi bằng chứng thời gian thuộc về M2. Việc phân biệt chiến dịch với câu khen sáo rỗng lặp lại tự nhiên ("Nước ngon, nhân viên thân thiện") do **M2b** đảm nhận (mục 8.3).

Đầu ra tham khảo: `m1a_top_clusters.csv` (các cụm trùng lớn nhất, nên đọc bằng mắt để chỉnh ngưỡng).

### 8.2. M1b — Nội dung quảng cáo (`src/modules/m1b_content.py`)

Hiện chạy bằng **luật**: có link, số điện thoại, từ khóa quảng cáo ("liên hệ", "zalo", "inbox", "voucher", "hotline"…), quá nhiều emoji/chữ in hoa.

**Vì sao không dùng mô hình học máy?** Nhóm đã thử huấn luyện TF-IDF + Logistic Regression trên **ViSpamReviews v2** (review sàn thương mại điện tử tiếng Việt, macro-F1 0,817 trên tập test của chính nó). Nhưng khi áp sang Google Maps, mô hình gắn cờ **43%** review, kể cả review thật nói về phục vụ hay không gian: trên sàn TMĐT, review "lạc đề so với sản phẩm" bị tính là spam. Đổi cách định nghĩa nhãn cũng không khắc phục (`reports/tables/m1b_label_variants.csv`). Kết luận: **không dùng dữ liệu khác miền để chấm điểm hay đánh giá**; đây là một phát hiện nên nêu trong phần Discussion. Cấu hình `m1b.mode: rules` giữ chế độ này; `--stages benchmark,m1b` vẫn tái lập được thí nghiệm.

### 8.3. M2 — Bất thường thời gian (M2a trong từng quán, M2b liên quán) (`src/modules/m2_temporal.py`)

**Đơn vị:** (địa điểm, ngày). Chỉ dùng review có ngày chính xác.

1. **Robust z-score** của số review trong ngày so với cửa sổ 28 ngày *trước đó*: `z = (x − median) / max(1,4826·MAD, √max(median,1))`. Dùng median/MAD thay trung bình/độ lệch chuẩn vì số liệu lệch phải và chính đợt tăng đột biến sẽ kéo trung bình. Sàn `√max(median,1)` tránh chia cho 0 ở quán thưa review.
2. **Isolation Forest** trên đặc trưng của ngày (số review, z, lệch điểm, tỷ lệ 5★, tỷ lệ 1★, tỷ lệ tài khoản mới, tỷ lệ không có chữ), bắt được ngày bất thường *đa biến* (số lượng không lớn nhưng toàn 5★ từ tài khoản mới).
3. Ngày bị gắn cờ nếu z ≥ 3 *và* có ≥ 3 review, hoặc IF gắn cờ; mở rộng ±1 ngày.
4. Điểm review = điểm ngày × mức *đồng hướng* (review 5★ trong đợt 5★ đáng ngờ hơn review 2★ trong cùng đợt).

**M2b — phối hợp liên quán** (module riêng trong bước gộp điểm, tên `coord`). Các bước 1–4 ở trên (M2a) chỉ thấy đột biến *trong từng quán*. Một chiến dịch rải mỗi quán vài bản sao sẽ không làm quán nào đột biến: trên demo, M2a chỉ gắn cờ 3/55 review spam sao chép. M2b lấy các cụm gần trùng do M1a tìm ra; với mỗi review có ngày chính xác, nó đếm số bản trùng **ở quán khác** đăng trong ±14 ngày (`m2.coord_window_days`) và so với kỳ vọng nếu các bản trùng rải ngẫu nhiên trên toàn khoảng thời gian của dữ liệu: `λ = (số bản trùng ở quán khác) × 2·14/span`, điểm = `−log10 P(Poisson(λ) ≥ số bản trong cửa sổ)`. Câu sáo rỗng rải qua nhiều năm cho điểm thấp; chiến dịch dồn vào vài ngày cho điểm cao. Review không thuộc cụm nào thì M2b không áp dụng. Thứ tự chạy bắt buộc: `m1a` trước `m2`.

*Vì sao tách ra như vậy?* Lý do là **lập luận thiết kế**: M2a chỉ thấy đột biến trong từng quán, nên không thấy chiến dịch rải mỏng trên nhiều quán. Thành phần này trước đây nằm trong M1a; gỡ hẳn thì mất tín hiệu, để trong M1a thì module trộn văn bản với thời gian. Tách thành M2b giữ được tín hiệu, và mỗi module chỉ đo một loại bằng chứng. Các con số trên demo chỉ để kiểm tra cài đặt, **không phải căn cứ chọn**: AP điểm tổng ở ba biến thể chênh nhau rất ít (bỏ hẳn 0,495 / gộp trong M1a 0,509 / tách M2b 0,536) và chưa có khoảng tin cậy. Ablation (dòng 9, 10) so lại cả ba trên gold set thật.

*Tình trạng trên dữ liệu thật (50 quán đầu):* M2b xác định được trên 237 review thuộc cụm gần trùng nhưng **chỉ khác 0 ở 6 review**, nên mọi con số về M2b (kể cả tương quan 0,21 với M1a) chỉ là sơ bộ. M2b còn phụ thuộc M1a về cấu trúc, vì nó chỉ tính trên review đã trùng.

**Lưu ý:** đợt tăng đột biến có thể là **thật** (khai trương, sự kiện, báo chí nhắc đến). Vì vậy M2 chỉ là một trong các tín hiệu, và `m2_top_bursts.csv` cần đọc tay. Có thể chỉnh `m2.contamination`, `m2.window`, `m2.z_threshold` nếu cờ quá nhiều.

### 8.4. M3 — Hành vi tài khoản (`src/modules/m3_behavior.py`)

Isolation Forest không giám sát trên các đặc trưng như: tài khoản ít review, không phải Local Guide, ít ảnh, sao cực đoan (1★/5★), lệch so với điểm trung bình của quán, review không có chữ hoặc quá ngắn, có link/SĐT, số quán khác nhau cùng reviewer đánh giá trong *cùng một ngày*, khoảng cách địa lý lớn nhất giữa các quán đó. Nếu đã thu `contributors` thì thêm đặc trưng từ lịch sử.

**Mẹo kỹ thuật:** mỗi đặc trưng được đổi chiều thành "lớn = đáng ngờ" rồi *cắt một phía* tại median. Nhờ vậy Isolation Forest chỉ cô lập các điểm lệch về phía đáng ngờ; nếu không, một Local Guide có 5.000 review cũng bị coi là "bất thường".

### 8.5. Fusion — Gộp điểm (`src/fusion/score.py`)

1. Điểm thô → thứ hạng phần trăm [0, 1].
2. Module không áp dụng được cho một review (ví dụ review chỉ chấm sao thì không có M1a/M1b) → **bỏ ra và chuẩn hóa lại trọng số**, không điền 0.
3. Gộp bằng **phương pháp Fisher** (mặc định, `fusion.method: fisher`): coi `1 − s` là p-value của từng module rồi gộp. Lý do: lấy trung bình làm loãng tín hiệu, vì mỗi kiểu spam thường chỉ nổi ở 1–2 module. Cũng có `mean` và `max`; ablation so sánh cả ba.
4. Chia vùng theo phân vị: `auto_flag` (≥ phân vị 98), `review` (≥ phân vị 90), còn lại `safe`. Đây là **hạn ngạch** nhóm chọn để giới hạn khối lượng cần xem tay, không phải ngưỡng đã hiệu chỉnh theo xác suất spam.
5. Xuất `module_correlation.csv` (tương quan Spearman giữa 4 điểm) để kiểm tra giả định độc lập của Fisher.
6. Xuất `scored_reviews.parquet` gồm mọi điểm con, điểm tổng, vùng và cột `reasons` (lý do dạng văn bản).

> Fusion Fisher là cấu hình chốt **trước** khi có nhãn. Trên nhãn thật nó thua weak supervision (mục 10.2), một phần vì M3 dùng Isolation Forest: review thuê là một cụm đông tài khoản mới, không phải điểm "lạ", nên điểm M3 riêng chỉ đạt AUC 0,58. Từ nay fusion là **baseline**; `scored_reviews.parquet` vẫn là đầu vào của bước `ws`.

### 8.6. Phương pháp chính — Weak supervision + Dawid–Skene (`src/fusion/ws_label_model.py`)

**Bước 1. 8 luật (labeling function).** Mỗi luật là một điều kiện đọc được, bỏ phiếu *Spam* (1), *Thật* (0) hoặc bỏ phiếu trắng (−1). Ngưỡng tròn, không tinh chỉnh.

| Luật | Phiếu | Điều kiện | Độ phủ |
|---|---|---|---|
| `LF_dup` | Spam | trùng gần y hệt (≥ 8 từ) với review của tài khoản khác (M1a) | 1,5% |
| `LF_ad` | Spam | có link, số điện thoại, Zalo, từ khóa khuyến mãi (M1b) | 0,7% |
| `LF_coord` | Spam | cùng nội dung ở quán khác trong ±14 ngày (M2b) | 0,2% |
| `LF_burst` | Spam | chấm 1★/5★ trong đợt dồn review của quán (M2a) | 17,1% |
| `LF_thin` | Spam | tài khoản ≤ 3 review, chấm 1★/5★, không phải Local Guide | 22,7% |
| `LF_veteran` | Thật | Local Guide có ≥ 20 review | 35,8% |
| `LF_detailed` | Thật | ≥ 30 từ và không trùng với ai | 16,6% |
| `LF_moderate` | Thật | chấm 2–4★ và không nằm trong đợt dồn | 24,9% |

80% review nhận ít nhất một phiếu.

**Bước 2. Label model Dawid–Skene.** Giả định các luật độc lập với nhau khi đã biết nhãn thật y. Tham số: π = P(spam) và, với mỗi luật, xác suất luật bỏ phiếu Spam/Thật/trắng khi y = spam và khi y = thật. Học bằng EM: khởi tạo bằng bỏ phiếu đa số mềm → ước lượng tham số → tính lại P(spam | phiếu) → lặp tới khi log-likelihood đổi < 10⁻⁷ (tương đối).

**Bước 3. Quyết định.** Xếp review theo P(spam) và đưa top 5%/10% cho người kiểm duyệt. **Không** dùng ngưỡng 0,5: mô hình quá tự tin (ước lượng π = 0,42 trong khi tỷ lệ spam quan sát được là 0,205) vì các luật thin/burst/sao cực đoan hay đi cùng nhau.

**Đầu ra:** `processed/ws_scores.parquet` (`review_id`, `ws_score`, phiếu của 8 luật), `reports/tables/ws_lf_summary.csv` (độ phủ, chồng lấp, xung đột, tham số đã học của từng luật).

---

## 9. Gán nhãn: người + Claude

> **Nhãn đang dùng cho kết quả hiện tại** là nhãn **theo quán** từ công cụ labeltool (cuối mục này), không phải gold set nhóm A/B. Gold set A/B là quy trình đã chuẩn bị nhưng chưa gán.

Vì không có nhãn công khai, nhóm tự xây **gold set** (khoảng 500 review) để đo hệ thống. Tiêu chí chi tiết (6 cờ, quy tắc kết luận, ngoại lệ) nằm ở **[`docs/huong_dan_gan_nhan.md`](docs/huong_dan_gan_nhan.md)**; đây là tóm tắt.

**Lấy mẫu (`--stages sample`, nằm trong `all`):**
- *Nhóm A* (~200): review đứng đầu theo từng module và theo điểm tổng, để có đủ ca dương tính.
- *Nhóm B* (300): ngẫu nhiên đều từ toàn bộ dữ liệu, dùng ước lượng **tỷ lệ spam nền** và tỷ lệ báo động giả. Lấy 300 thay vì 80 vì với tỷ lệ nền ~3,75%, khoảng tin cậy Wilson ở 80 mẫu trải 1,3–10,5% (gần như không đọc được), ở 300 mẫu thu về 2,1–6,5%. (Tỷ lệ nền chỉ được tính từ Nhóm B, vì Nhóm A được chọn theo chính điểm của hệ thống.)
- File gán nhãn `gold_sample_blind.csv` đã **xáo trộn, ẩn mọi điểm số và nhóm**, nhưng có **bối cảnh**: số review của tài khoản, các review khác của cùng người, số review của quán trong ±3 ngày. Bối cảnh là bắt buộc vì đọc một review riêng lẻ gần như không phân biệt được thật/giả.

**Ý nghĩa nhãn:** `Genuine` = *không tìm thấy dấu hiệu spam* (không có nghĩa đã chứng minh là thật); `Uncertain` = có dấu hiệu cụ thể nhưng chưa đủ; `Spam` = có bằng chứng. Review chỉ chấm sao vẫn được gán nhãn, nhưng chỉ dựa vào bối cảnh (cờ f4–f6) và được **báo cáo riêng**.

**Quy trình:**

| # | Ai | Việc | Lệnh |
|---|---|---|---|
| 1 | A và B (độc lập) | Gán 30 dòng đầu (pilot) trong `data/labels/annotator_A.csv`, `annotator_B.csv` (file chỉ chứa phần người gán: toàn bộ Nhóm B + 60 mục Nhóm A, ~360 dòng) | `--stages agreement` để xem Kappa. Nếu **< 0,6**: họp chỉnh hướng dẫn rồi mới gán tiếp |
| 2 | A và B | Gán hết file; hòa giải bất đồng vào `adjudication.csv` (`item_id,label`) | |
| 3 | Bạn | Xuất dữ liệu cho Claude (toàn bộ Nhóm A, gồm 60 mục chung với người để kiểm chứng) | `--stages llm_export` → `data/labels/claude_ai/` có `PROMPT.md` và `batch_01.csv…` |
| 4 | Bạn + Claude (claude.ai) | Tạo một Project, dán `PROMPT.md` vào *Project instructions*. **Mỗi batch một chat mới**: upload `batch_XX.csv`, gõ "Gán nhãn file này", lưu CSV kết quả vào `data/labels/claude_ai/ket_qua/` (hoặc dán nguyên câu trả lời vào file `.txt`) | |
| 5 | Máy | Kiểm tra định dạng, đủ mục, gộp thành `annotator_C.csv` | `--stages llm_import` |
| 6 | Máy | So người với Claude, quyết định dùng nhãn Claude hay không | `--stages agreement` |

Không dùng API: Claude gán qua giao diện claude.ai, không cần API key. Claude chỉ thấy đúng dữ liệu người thấy (kèm 5 review giống nhất trong corpus để thay thao tác Ctrl+F xét "trùng nội dung").

**Nhãn người là nguồn chính.** File gán nhãn đưa cho Claude đúng những bối cảnh mà M1a/M2/M3 dùng để chấm điểm, nên nếu lấy nhãn Claude làm số chính thì phép đánh giá có nguy cơ đo lại chính đặc trưng của hệ thống. Vì vậy P@25, P@50, PR-AUC, tỷ lệ nền, FPR tính trên nhãn người; P@100, P@150 mới mở rộng sang nhãn Claude và ghi rõ nguồn.

**Khi nào được dùng nhãn Claude cho phần còn lại của Nhóm A?** Chỉ khi cả năm điều kiện cùng đúng: κ(A–B) ≥ 0,6; đã hòa giải hết bất đồng của người; có ≥ 30 mục chung; κ(người–Claude) ≥ 0,6; và κ(người–Claude) không thấp hơn κ(A–B) quá 0,1. Kết luận và lý do ghi ở `reports/tables/llm_validation.csv`; `gold_final.csv` có cột `label_source` (`human`/`llm`) để báo cáo tách riêng.

**Gán nhãn theo quán bằng giao diện:** `python run_pipeline.py --stages labeltool_all` tạo `data/labels/labeltool/gan_nhan_review.html`. Đây là một file HTML duy nhất, giao diện giống Google Maps, hiển thị ảnh/video của review, kèm bằng chứng nội dung trùng, ảnh trùng và các review khác của cùng tài khoản ở mọi quán. Người gán chọn một hoặc nhiều quán, gán nhãn bằng chuột hoặc phím tắt, rồi xuất CSV. Hướng dẫn ở mục 6 của [`docs/huong_dan_gan_nhan.md`](docs/huong_dan_gan_nhan.md).

---

## 10. Đánh giá

```bash
python run_pipeline.py --stages eval     # = agreement, metrics, ablation, sensitivity
```

| Chỉ số | Ý nghĩa | File |
|---|---|---|
| **Cohen's Kappa** + % đồng ý | Hai người (và người–Claude) đồng thuận đến đâu. Với cờ hiếm gặp Kappa có thể ≈ 0 dù đồng ý rất cao (nghịch lý Kappa), nên luôn xem cả % đồng ý | `agreement.csv` |
| **Precision@K** | Trong K review điểm cao nhất, bao nhiêu thực sự là Spam. K = 25, 50 trên **nhãn người** (số chính); K = 100, 150 trên nhãn người + Claude (cột `nguồn nhãn`) | `metrics_main.csv`, `fig_pk_curve.png` |
| **PR-AUC** | Chỉ số tổng hợp, không phụ thuộc ngưỡng | `metrics_main.csv`, `fig_pr_curve.png` |
| **Tỷ lệ spam nền** | % Spam trong Nhóm B (kèm khoảng Wilson 95%) | `metrics_main.csv` |
| **FPR vùng auto_flag** | Review Genuine ở Nhóm B mà bị gắn auto_flag | `metrics_main.csv` |
| **Recall ước lượng** | **Chỉ là ước lượng** từ tỷ lệ nền | `metrics_main.csv` |
| **Khoảng tin cậy 95%** | Bootstrap 1.000 lần, vì gold set nhỏ | mọi bảng trên |
| **Tương quan giữa các module** | Kiểm tra giả định độc lập của phép gộp Fisher (Spearman, kèm số review dùng cho từng cặp trong `module_correlation_n.csv`). Code in cảnh báo khi \|ρ\| > 0,3; đây chỉ là ngưỡng nhắc việc trong code, **không** phải tiêu chí chấp nhận đã đăng ký trước, nên khi viết bài chỉ báo cáo con số | `module_correlation.csv` |
| **Độ vững** | Kết quả có đổi không nếu xử lý nhãn Uncertain khác đi (loại / coi là Spam / coi là Genuine), hoặc nếu xét riêng nhóm có chữ / chỉ chấm sao, người / Claude | `metrics_robustness.csv` |
| **Ablation** | Từng module riêng lẻ (gồm M2a, M2b riêng), M1, M1+M2, đầy đủ với 3 cách gộp (**7b Fisher là cấu hình chính, chốt trước khi có nhãn**), trọng số học bằng Logistic Regression (CV 5 fold), và 2 biến thể vị trí tín hiệu phối hợp liên quán: bỏ hẳn / gộp trong M1a như thiết kế cũ | `ablation.csv`, `fig_ablation.png` |
| **Độ nhạy tham số** | Đổi ngưỡng LSH, `contamination`, cửa sổ thời gian → P@50 thay đổi ra sao | `sensitivity.csv` |
| **Phân tích lỗi** | Các ca báo nhầm và bỏ sót điển hình | `fp_top.csv`, `fn_top.csv`, `top100_suspicious.csv` |

Nếu kết quả không như kỳ vọng (ví dụ kết hợp nhiều tín hiệu không hơn một tín hiệu đơn), hãy **báo cáo trung thực** và phân tích nguyên nhân; đó cũng là kết quả có giá trị.

Bảng trên là quy trình cho **gold set** (mục 9). Kết quả hiện có dùng nhãn theo quán, ở mục 10.2.

### 10.2. Đánh giá phương pháp chính trên nhãn theo quán

**Dữ liệu:** các file `data/labels/nhan_review_*.csv` xuất từ labeltool. Hiện có 568 review của 4 quán: 3 quán gán **toàn bộ** review có chữ/ảnh, quán thứ 4 gán 81 review gần nhất. Vì không chọn review theo điểm hệ thống, precision/recall/AUC đọc trực tiếp, không cần hiệu chỉnh thiên lệch chọn mẫu. Nhãn: 85 Spam, 153 Uncertain, 330 Genuine.

```bash
python run_pipeline.py --stages ws                # điểm của phương pháp chính
python -m src.evaluation.labeltool_eval           # hệ thống Fisher vs baseline (bảng labeltool_eval_*.csv)
python -m src.evaluation.exp_pu_ws                # PU learning + weak supervision (exp_pu_ws_*.csv, ~1,5 phút)
python -m src.evaluation.ws_paper_results         # toàn bộ bảng/hình cuối cùng (paper_*.csv, paper_*.pdf|png)
```

`ws_paper_results.py` cần đã chạy `ws` và `exp_pu_ws`.

**Cách so sánh:** 7 baseline (tài khoản mới, checklist 4 luật, PU learning kiểu Spy [Li et al. 2014], Isolation Forest trên đặc trưng thô, TF-IDF + LR học trên ViSpamReviews, fusion Fisher cũ, bỏ phiếu đa số trên cùng 8 luật). Chỉ số: ROC-AUC, PR-AUC (AP), P@50, precision/recall khi đọc top 5%/10% corpus. CI 95% bằng bootstrap phân tầng 2.000 lần; so sánh ghép cặp bằng DeLong (AUC) và bootstrap (ΔAP), hiệu chỉnh Holm. Uncertain xử lý 3 cách (loại / = Spam / = Genuine).

**Kết quả chính (loại Uncertain: 415 review, 85 spam):**

| Phương pháp | ROC-AUC | PR-AUC | WS hơn có ý nghĩa? (AUC / AP) |
|---|---|---|---|
| **WS + Dawid–Skene** | **0,899** [0,86–0,94] | **0,715** [0,64–0,80] | — |
| Bỏ phiếu đa số (8 luật) | 0,899 | 0,695 | không / không |
| Fusion Fisher (cũ) | 0,831 | 0,591 | có / không |
| Checklist 4 luật | 0,838 | 0,571 | có / có |
| Tài khoản mới | 0,842 | 0,511 | có / có |
| PU learning (Spy) | 0,853 | 0,614 | có / không |
| Học chuyển miền ViSpamReviews | 0,615 | 0,297 | có / có |
| Isolation Forest | 0,227 | 0,131 | có / có |

Đọc top 10% corpus: WS tìm được **73%** spam đã gán nhãn (độ chính xác 74%), Fisher 48%, checklist 38%. Ablation: bỏ luật `LF_thin` thì AUC còn 0,829; luật `LF_ad` không mang thông tin. Hạn chế của phép đánh giá này ở mục 16.

---

## 11. Quyền riêng tư và đạo đức

1. Chỉ thu thập dữ liệu **công khai**, tốc độ thấp, có nghỉ giữa các quán.
2. Tên và ID reviewer được **băm có salt** ngay ở bước `parse`; mọi file phân tích chỉ chứa hash.
3. **Không chia sẻ, không nộp kèm, không push**: `data/raw/` (có tên reviewer), `data/interim/private_reviewer_map.parquet` (ID thật), `.env` (salt), và thư mục profile Chrome (cookie đăng nhập). Tất cả đã nằm trong `.gitignore`.
4. **Không công bố tên quán bị gắn "nghi vấn"**: điểm số là xác suất, không phải kết luận, và có thể gây hại cho một cơ sở kinh doanh bị báo nhầm.
5. Khi gán nhãn **không tra cứu danh tính** người viết ở bên ngoài.
6. Khung pháp lý về scraping trong tài liệu tham khảo là của Hoa Kỳ; đề tài áp dụng tinh thần (dữ liệu công khai, ẩn danh, không chia sẻ dữ liệu thô) chứ không viện dẫn điều luật. Việc scrape có thể trái điều khoản dịch vụ của Google: chỉ dùng cho nghiên cứu, số lượng vừa phải.

---

## 12. Cấu hình (`config.yaml`)

Mọi ngưỡng, đường dẫn, selector đều nằm trong file này; **không sửa mã để đổi tham số**. Các nhóm chính:

| Mục | Khóa tiêu biểu | Ý nghĩa |
|---|---|---|
| `paths` | `root`, `reports` | Thư mục dữ liệu (mặc định `data/`). Đổi `root` để chạy trên bản sao dữ liệu mà không đụng dữ liệu gốc |
| `collect.place_filter` | `min_reviews`, `max_reviews` | Lọc quán khi tìm (100–3.000) |
| `collect.playwright` | `headless`, `user_data_dir`, `executable_path`, `max_idle_rounds`, `scroll_pause_ms`, `delay_between_places_s`, `network_url_keywords` | Điều khiển trình duyệt; `network_url_keywords` quyết định gói RPC nào được lưu |
| `collect.playwright.selectors` | `review_card`, `name`, `text`, `date`… | Selector CSS; sửa khi Google đổi giao diện (mục 15) |
| `preprocess` | `teencode_file`, `word_segment` | Chuẩn hóa tiếng Việt |
| `m1a` | `min_words`, `shingle_k`, `num_perm`, `lsh_threshold`, `sim_threshold` | Tham số gần trùng |
| `m1b` | `mode` (`rules`/`model`/`auto`), `ad_keywords` | Chế độ chấm nội dung |
| `m2` | `freq` (D/W), `window`, `z_threshold`, `min_count`, `contamination`, `expand`, `coord_window_days` (M2b) | Tham số burst |
| `m3` | `contamination` | Tỷ lệ bất thường kỳ vọng của IF |
| `fusion` | `method` (`fisher`/`mean`/`max`), `zones.high_quantile`, `zones.low_quantile` | Cách gộp và ngưỡng vùng |
| `evaluation` | `group_a_top_per_module`, `group_b_random`, `pilot_n`, `human_a_n`, `k_list_human`, `k_list_extended`, `kappa_min`, `llm_max_kappa_gap`, `llm.batch_size`, `k_list`, `bootstrap` | Gold set, ngưỡng chấp nhận nhãn Claude, chỉ số |

Mọi tham số trên đều có chú thích ngay trong `config.yaml`. Để thử một cấu hình khác mà không sửa file chính, tạo file mới có dòng `base: config.yaml` rồi ghi đè các khóa cần đổi (xem `config.demo.yaml` làm mẫu), và truyền bằng `--config`.

---

## 13. Bảng tất cả các lệnh

Cú pháp: `python run_pipeline.py [--config FILE] --stages a,b,c`. Có thể ghép nhiều bước, cách nhau bằng dấu phẩy.

| Bước | Làm gì | Ghi chú |
|---|---|---|
| `search` | Tìm địa điểm theo truy vấn | Sinh `places_candidates.csv` và `places_selected.csv` |
| `reviews` | Thu thập review | Chạy lại được; chỉ một tiến trình |
| `contributors` | Thu lịch sử reviewer nghi vấn | Cần đã chạy `all` ít nhất một lần |
| `parse` | Gộp dữ liệu thô về một lược đồ, ghép timestamp, ẩn danh | |
| `clean` | Làm sạch và chuẩn hóa tiếng Việt | |
| `quality` | Bảng chất lượng 6 chiều | |
| `eda` | Bảng và hình mô tả | |
| `m1a` `m1b` `m2` `m3` | Các module chấm điểm (`m2` tính cả M2a và M2b) | Chạy `m1a` trước `m2` |
| `fusion` | Gộp điểm Fisher, chia 3 vùng, xuất `top100_suspicious.csv` (baseline) | |
| `ws` | **Phương pháp chính**: 8 luật → Dawid–Skene → `ws_scores.parquet`, `ws_lf_summary.csv` | Cần đã chạy `all` (đọc `scored_reviews.parquet`) |
| `media` `img_hash` `labeltool` | Công cụ gán nhãn theo quán (nhóm `labeltool_all`) | Mục 9 |
| `sample` | Lấy mẫu gold set, tạo file gán nhãn | Không ghi đè `annotator_A/B.csv` đã có |
| `llm_export` / `llm_import` | Xuất batch cho claude.ai / gộp kết quả | Mục 9 |
| `agreement` | Kappa, so người–Claude, hợp nhất nhãn → `gold_final.csv` | |
| `metrics` | P@K, PR-AUC, FPR, độ vững | |
| `ablation` / `sensitivity` | So sánh cấu hình / độ nhạy tham số | |
| `benchmark` | Tải ViSpamReviews v2 (cần `HF_TOKEN`) | Chỉ cho thí nghiệm lệch miền |
| `m1b_phobert` | Phiên bản PhoBERT của M1b | **Chưa chạy thử**, nên thử trên Colab có GPU |
| `demo` / `demo_labels` | Sinh dữ liệu và nhãn giả lập | Chỉ để chạy thử |

Nhóm lệnh: `all` = `parse,clean,quality,eda,m1a,m1b,m2,m3,fusion,sample`; `eval` = `agreement,metrics,ablation,sensitivity`; `labeltool_all` = `media,img_hash,labeltool`.

Đánh giá trên nhãn theo quán chạy bằng `python -m` (mục 10.2): `src.evaluation.labeltool_eval`, `src.evaluation.exp_pu_ws`, `src.evaluation.ws_paper_results`.

---

## 14. Đầu ra và mục tương ứng trong paper

Các file nằm trong `reports/` (bảng ở `tables/`, hình ở `figures/`).

| Mục paper | File |
|---|---|
| Data collection | `tables/collect_stats_playwright.csv`, `tables/coverage_by_place.csv` |
| Data quality | `tables/data_quality.csv` (6 chiều), `tables/cleaning_log.csv`, `tables/accuracy_check_sample.csv` |
| EDA | `figures/fig_eda_*.png`, `tables/eda_summary.csv` |
| Methodology (case study) | `tables/m1a_top_clusters.csv`, `tables/m2_top_bursts.csv`, `figures/fig_m2_bursts.png` |
| Thí nghiệm lệch miền M1b | `tables/m1b_benchmark.csv`, `tables/m1b_label_variants.csv`, `tables/m1b_gmaps_spotcheck.csv` |
| Annotation | `tables/agreement.csv`, `tables/llm_validation.csv`, `tables/human_vs_claude_confusion.csv` |
| Results | `tables/metrics_main.csv`, `tables/metrics_robustness.csv`, `figures/fig_pk_curve.png`, `figures/fig_pr_curve.png` |
| Ablation / sensitivity | `tables/ablation.csv`, `figures/fig_ablation.png`, `tables/sensitivity.csv` |
| Error analysis | `tables/fp_top.csv`, `tables/fn_top.csv`, `tables/top100_suspicious.csv` |
| **Phương pháp chính (WS)** | `tables/ws_lf_summary.csv` |
| **Kết quả trên nhãn theo quán** | `tables/paper_main_comparison.csv` (3 cách xử lý Uncertain), `paper_operating_points.csv`, `paper_lf_table.csv` (ablation + độ chính xác từng luật), `paper_within_stratum.csv`, `paper_per_place.csv`, `paper_data_stats.csv`, `paper_label_stats.csv`, `figures/paper_roc_pr.pdf`, `figures/paper_forest.pdf` |
| Thí nghiệm phụ | `tables/labeltool_eval_*.csv`, `tables/exp_pu_ws_compare.csv` |

---

## 15. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `Failed to create a ProcessSingleton` / `SingletonLock: File exists` | Hai Chrome dùng chung một profile (scraper đang chạy, hoặc có Chrome còn sót) | Dừng tiến trình scraper cũ (`pkill -f "run_pipeline.py --stages reviews"`), đóng Chrome dùng profile đó. Chỉ khi chắc chắn không còn tiến trình nào mới xóa `Singleton*` trong thư mục profile |
| `Không thấy thẻ review nào` / lấy 0 review | Google hiện "chế độ hạn chế" (chưa đăng nhập), hoặc selector cũ | Làm lại mục 6.1; thử lại sau ít phút; nếu vẫn lỗi xem dòng dưới |
| Tab *Bài đánh giá* không bấm được, cuộn không ra review | Google đổi giao diện | Mở Maps trong Chrome → chuột phải vào review → *Inspect* → lấy lại class của thẻ review, tên, sao, ngày, nội dung → sửa `collect.playwright.selectors` (mỗi mục có thể khai báo nhiều phương án, cách nhau bằng dấu phẩy) |
| Parse báo `timestamp RPC dùng=0` | Không bắt được gói RPC (Google đổi endpoint) | Kiểm tra `data/raw/network/<quán>/` có file không. Nếu rỗng, thêm từ khóa URL mới vào `network_url_keywords`. Thử một quán bằng `headless: false` |
| Số review lấy được thấp hơn nhiều so với Google báo | Cuộn dừng sớm, mạng chậm | Tăng `max_idle_rounds` hoặc `scroll_pause_ms`; xóa `*.done.json` của quán đó rồi chạy lại |
| `FileNotFoundError: reviews_clean.parquet` khi chạy `m1b` | Chưa chạy `parse`/`clean` | Chạy `--stages all`, hoặc `parse,clean` trước |
| Kết quả M2 gắn cờ quá nhiều ngày | Ngưỡng quá thoáng với dữ liệu thật | Giảm `m2.contamination`, tăng `m2.z_threshold` hoặc `m2.min_count`, rồi chạy lại `m2,fusion` |
| `llm_import` báo THIẾU mục | Claude bỏ sót dòng | Gửi riêng các dòng đó trong một chat mới, lưu thêm file vào `ket_qua/`, chạy lại `llm_import` |
| Agreement báo "KHÔNG dùng nhãn Claude" | Chưa đạt một trong năm điều kiện | Đọc dòng "chưa đạt:" trong log hoặc `llm_validation.csv`; thường cần người gán thêm hoặc chỉnh hướng dẫn rồi cho Claude gán lại |
| `hash` của reviewer thay đổi giữa các lần chạy | Đã đổi `REVIEWER_SALT` | Đặt lại salt cũ; nếu mất, phải `parse` lại toàn bộ dữ liệu thô và gán nhãn lại |
| Lỗi ở `test_scraper_mock` khi chạy pytest | Scraper thật đang chạy, giữ khóa profile | Chạy `pytest tests/test_units.py` |

Nhật ký chạy nằm ở `logs/pipeline.log` và `logs/reviews_run.log`.

---

## 16. Hạn chế đã biết

1. **Scraper phụ thuộc giao diện Google.** Selector và endpoint RPC có thể hỏng bất cứ lúc nào (mục 15). Scraper không phải sản phẩm ổn định lâu dài.
2. **Không có nhãn "sự thật".** Gold set là phán đoán của người và Claude dựa trên bằng chứng nhìn thấy được; nhãn `Genuine` nghĩa là "không thấy dấu hiệu". Cỡ mẫu nhỏ nên mọi chỉ số đều kèm khoảng tin cậy rộng.
3. **Thiên lệch chọn mẫu.** Nhóm A được chọn theo chính điểm của hệ thống, nên P@K của từng module riêng lẻ có thể bị thổi phồng. Tỷ lệ spam nền chỉ lấy từ Nhóm B; ablation chỉ mang tính so sánh tương đối. Phải nêu trong Limitations.
4. **Nhãn Claude cần kiểm chứng.** Claude chỉ thay thế người khi đạt các điều kiện ở mục 9, và kết quả luôn được báo cáo tách theo nguồn nhãn.
5. **Review chỉ chấm sao (~37%)** có rất ít bằng chứng; chỉ M2a, M3 chấm được (M1a, M1b, M2b cần có chữ).
6. **Burst có thể là thật** (khai trương, sự kiện, báo chí), gây báo nhầm ở M2.
7. **Chưa xử lý ảnh.** Nhiều review có ảnh giống nhau nhưng nội dung khác nhau cũng có thể là spam; module M1c (so ảnh bằng perceptual hash) là phần mở rộng làm sau. Link ảnh đã có sẵn trong các gói RPC đã lưu.
8. **Fusion Fisher đã bị thay** bằng weak supervision sau khi có nhãn thật (mục 10.2); chỉ còn làm baseline.
9. **Ngân sách 5%/10% là hạn ngạch**, không phải ngưỡng đã hiệu chỉnh theo xác suất spam: nhóm chọn để giới hạn khối lượng xem tay.
10. **`m1b_phobert.py` chưa được chạy thử** (không tải được model trong môi trường phát triển).
11. **Không so sánh với dữ liệu nước ngoài.** Bộ GMR-PL (Google Maps, tiếng Ba Lan) và các bộ Yelp bị loại khỏi đánh giá vì khác ngôn ngữ/thị trường.
12. **Luật trùng với hướng dẫn gán nhãn.** `LF_thin` gần giống cờ f4, `LF_burst` gần giống cờ f6 của rubric, nên một phần kết quả có thể chỉ là "đo lại rubric". Checklist dùng cùng thông tin vẫn thua có ý nghĩa, và trong nhóm review người gán đã bật cờ đợt dồn, AUC của WS vẫn 0,961; nhưng cần nhãn độc lập để kết luận chắc.
13. **Tập nhãn nhỏ:** 4 quán, 79/85 spam đến từ 2 quán; mỗi quán một người gán nên chưa đo được κ; người gán có xem bằng chứng tính từ cùng các tín hiệu. 98,4% nhãn trùng với luật tham khảo của rubric.
14. **Kết quả mang tính khám phá:** bộ 8 luật được viết sau một phân tích sơ bộ trên chính tập nhãn này. Cần gán nhãn quán mới (2 người, không xem gợi ý cờ) để xác nhận.
15. **Dawid–Skene quá tự tin** (giả định độc lập bị vi phạm): P(spam) bão hòa gần 1 với nhiều review, chỉ dùng để xếp hạng. Trên cùng 8 luật, nó chưa hơn bỏ phiếu đa số về AUC.

---

## 17. Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| [`docs/huong_dan_gan_nhan.md`](docs/huong_dan_gan_nhan.md) | Tiêu chí gán nhãn đầy đủ cho người và Claude |

**Trích dẫn chính:** Dawid & Skene (1979), *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm* (label model); Ratner et al. (2017), *Snorkel: Rapid Training Data Creation with Weak Supervision* (weak supervision); Li et al. (2014), *Spotting Fake Reviews using Positive-Unlabeled Learning* (baseline PU); Gryka & Janicki (2023), *Detecting Fake Reviews in Google Maps—A Case Study*, Applied Sciences 13(10), 6331 (bộ dữ liệu Google Maps có nhãn duy nhất, tiếng Ba Lan); Van Dinh & Luu (2024), *Metadata Integration for Spam Reviews Detection on Vietnamese E-Commerce Websites* (ViSpamReviews v2).
