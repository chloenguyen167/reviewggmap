# PLAN TRIỂN KHAI (v2) — Phát hiện Đánh giá Giả mạo trên Google Maps tại Việt Nam

> Phương pháp đa tín hiệu: Tương đồng Văn bản · Hành vi Tài khoản · Bất thường Thời gian
> Người thực hiện: Nguyễn Phi Yến – 250104059 · Môn: Kỹ thuật Xử lý Dữ liệu

Tài liệu này là bản kế hoạch kỹ thuật. **v2** (30/09/2026): bổ sung thu thập bằng Playwright và các điều chỉnh phát hiện khi chạy thử — đánh dấu "(v2)". Mỗi phần nêu rõ: làm gì, input/output, quyết định thiết kế, và tiêu chí "xong".

---

## 0. ĐIỀU CHỈNH QUAN TRỌNG SO VỚI ĐỀ CƯƠNG

| Vấn đề | Chi tiết | Hướng xử lý |
|---|---|---|
| **Google Places API chỉ trả tối đa 5 review/địa điểm**, không phân trang, không sắp xếp | Với 5 review thì không dựng được chuỗi thời gian → Module Temporal Burst không chạy được | Đổi nguồn chính sang **SerpApi – Google Maps Reviews API** (có `next_page_token` phân trang, `sort_by=newestFirst`, `hl=vi`). Places API chỉ dùng để **tìm place & lấy metadata** |
| Blocking theo `place_id` bỏ sót review trùng **giữa các địa điểm khác nhau** | Mà đây lại là dấu hiệu spam mạnh nhất (1 mẫu văn bản rải cho nhiều quán) | Dùng **MinHash-LSH** làm bước Blocking toàn corpus (đúng tinh thần Blocking ở Chương 2), thay vì block theo place |
| ViSpamReviews v2 là bài toán **phân loại**, còn Module 1 ban đầu là **so khớp tương đồng** → không đo trực tiếp được | Không khớp mục tiêu kiểm định | Tách Module 1 thành **1a Near-duplicate** (không giám sát) + **1b Content-spam classifier** (huấn luyện trên ViSpamReviews, chuyển miền sang GMaps). ViSpamReviews dùng để đánh giá **1b** |

| **(v2) Thu thập bằng Playwright** thay vì trả phí SerpApi | Miễn phí, lấy được toàn bộ review; nhưng giao diện web chỉ hiện **ngày tương đối** ("3 tháng trước") → độ chính xác ngày giảm dần theo tuổi review | Playwright là **nguồn chính**; SerpApi giữ làm phương án thay thế. Scraper **bắt thêm các gói tin mạng** của trang (RPC `listugcposts`) để lấy timestamp tuyệt đối, đối chiếu với ngày tương đối, và chỉ giữ timestamp nào khớp (xem A3-PW) |
| **(v2) MAD = 0 trên chuỗi thưa** | Quán nhỏ thường 0–2 review/ngày → median = MAD = 0 → z-score chia cho ~0, ngày nào có 1 review cũng thành "burst" | Thêm sàn độ lệch σ_floor = √max(median, 1) (xấp xỉ độ lệch chuẩn Poisson) và yêu cầu tối thiểu `n ≥ 3` review/ngày mới gắn cờ (C3) |
| **(v2) Isolation Forest không phân biệt hướng** | Một Local Guide có 5.000 review cũng "bất thường" dù không đáng ngờ | **Cắt một phía**: mỗi đặc trưng được đổi chiều sao cho lớn = đáng ngờ, rồi mọi giá trị dưới median bị nâng lên median → IF chỉ cô lập được phía đáng ngờ (C4) |

| **(v2) Gộp điểm bằng trung bình làm loãng tín hiệu** | Mỗi kiểu spam thường chỉ nổi rõ ở 1–2 module; lấy trung bình với các module "bình thường" kéo điểm xuống (trên dữ liệu demo: AP 0,10 so với 0,46) | Mặc định gộp bằng **phương pháp Fisher** (gộp p-value, pᵢ = 1 − sᵢ). Ablation báo cáo cả 3 cách gộp (trung bình / Fisher / max) để kiểm chứng lại trên gold set thật (C5, D5) |
| **(v2) Câu sáo rỗng tự nhiên bị M1a coi là trùng** | Ngoài đời có những câu khen chung chung lặp ở nhiều quán, rải đều qua nhiều năm | M1a đo **mức bất ngờ Poisson** của số bản trùng rơi trong cùng cửa sổ 14 ngày so với kỳ vọng ngẫu nhiên → chiến dịch dồn vào vài ngày mới có điểm cao (C1) |

➡️ Cần cập nhật lại mục 4.1 (Data) và 4.2 (Module 1) trong đề cương/paper cho khớp.

---

## 1. TỔNG QUAN KIẾN TRÚC

```
[Playwright: tìm địa điểm trên Maps]  hoặc  [Places API: Text Search]
            │                      ──► raw/places/places_selected.csv
            ▼
[Playwright: cuộn tab Bài đánh giá + bắt RPC]  hoặc  [SerpApi: google_maps_reviews]
            │                      ──► raw/reviews/*.jsonl, raw/network/  (append-only)
            ▼
[Parse: gộp 2 nguồn về 1 schema, ghép timestamp RPC, ẩn danh reviewer]
            │                      ──► interim/reviews_parsed.parquet, interim/places.parquet
            ▼
[Clean & Normalize]  ──► interim/reviews_clean.parquet
   - dedup kỹ thuật, parse ngày, chuẩn hóa Unicode/teencode, tách từ
   - hash reviewer_id (ẩn danh)
            │
            ├──► M1a Near-duplicate (MinHash-LSH + Jaccard/cosine)
            ├──► M1b Content-spam classifier (TF-IDF+LR baseline / PhoBERT)
            ├──► M2  Temporal Burst (robust z-score + IsolationForest theo place-day)
            └──► M3  Behavioral (đặc trưng tài khoản)
                        │
                        ▼
            [Fusion: chuẩn hóa percentile → Suspicion Score → 3 vùng quyết định]
                        │
                        ▼
            processed/scored_reviews.parquet
                        │
                        ▼
[Sampling gold set] → [Gán nhãn thủ công 2 người] → [Evaluation: P@K, FPR, Kappa, Ablation]
```

**Nguyên tắc chung**
- Raw data **bất biến** (immutable) — mọi bước sau đọc từ raw, ghi ra thư mục mới → chạy lại được toàn bộ.
- Mọi tham số (ngưỡng, trọng số, cửa sổ) để trong `config.yaml`, không hard-code.
- Mỗi module xuất một cột điểm riêng `s_dup`, `s_content`, `s_burst`, `s_behav` ∈ [0,1] + cột giải thích (`reason_*`) để phục vụ gán nhãn và viết paper.
- Seed cố định (`random_state=42`) cho mọi bước ngẫu nhiên.

---

## 2. CẤU TRÚC THƯ MỤC & MÔI TRƯỜNG

```
gmaps-fake-review-vn/
├── config.yaml               # mọi tham số
├── config.demo.yaml          # kế thừa config.yaml, dữ liệu demo nằm ở demo/
├── run_pipeline.py           # điều phối: --stages search,reviews,all,eval ...
├── requirements.txt  .env.example  README.md
├── resources/teencode_vi.txt
├── docs/PLAN.md
├── src/
│   ├── collect/    scrape_playwright.py  serpapi_client.py  parse_raw.py
│   │               relative_date.py  rpc_timestamps.py  place_url.py
│   ├── preprocess/ normalize_vi.py  clean.py  quality.py  eda.py
│   ├── modules/    m1a_duplicate.py  m1b_content.py  m1b_phobert.py  m2_temporal.py  m3_behavior.py
│   ├── fusion/     score.py
│   ├── evaluation/ sample_gold.py  agreement.py  metrics.py  ablation.py
│   ├── synthetic/  make_demo_data.py  simulate_annotations.py
│   └── utils/      config.py  common.py  plotstyle.py
├── tests/          test_units.py  test_scraper_mock.py  mock_gmaps.py
├── data/           (tạo khi chạy) raw/ interim/ processed/ labels/ benchmark/ models/
└── reports/        (tạo khi chạy) figures/ tables/
```

**Thư viện dự kiến**: `requests`, `pandas`, `pyarrow`, `pyyaml`, `python-dotenv`, `tenacity` (retry), `underthesea` hoặc `pyvi` (tách từ tiếng Việt), `datasketch` (MinHash-LSH), `scikit-learn`, `scipy`, `sentence-transformers`, `transformers` + `torch` (PhoBERT, tùy chọn), `matplotlib`/`seaborn`.

> Việc 0.1: kiểm tra `pip install` được hết trên máy/Colab; PhoBERT yêu cầu input đã tách từ (word-segmented) — chốt công cụ tách từ ngay từ đầu để dùng thống nhất.

---

## 3. PHẦN A — THU THẬP DỮ LIỆU (Data Ingestion)

### A0. (v2) Chọn công cụ thu thập

| Tiêu chí | **Playwright (tự scrape)** — nguồn chính | SerpApi — thay thế | Places API chính thức |
|---|---|---|---|
| Chi phí | Miễn phí | Gói free có giới hạn lượt/tháng | Có free tier |
| Số review/địa điểm | Toàn bộ (cuộn tới hết) | Toàn bộ (phân trang) | **Tối đa 5** → loại |
| Ngày review | Tương đối trên giao diện; tuyệt đối nếu bắt được gói RPC | Tuyệt đối (`iso_date`) | Tuyệt đối |
| Độ bền | Dễ gãy khi Google đổi giao diện → selector để trong `config.yaml` | Ổn định | Ổn định |
| Pháp lý | Vùng xám điều khoản Google → quy mô nhỏ, chậm, học thuật, nêu trong Ethics | Bên thứ ba chịu | Rõ ràng |

**Hạn chế cần nêu trong paper khi dùng Playwright**
1. Chỉ chạy được trên máy có truy cập Google Maps (máy cá nhân), không chạy trên server bị chặn.
2. Ngày tương đối: review càng cũ thì càng mờ ("1 năm trước" ≈ sai số 365 ngày). Module Burst chỉ dùng review có độ chính xác ngày ≤ 1 ngày (hoặc ≤ 7 ngày nếu chạy theo tuần), review còn lại có `s_burst = NaN`, được xử lý ở Fusion.
3. Google có thể hiện chế độ "xem giới hạn" hoặc trang đồng ý cookie cho người chưa đăng nhập → hỗ trợ `user_data_dir` (dùng profile trình duyệt đã đăng nhập một tài khoản phụ).

### A3-PW. (v2) Thu thập bằng Playwright (`scrape_playwright.py`)

1. **Tìm địa điểm** (không cần API key): mở `google.com/maps/search/<truy vấn>?hl=vi`, cuộn danh sách kết quả, lấy tên, URL, số sao, số review. Từ URL tách `place_id` (`!19s…`), `data_id` (`!1s0x…:0x…`), vĩ độ/kinh độ (`!3d…!4d…`). Lọc theo tiêu chí A1 → `places_selected.csv`. Bạn có thể sửa tay file này (thêm/bớt quán) trước khi lấy review.
2. **Lấy review** cho từng URL: mở trang → bấm tab "Bài đánh giá" → sắp xếp "Mới nhất" → tìm khung cuộn (đi ngược từ thẻ review đầu tiên tới phần tử cha có thanh cuộn) → cuộn tới khi đủ `max_reviews` hoặc N vòng liên tiếp không có review mới → bấm mọi nút "Thêm" → trích dữ liệu từng thẻ bằng JS.
3. Trường lấy từ DOM: `review_id` (`data-review-id`), tên, link hồ sơ → `contributor_id`, dòng meta ("Local Guide · 132 bài đánh giá · 1.024 ảnh"), số sao (`aria-label`), ngày tương đối, nội dung, phản hồi của chủ quán, số ảnh.
4. **Bắt gói tin mạng**: lưu body mọi response có URL chứa `listugcposts` vào `raw/network/<place>/`. Bộ parser duyệt cây JSON, tìm cặp (review_id, timestamp dạng micro-giây). Timestamp chỉ được dùng nếu **nằm trong khoảng** suy ra từ ngày tương đối của chính review đó; nếu lệch thì bỏ và giữ ngày tương đối. Cách này an toàn kể cả khi cấu trúc RPC thay đổi.
5. **Lịch sự với máy chủ**: nghỉ ngẫu nhiên 0,8–1,6 giây mỗi lần cuộn, 20–45 giây giữa hai địa điểm, chạy tuần tự, mỗi địa điểm ghi ra file riêng (checkpoint) → chạy lại sẽ bỏ qua quán đã xong.
6. **Lịch sử reviewer** (tùy chọn): mở `google.com/maps/contrib/<id>/reviews` cho ~300 reviewer nghi vấn nhất, dùng lại bộ trích xuất thẻ review.

### A1. Phạm vi mẫu (sampling frame)

| Tham số | Giá trị đề xuất | Lý do |
|---|---|---|
| Thành phố | TP.HCM (+ Hà Nội nếu còn quota) | Gần gũi, dễ kiểm chứng thực địa |
| Ngành | Nhà hàng/quán ăn, quán cà phê, spa/thẩm mỹ, khách sạn nhỏ | Các ngành hay bị "chạy review" theo báo chí |
| Số địa điểm | **40–60** | Đủ đa dạng, vừa quota API |
| Tiêu chí chọn | 100 ≤ tổng số review ≤ 3.000; trộn điểm TB cao (≥4.7) và thấp | Có đủ dữ liệu để dựng chuỗi thời gian; điểm TB quá cao bất thường là "ứng viên nghi vấn" tự nhiên |
| Khung thời gian | Toàn bộ lịch sử có thể lấy, ưu tiên 24 tháng gần nhất | Đủ dài để có baseline cho burst |

Nên chọn **có chủ đích một phần** (ví dụ vài quán mới mở có rất nhiều review 5★ trong thời gian ngắn) và **ngẫu nhiên một phần** (lấy từ kết quả tìm kiếm theo từ khóa ngành + quận) → tránh toàn bộ dữ liệu bị thiên lệch về phía "nghi vấn".

### A2. Bước 1 — Tìm địa điểm (`places_search.py`)
- Google Places API **Text Search** với truy vấn kiểu `"quán cà phê quận 1"`, `languageCode=vi`, `regionCode=VN`.
- Lưu: `place_id`, `name`, `types`, `rating`, `user_rating_count`, `lat`, `lng`, `formatted_address`.
- Lọc theo tiêu chí A1 → `data/raw/places/places_selected.jsonl`.
- Lấy thêm `data_id` của SerpApi (qua engine `google_maps` tìm theo tên/place_id) vì Reviews API của SerpApi dùng `data_id`/`place_id`.

### A3. Bước 2 — Lấy toàn bộ review (`reviews_serpapi.py`)
- Engine `google_maps_reviews`, tham số: `hl=vi`, `sort_by=newestFirst`, phân trang qua `next_page_token` cho đến hết.
- Mỗi trang ghi **ngay** một dòng jsonl (checkpoint) kèm `place_id`, `page_idx`, `fetched_at` → mất mạng giữa chừng thì chạy tiếp từ token cuối, không tải lại từ đầu.
- **Bẫy đã biết**: có báo cáo công khai (issue #4378, repo `serpapi/public-roadmap`) rằng ở trang sâu, API đôi khi trả về trang rỗng với trạng thái "Success" và không có token tiếp → client tưởng đã hết. **Xử lý**: nếu số review đã lấy < `place_info.reviews` đáng kể và trang rỗng → gửi lại token cũ với `no_cache=true` (tối đa 3 lần) trước khi kết luận hết.
- Ghi lại `expected_total` vs `fetched_total` cho từng place → báo cáo **độ phủ (coverage)** trong paper (đây là chỉ số Completeness của Chương 1).

### A4. Bước 3 — Lịch sử reviewer (tùy chọn, chỉ cho top nghi vấn) (`contributors_serpapi.py`)
- Engine `google_maps_contributor_reviews` với `contributor_id` → các review khác của cùng tài khoản (địa điểm, thời gian, số sao).
- **Chỉ gọi cho ~300–500 reviewer có điểm M1/M2 cao nhất** (không gọi cho tất cả — tốn quota). Dùng cho đặc trưng "phân tán địa lý" của M3.

### A5. HTTP client dùng chung (`http_client.py`) — áp dụng đúng Chương 2
- `timeout=10s`; retry bằng `tenacity`: **exponential backoff** 2s → 4s → 8s (+ jitter), tối đa 5 lần.
- HTTP 429: đọc header `Retry-After` nếu có, nếu không thì backoff.
- HTTP 4xx khác (401/403): **dừng ngay**, không retry (lỗi key/quota).
- Log mọi request (URL không kèm key, status, thời gian) vào `logs/collect.log`.
- Đếm số lượt gọi API mỗi lần chạy, dừng khi chạm `max_calls` trong config.

### A6. Ước lượng quota & chi phí
- SerpApi trả ~10–20 review/trang. Ví dụ 50 địa điểm × trung bình 600 review ÷ 20 ≈ **1.500 lượt gọi** + ~400 lượt cho contributor.
- Gói miễn phí của SerpApi có giới hạn lượt gọi/tháng (kiểm tra con số hiện tại trên trang giá trước khi bắt đầu). Nếu không đủ: (1) giảm số địa điểm, (2) giới hạn 24 tháng gần nhất, (3) phương án dự phòng bên dưới.
- **Phương án dự phòng**: tự viết scraper bằng Playwright cho một số ít địa điểm (headless, cuộn danh sách review, tốc độ chậm có delay). Rủi ro: dễ gãy khi Google đổi giao diện và vùng xám điều khoản → chỉ dùng nếu SerpApi không khả thi, và nêu rõ trong mục Ethics.

### A7. Schema dữ liệu thô chuẩn hóa (sau khi parse)

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `review_id` | str | id của review từ API; nếu thiếu → hash(place_id + reviewer + date + text) |
| `place_id` | str | |
| `reviewer_hash` | str | SHA-256(contributor_id + salt) — **không lưu id thật trong file phân tích** |
| `reviewer_n_reviews` | int | tổng số review của tài khoản (nếu API trả về) |
| `reviewer_local_guide` | bool | nếu API trả về |
| `rating` | int 1–5 | |
| `text` | str | có thể rỗng (chỉ chấm sao) |
| `review_date` | datetime (UTC+7) | từ trường ngày ISO của API |
| `has_owner_response` | bool | |
| `n_photos` | int | |
| `lang` | str | nhận diện ngôn ngữ (vi/en/khác) |

> Việc A7.1 (Phase 0): gọi thử 1 place, **in ra JSON thật** và xác nhận trường nào có/không có (đặc biệt: ngày tuyệt đối, số review của reviewer, cờ Local Guide). Schema cuối cùng chốt theo dữ liệu thật, không theo giả định.

**Tiêu chí "xong" Phần A**: ≥ 40 địa điểm, ≥ 15.000 review, coverage trung bình ≥ 80%, log không còn lỗi chưa xử lý.

---

## 4. PHẦN B — LÀM SẠCH & CHUẨN HÓA (Preprocessing)

### B1. Kiểm tra chất lượng theo 6 chiều (Chương 1) — xuất bảng vào paper

| Chiều | Kiểm tra cụ thể |
|---|---|
| Completeness | % review rỗng text; % thiếu ngày; coverage theo place |
| Uniqueness | trùng `review_id`; trùng hoàn toàn (place, reviewer, text, date) do phân trang lặp |
| Validity | rating ∉ {1..5}; ngày trong tương lai/trước khi place tồn tại |
| Consistency | rating trung bình tính lại vs `rating` do Places API báo |
| Timeliness | phân bố ngày review, % review 24 tháng gần nhất |
| Accuracy | đối chiếu thủ công 20 review ngẫu nhiên với giao diện Google Maps |

### B2. Làm sạch (`clean.py`)
1. Bỏ trùng kỹ thuật (do pagination) — **khác** với "trùng nội dung" (đó là tín hiệu spam, giữ lại cho M1a).
2. Chuẩn hóa thời gian về `Asia/Ho_Chi_Minh`, tạo cột `review_day`.
3. Nhận diện ngôn ngữ; giữ cả tiếng Việt và tiếng Anh nhưng M1b chỉ áp dụng cho `lang=vi`.
4. Review chỉ có sao (không text): giữ lại cho M2/M3, bỏ qua M1a/M1b (điểm = NaN → xử lý ở Fusion).

### B3. Chuẩn hóa văn bản tiếng Việt (`normalize_vi.py`)
- Unicode NFC (tránh 2 kiểu gõ dấu tổ hợp/dựng sẵn làm sai so khớp).
- lowercase, bỏ URL/emoji (nhưng **đếm số URL, số điện thoại, emoji** thành đặc trưng riêng trước khi xóa — URL/SĐT là dấu hiệu spam quảng cáo).
- Chuẩn hóa teencode/viết tắt phổ biến bằng từ điển nhỏ tự xây (k→không, dc/đc→được, ko→không, ...), ghi nhận là hạn chế.
- Hai phiên bản văn bản: `text_norm` (cho MinHash, TF-IDF) và `text_seg` (đã tách từ, cho PhoBERT).

### B4. EDA (`eda.py` / `01_eda.ipynb`) — hình đưa vào paper
- Phân bố rating toàn corpus và theo ngành (thường lệch về 5★ → J-shape).
- Phân bố độ dài review (boxplot, log-scale).
- Chuỗi số review/ngày của vài place tiêu biểu (minh họa burst bằng mắt).
- Phân bố `reviewer_n_reviews` (bao nhiêu % tài khoản chỉ có 1 review).

**Tiêu chí "xong" Phần B**: `reviews_clean.parquet` + bảng Data Quality + 4–5 hình EDA.

---

## 5. PHẦN C — PHẦN CHÍNH: 3 MODULE TÍN HIỆU + FUSION

### C1. Module 1a — Near-duplicate (`m1a_duplicate.py`) — không giám sát

**Ý tưởng**: 1 mẫu văn bản bị dán/biến tấu nhẹ cho nhiều quán hoặc nhiều tài khoản.

1. **Blocking**: MinHash (128 permutations) trên shingle 3-từ của `text_norm`; LSH ngưỡng ~0.5 → chỉ so sánh các cặp rơi cùng bucket (tránh O(N²), đúng bài toán bùng nổ tổ hợp Chương 2).
2. **Matching** trên cặp ứng viên: Jaccard thật + cosine của sentence embedding tiếng Việt (bắt được paraphrase mà Jaccard bỏ sót).
3. Chỉ xét review có ≥ 8 từ (review "Ngon", "Tuyệt vời" trùng nhau là tự nhiên, không phải spam).
4. Đặc trưng xuất ra:
   - `dup_max_sim`: độ tương đồng lớn nhất với review khác
   - `dup_cross_place`: số **địa điểm khác** có review gần trùng (trọng số cao nhất)
   - `dup_cross_user`: số tài khoản khác đăng nội dung gần trùng
5. (v2) **Phối hợp thời gian**: `coord` = số bản trùng đăng trong vòng ±14 ngày; kỳ vọng ngẫu nhiên λ = (số bản trùng) × 28/(khoảng thời gian corpus); `surprise = −log10 P(Poisson(λ) ≥ coord)`.
6. `s_dup_raw = max_sim × (1 + ½·log(1+cross_place)) × (1 + ½·log(1+cross_user)) × (1 + surprise)` → percentile rank.

**Kiểm tra nhanh**: in 20 cụm trùng lớn nhất → đọc bằng mắt xem có hợp lý không; chỉnh ngưỡng LSH và độ dài tối thiểu.

### C2. Module 1b — Content-spam classifier (`m1b_content.py`) — có giám sát, chuyển miền

1. Dữ liệu: **ViSpamReviews v2** (Van Dinh & Luu, 2024). Dataset là repo "gated" trên HuggingFace (`sonlam1102/vispamdetection2`): đăng nhập, bấm đồng ý chia sẻ thông tin để tải; nếu cần thì liên hệ tác giả (email trên repo GitHub `sonlam1102/vispamdetection`) — **làm trong Phase 0**. Tên cột chưa được công bố nên bộ nạp dữ liệu tự dò cột văn bản/nhãn, có thể ghi đè trong `config.yaml`.
2. Mô hình theo 2 bậc:
   - Baseline: TF-IDF (word 1–2gram + char 3–5gram) + Logistic Regression (nhanh, giải thích được).
   - Nâng cao (nếu có GPU/Colab): fine-tune PhoBERT-base, vài epoch.
3. Train/val/test theo **split gốc** của dataset (để so sánh được với số liệu công bố).
4. Áp dụng sang review Google Maps → `s_content` = xác suất spam.
5. **Lưu ý chuyển miền** (e-commerce → location review): ghi nhận là hạn chế; kiểm tra thủ công 30 dự đoán trên GMaps xem mô hình có "hiểu nhầm" kiểu review địa điểm không.

### C3. Module 2 — Temporal Burst (`m2_temporal.py`)

**Đơn vị phân tích**: (place, ngày). Từ đó gán điểm ngược về từng review.

1. Dựng chuỗi theo ngày cho mỗi place (điền 0 cho ngày không có review):
   - `n_reviews`, `mean_rating`, `share_5star`, `share_1star`, `share_new_accounts` (tỷ lệ reviewer có ≤ 3 review), `share_no_text`.
2. **Robust z-score** cho `n_reviews` so với cửa sổ trượt 28 ngày *trước đó* (median & MAD, không dùng mean/std vì phân phối lệch phải và bị chính outlier kéo — đúng cảnh báo ở Chương 3/4):
   `z = (x_t − median_{t−28..t−1}) / max(1.4826 · MAD, √max(median, 1))`
   (v2: sàn √max(median,1) tránh chia cho 0 trên chuỗi thưa; chỉ gắn cờ khi `x_t ≥ 3`)
3. **Isolation Forest** trên vector đặc trưng ngày (5 đặc trưng trên) → bắt được ngày bất thường đa biến (ví dụ số lượng không quá lớn nhưng toàn 5★ từ tài khoản mới), `contamination` thử {0.01, 0.02, 0.05}.
4. Gộp: ngày bất thường nếu `z > 3` **hoặc** IF gắn cờ. Mở rộng ±1 ngày (chiến dịch thường kéo dài 2–3 ngày).
5. Review nhận `s_burst` = điểm bất thường của ngày nó thuộc về, **nhân thêm** mức "đồng hướng" (review 5★ trong một đợt burst 5★ bị nghi hơn review 2★ trong cùng đợt).
6. Xử lý place ít dữ liệu: nếu place có < 60 ngày có review → bỏ qua z-score, chỉ dùng IF toàn cục.

**Hình cho paper**: chuỗi thời gian 2–3 place, tô đỏ ngày burst, kèm phân bố sao trong đợt burst.

### C4. Module 3 — Behavioral (`m3_behavior.py`)

| Đặc trưng | Nguồn | Trực giác |
|---|---|---|
| `n_reviews_account` (log) | trường của API | tài khoản chỉ có 1–3 review đáng ngờ hơn |
| `is_local_guide` | trường của API | Local Guide lâu năm ít khả năng là tài khoản ảo |
| `extreme_rating` | rating ∈ {1,5} | spam thường cực đoan |
| `rating_deviation` | \|rating − điểm TB của place\| | |
| `text_len_bucket`, `no_text` | text | 5★ không chữ từ tài khoản mới |
| `n_url_phone` | B3 | quảng cáo chèn link/SĐT |
| `geo_dispersion_km` *(chỉ top nghi vấn)* | A4 + tọa độ place | tổng quãng đường giữa các địa điểm tài khoản review trong 24h |
| `same_day_multi_place` *(chỉ top nghi vấn)* | A4 | số địa điểm khác nhau tài khoản review trong cùng ngày |

Tính điểm: vì không có nhãn, dùng **Isolation Forest không giám sát** trên các đặc trưng trên → `s_behav`. (Nếu sau khi gán nhãn thấy có ích, thử thêm hướng có giám sát nhẹ ở Phần D.)

(v2) Trước khi đưa vào IF: đổi chiều mọi đặc trưng để "lớn = đáng ngờ" (ví dụ `newness = −log(1 + n_reviews)`), rồi **cắt một phía**: giá trị nhỏ hơn median bị nâng lên bằng median. Nhờ vậy IF chỉ cô lập được các điểm lệch về phía đáng ngờ. Bổ sung đặc trưng **trong corpus** không cần gọi thêm API: số quán khác nhau cùng reviewer đánh giá trong cùng ngày, và khoảng cách địa lý lớn nhất giữa các quán đó (dùng tọa độ lấy từ URL địa điểm).

### C5. Fusion (`score.py`)

1. Chuẩn hóa mỗi điểm về [0,1] bằng **percentile rank** (tránh module có thang đo lớn lấn át).
2. Điểm thiếu (NaN, ví dụ review không có text → không có `s_dup`, `s_content`) → **tái chuẩn hóa trọng số** trên các module có mặt, không điền 0.
3. (v2) Ba cách gộp, chọn bằng `fusion.method`:
   - `mean`: `Σ w_i · s_i / Σ w_i` (các module có mặt)
   - `fisher` (**mặc định**): `X = −2 Σ w_i · ln(1 − s_i) ~ χ²(2k)` → `p_gộp` → điểm `x/(1+x)` với `x = −log10 p_gộp`
   - `max`: điểm lớn nhất
   - Mặc định **trọng số bằng nhau** (không cần nhãn).
   - Phương án 2 (so sánh ở D): Logistic Regression học trọng số trên gold set bằng cross-validation 5-fold — báo cáo cả hai, **không** tinh chỉnh trên chính tập dùng để báo cáo kết quả.
4. Ba vùng quyết định (giống cơ chế 3 vùng ở bài Entity Resolution):
   - **Auto-flag**: Suspicion ≥ θ_high
   - **Review**: θ_low ≤ Suspicion < θ_high
   - **An toàn**: < θ_low
   - θ ban đầu theo percentile (ví dụ top 2% / top 10%), sau đó hiệu chỉnh theo đường cong P@K ở Phần D.
5. Xuất `scored_reviews.parquet` gồm tất cả điểm con, điểm tổng, vùng quyết định và `reasons` (chuỗi mô tả ngắn: "trùng nội dung với 4 quán khác; trong đợt burst 12/03; tài khoản 1 review").

**Tiêu chí "xong" Phần C**: chạy end-to-end bằng `python run_pipeline.py` không lỗi; đọc tay top 30 review nghi vấn thấy có lý do rõ ràng.

---

## 6. PHẦN D — ĐÁNH GIÁ (Evaluation)

### D1. Đánh giá Module 1b trên benchmark ViSpamReviews v2
- Metrics: Accuracy, Macro-F1, Precision/Recall lớp spam, confusion matrix.
- So với số liệu công bố của bài gốc (để chứng minh pipeline xử lý tiếng Việt đúng), không cần vượt SOTA.

### D2. Tạo gold set (`sample_gold.py`) — chi tiết như đã thống nhất
- **Nhóm A** (stratified theo module): top 50–70 theo **mỗi** `s_dup`, `s_burst`, `s_behav`, `Suspicion` → loại trùng → ~150–200.
- **Nhóm B** (control): ~50–100 review lấy ngẫu nhiên đều từ toàn corpus.
- Trộn ngẫu nhiên thứ tự, **ẩn** mọi điểm số và nhóm → xuất `gold_sample.csv` cho annotator.
- Kèm link Google Maps của place (để annotator xem bối cảnh), nhưng **không** kèm tên reviewer thật trong file chung.

### D3. Gán nhãn (`agreement.py`)
- 2 annotator độc lập, rubric 6 cờ + kết luận Genuine / Uncertain / Spam.
- **Pilot 30 review trước** → tính Cohen's Kappa → nếu < 0.6 thì họp chỉnh rubric rồi mới gán phần còn lại.
- Báo cáo Kappa trước khi hòa giải; bất đồng → thảo luận / người thứ 3.
- Kết quả: `gold_final.csv`.

### D4. Metrics chính (`metrics.py`)

| Chỉ số | Cách tính | Ý nghĩa |
|---|---|---|
| **Precision@K** | trong top-K theo Suspicion (chỉ xét review có trong gold set, bỏ Uncertain), tỷ lệ Spam; K = 25, 50, 100, 150 | Hệ thống gắn cờ có đúng không |
| **Đường cong P@K** | vẽ theo K | thấy độ chính xác giảm dần khi hạ ngưỡng |
| **Base rate** | % Spam trong Nhóm B | tỷ lệ spam nền ước lượng |
| **FPR trên Nhóm B** | % Genuine ở Nhóm B nhưng rơi vào vùng Auto-flag | hệ thống có báo động giả nhiều không |
| **Recall ước lượng** | (số Spam bắt được trong vùng flag) ÷ (base rate × kích thước corpus) | **ghi rõ là ước lượng** |
| **PR-AUC** | trên toàn gold set (Spam=1, Genuine=0) | chỉ số tổng hợp không phụ thuộc ngưỡng |
| **Khoảng tin cậy 95%** | bootstrap 1.000 lần trên gold set | cỡ mẫu nhỏ → bắt buộc báo cáo độ bất định |

### D5. Ablation (`ablation.py`)
So sánh cùng bộ metrics cho các cấu hình:
1. Chỉ M1a · 2. Chỉ M1b · 3. Chỉ M2 · 4. Chỉ M3
5. M1 (a+b) · 6. M1 + M2 · 7. **Full (M1 + M2 + M3)**, trọng số bằng nhau — (v2) báo cáo riêng 3 cách gộp: 7a trung bình, 7b Fisher, 7c max
8. Full, trọng số học bằng LR (5-fold CV, dự đoán out-of-fold)

➡️ Kết quả kỳ vọng để kiểm chứng: cấu hình 7/8 > từng module riêng lẻ. **Nếu không đúng như kỳ vọng thì vẫn báo cáo trung thực** và phân tích lý do (đó cũng là kết quả có giá trị).

**Lưu ý thiên lệch chọn mẫu**: Nhóm A được chọn *theo chính điểm của module* → P@K của từng module riêng lẻ có thể bị thổi phồng khi so sánh. Cách giảm: khi báo cáo ablation, dùng **toàn bộ gold set (A+B)** để xếp hạng lại theo từng cấu hình, và nêu rõ hạn chế này trong paper.

### D6. Phân tích định tính (Case study)
- 3–5 "chiến dịch" tiêu biểu: cụm review trùng rải nhiều quán, đợt burst 5★ của quán mới mở, "bão 1 sao"...
- Phân tích lỗi: 10 False Positive + 10 False Negative điển hình → rút ra nguyên nhân (ví dụ: review ngắn trùng tự nhiên, quán có sự kiện khai trương thật gây burst thật).

### D7. Độ nhạy tham số
- Ngưỡng LSH, `contamination` của IF, cửa sổ 28 ngày → mỗi tham số thử 3 giá trị, xem P@50 thay đổi thế nào (bảng nhỏ trong phụ lục).

---

## 6b. (v2) DỮ LIỆU DEMO & KIỂM THỬ

- `config.demo.yaml` + `src/synthetic/`: sinh ~50 quán, ~2 năm review với 5 kiểu chiến dịch spam cài sẵn (dán mẫu nhiều quán, burst 5★, "bão 1★", quảng cáo chèn SĐT/link, tài khoản ảo đi nhiều quán cùng ngày) và một benchmark kiểu ViSpamReviews giả lập. Dữ liệu demo nằm riêng trong `demo/`, **không trộn** với `data/`.
- Mục đích: chạy thử toàn bộ pipeline trước khi có dữ liệu thật. **Số liệu từ demo không được đưa vào paper.**
- Scraper được kiểm thử trên một trang HTML giả lập cấu trúc Google Maps (chạy cục bộ) để xác nhận logic cuộn, mở rộng "Thêm", trích xuất và bắt gói RPC. Việc tương thích với Google Maps thật phải kiểm tra trên máy của bạn (Phase 0).

## 7. ĐẠO ĐỨC & TUÂN THỦ (đưa vào mục Ethics của paper)
- Chỉ thu thập dữ liệu **công khai**, quy mô nhỏ, mục đích học thuật phi thương mại.
- Ẩn danh reviewer bằng hash có salt; **không công bố** file review thô; ví dụ trích trong paper được paraphrase/cắt ngắn và bỏ tên người.
- Không công bố tên địa điểm cụ thể bị hệ thống gắn "nghi vấn" (điểm số là xác suất, không phải kết luận — tránh gây tổn hại danh tiếng cho doanh nghiệp).
- Ghi rõ nguồn dữ liệu (SerpApi/Places API) và giới hạn điều khoản sử dụng.

---

## 8. LỘ TRÌNH THEO MILESTONE MÔN HỌC

| Phase | Nội dung | Deliverable | Gắn với milestone |
|---|---|---|---|
| **0. Chuẩn bị** (2–3 ngày) | Tạo repo, key API, **email xin ViSpamReviews**, gọi thử 1 place, chốt schema thật (A7.1), chốt công cụ tách từ | `config.yaml`, 1 file JSON mẫu | — |
| **1. Thu thập** (4–5 ngày) | A2–A5, chạy thu thập, theo dõi coverage | `raw/`, bảng coverage | Proposal (tuần 6): trình bày nguồn dữ liệu & thống kê sơ bộ |
| **2. Làm sạch + EDA** (3 ngày) | Phần B | `reviews_clean.parquet`, bảng DQ, hình EDA | Proposal (tuần 6) |
| **3. Module 1a, 2, 3** (5–6 ngày) | C1, C3, C4 | `features_*.parquet` | Progress 2 (tuần 8) |
| **4. Module 1b + D1** (3–4 ngày, song song Phase 3) | C2, benchmark | bảng kết quả ViSpamReviews | Progress 2 (tuần 8) |
| **5. Fusion + gold set** (4–5 ngày) | C5, D2, D3 (pilot → full) | `scored_reviews.parquet`, `gold_final.csv`, Kappa | Progress 2 (tuần 8): nộp bản nháp paper có kết quả sơ bộ |
| **6. Đánh giá + viết paper** (5–6 ngày) | D4–D7, hình/bảng, viết Results/Discussion/Limitations | paper hoàn chỉnh + slide | Final (tuần 9) |

**Đường găng (critical path)**: Thu thập → Fusion → Gán nhãn. Gán nhãn thủ công 250 review × 2 người mất khoảng 2 buổi — **lên lịch với người gán nhãn thứ 2 ngay từ bây giờ**.

---

## 9. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Khả năng | Ảnh hưởng | Dự phòng |
|---|---|---|---|
| Hết quota SerpApi | Trung bình | Thiếu dữ liệu | Giảm số place / giới hạn 24 tháng / Playwright cho vài place |
| Không xin được ViSpamReviews kịp | Trung bình | Mất D1, M1b yếu | Dùng M1b dạng luật (URL/SĐT/độ dài/từ khóa quảng cáo) và nêu rõ; hoặc gán nhãn thêm 300 review GMaps để train nhẹ |
| API không trả số review của reviewer | Thấp–TB | M3 yếu | Dùng contributor API cho tập nhỏ; thay bằng đặc trưng từ chính corpus (số review của reviewer trong corpus) |
| Quá ít spam trong gold set (base rate rất thấp) | Trung bình | P@K không ổn định | Tăng Nhóm A; báo cáo CI bootstrap; chọn thêm place "nghi vấn" có chủ đích ở A1 |
| Kappa thấp | Trung bình | Nhãn không tin cậy | Pilot + chỉnh rubric trước khi gán toàn bộ |
| Không đủ GPU cho PhoBERT | Thấp | | Dùng baseline TF-IDF+LR (vẫn hợp lệ cho paper) hoặc Colab |

---

## 10. CHECKLIST "SẴN SÀNG VIẾT CODE"

- [ ] Đã chạy `pytest` và `run_pipeline.py --config config.demo.yaml` thành công trên máy
- [ ] Đã chạy thử scraper Playwright cho 1 quán thật, kiểm tra selector còn đúng (Phase 0)
- [ ] (Chỉ khi dùng SerpApi) có SerpApi key, đã kiểm tra quota
- [ ] Đã xin quyền tải ViSpamReviews v2 (dataset "gated" trên HuggingFace: `sonlam1102/vispamdetection2`)
- [ ] Đã gọi thử 1 place, lưu JSON mẫu, chốt schema A7
- [ ] Đã chốt danh sách 40–60 địa điểm (file `places_selected.jsonl`)
- [ ] Đã chốt công cụ tách từ + mô hình embedding tiếng Việt, cài đặt chạy được
- [ ] Đã có người gán nhãn thứ 2 và lịch gán nhãn
- [ ] Đã cập nhật mục 4.1 & 4.2 của đề cương theo Mục 0

---

## 11. ÁNH XẠ SANG CÁC MỤC CỦA PAPER

| Mục paper | Lấy từ |
|---|---|
| Data Collection | Phần A + bảng coverage |
| Data Quality & Preprocessing | B1 (bảng 6 chiều), B2–B3 |
| Methodology | Phần C (sơ đồ Mục 1 + công thức C3, C5) |
| Experimental Setup | D1–D3 (benchmark, gold set, Kappa) |
| Results | D4–D5 (bảng P@K, ablation, đường cong) |
| Discussion | D6–D7 (case study, lỗi, độ nhạy) |
| Limitations & Ethics | Mục 0, D5 (lưu ý thiên lệch chọn mẫu), Mục 7, Mục 9 |
