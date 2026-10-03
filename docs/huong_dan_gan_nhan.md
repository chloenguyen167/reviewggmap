# Hướng dẫn gán nhãn gold set — review Google Maps nghi vấn

Hai người (A, B) gán nhãn **độc lập** toàn bộ **Nhóm B (300 mục ngẫu nhiên)** và **60 mục Nhóm A**: chính là các dòng có sẵn trong `annotator_A.csv`/`annotator_B.csv` (khoảng 360 mục). Claude (trên claude.ai) gán **toàn bộ Nhóm A** theo đúng hướng dẫn này; 60 mục chung với người dùng để kiểm chứng. **Nhãn người là nguồn chính của mọi con số kết quả**; nhãn Claude chỉ dùng để mở rộng (mục 5). Tổng công gán tay khoảng 360 mục mỗi người, nên chia việc sớm. Mỗi mục mất khoảng 1–2 phút.

## 1. Nguyên tắc chung

1. **Không trao đổi** với người kia cho tới khi cả hai nộp file. Không sửa nhãn sau khi xem nhãn của người kia; bất đồng được xử lý ở bước hòa giải (mục 5).
2. **Chỉ dựa trên bằng chứng nhìn thấy được** trong file và trong `corpus_lookup.csv`. Có thể mở `place_url` để xem bối cảnh của quán. **Không** tra cứu danh tính người viết ở ngoài.
3. Câu hỏi cần trả lời: *review này có phản ánh một trải nghiệm thật, tự nguyện của khách hay không?* Thích hay ghét quán, viết hay hay dở, đồng ý hay không với nội dung đều **không** phải tiêu chí.
4. Khen nhiệt tình hoặc chê gay gắt **tự nó không phải** bằng chứng spam.
5. Đánh giá từng cờ trước, kết luận sau. Thiếu thông tin để quyết một cờ thì điền **0**.
6. **Nhãn nói về bằng chứng, không nói về sự thật.** `Genuine` nghĩa là *không tìm thấy dấu hiệu spam*, không có nghĩa là đã chứng minh review thật. Không ai kiểm chứng được 100%, nên nhãn chỉ ghi nhận bằng chứng nhìn thấy được.
7. **Không dùng ảnh** khi gán nhãn: file không chứa ảnh và Claude không xem được ảnh, nên bỏ ảnh để người và Claude được so trên cùng thông tin. Nếu tình cờ thấy dấu hiệu từ ảnh (ví dụ cùng một ảnh xuất hiện ở nhiều review), ghi vào `ghi_chu` nhưng **không** đổi cờ hay nhãn.

## 2. Các cột trong `gold_sample_blind.csv`

| Cột | Ý nghĩa |
|---|---|
| `item_id` | Mã mục. Thứ tự đã được xáo trộn; file không có điểm số và không cho biết mục thuộc nhóm nào |
| `place_name`, `place_url` | Quán được review |
| `rating`, `text` | Số sao và nội dung (trống = chỉ chấm sao) |
| `date_text`, `review_day` | Ngày Google hiển thị và ngày quy đổi. **Chỉ coi `review_day` là chính xác khi `date_text` dạng "x ngày/tuần trước" hoặc khi `place_reviews_pm3d` có giá trị** |
| `reviewer_n_reviews`, `reviewer_n_photos`, `reviewer_local_guide` | Tổng số review và số ảnh của tài khoản; tài khoản có phải Local Guide không |
| `reviewer_other_reviews` | Tối đa 5 review khác của cùng tài khoản trong dữ liệu: quán (ngày, số sao) |
| `place_daily_avg` | Số review trung bình mỗi ngày của quán |
| `place_reviews_pm3d` | Số review của quán trong cửa sổ ±3 ngày quanh review này (7 ngày). Trống = ngày không đủ chính xác |
| `has_owner_response` | Chủ quán có phản hồi hay không (không dùng để kết luận) |
| `n_photos` | Số ảnh đính kèm review (chỉ để tham khảo, xem nguyên tắc 7) |

`corpus_lookup.csv` chứa **toàn bộ** review. Dùng Ctrl+F để kiểm tra cờ f3.

## 3. Sáu cờ (điền 0 hoặc 1)

| Cờ | Gắn **1** khi | **Không** gắn (giữ 0) khi |
|---|---|---|
| **f1_chung_chung** — nội dung chung chung | Có thể dán y nguyên cho hầu hết quán cùng loại: chỉ gồm lời khen/chê chung ("quán đẹp, nước ngon, nhân viên nhiệt tình") và **không có chi tiết cụ thể nào** (tên món, giá, sự việc, thời điểm, người cụ thể) | Có ít nhất 1 chi tiết riêng của quán hoặc của lần ghé. Review không có chữ |
| **f2_quang_cao** — quảng cáo / có động cơ | Có link, SĐT, Zalo, mã giảm giá, lời mời liên hệ, mua bán; quảng bá quán/sản phẩm **khác**; nội dung do chính quán tự viết (giới thiệu địa chỉ, menu, hotline như tờ rơi); nói rõ được tặng quà hoặc giảm giá **để đổi lấy review** | Chỉ nhắc tên món hay khuyến mãi của quán như một trải nghiệm ("đang giảm 20% nên ghé thử") |
| **f3_trung_noi_dung** — trùng nội dung | Ctrl+F một cụm 5–8 từ đặc trưng trong `corpus_lookup.csv`, thấy **ít nhất 1 review khác gần như y hệt** (chỉ khác vài từ, dấu câu, emoji) của **tài khoản khác** hoặc ở **quán khác**, và review dài **≥ 8 từ** | Câu sáo ngắn dưới 8 từ trùng nhau (chỉ xét f1). Trùng chủ đề nhưng khác cách viết |
| **f4_tai_khoan_it_review_cung_muc_sao** — tài khoản mỏng, chấm cực đoan | `reviewer_n_reviews ≤ 3` **và** `rating` là 1★ hoặc 5★ **và** không phải Local Guide. Cũng gắn khi mọi review trong `reviewer_other_reviews` đều cùng 1★ hoặc cùng 5★ và tài khoản có ≤ 10 review, gần như không có ảnh | Tài khoản ít review nhưng chấm 2–4★. Thiếu số review của tài khoản |
| **f5_nhieu_quan_xa_nhau_cung_luc** — hoạt động bất thường | Trong `reviewer_other_reviews`: cùng tài khoản review **≥ 3 quán khác nhau trong cùng 1 ngày**, hoặc 2 quán ở **khác tỉnh/thành** trong cùng 1 ngày. Chỉ xét khi ngày chính xác | Ngày chỉ là ước lượng ("2 năm trước"). 2 quán gần nhau cùng ngày (đi chơi thật vẫn có thể như vậy) |
| **f6_trong_cum_dang_bat_thuong** — nằm trong đợt dồn review | `place_reviews_pm3d ≥ 5` **và** `≥ 3 × (place_daily_avg × 7)` (tức gấp 3 lần mức bình thường của 7 ngày), **và** review cùng chiều với đợt (vd. 5★ trong đợt 5★) | `place_reviews_pm3d` trống. Quán đông khách vốn có nhiều review mỗi ngày (mức bình thường đã cao) |

> Ví dụ f6: quán có `place_daily_avg = 0,4` → mức bình thường 7 ngày ≈ 2,8 review. Nếu `place_reviews_pm3d = 12` (≥ 5 và ≥ 8,4) thì gắn f6 = 1.

## 4. Kết luận (cột `label`)

| Nhãn | Khi nào |
|---|---|
| **Spam** | Có bằng chứng review không phản ánh trải nghiệm thật và tự nguyện: quảng cáo, tự review, review thuê hoặc đổi quà, chiến dịch đẩy hay dìm điểm, nội dung sao chép |
| **Uncertain** | Có **dấu hiệu cụ thể** nhưng chưa đủ để kết luận (ví dụ đúng 2 cờ, hoặc 1 cờ mạnh mà bối cảnh mâu thuẫn). **Không** dùng Uncertain chỉ vì "không chắc chắn là thật": mục nào cũng như vậy |
| **Genuine** | Không có hoặc gần như không có dấu hiệu (nguyên tắc 6) |

Uncertain không bị bỏ đi: kết quả đánh giá được báo cáo theo 3 cách (loại Uncertain / coi là Spam / coi là Genuine) để thấy kết luận có vững hay không.

**Luật tham khảo** (giống luật trong `agreement.py`):
- có cờ **f2 hoặc f3**, hoặc **≥ 3 cờ** → **Spam**
- **2 cờ** → **Uncertain**
- **0–1 cờ** → **Genuine**

Bạn **được phép kết luận khác luật**, nhưng phải ghi lý do ngắn vào `ghi_chu`. Các mục như vậy sẽ được liệt kê lại trong `label_vs_rubric_inconsistencies.csv`. Các trường hợp hay gặp:
- **Chỉ chấm sao** (khoảng 37% dữ liệu): f1–f3 luôn = 0, chỉ dựa vào f4–f6. Không có dấu hiệu → **Genuine** (theo nguyên tắc 6, nghĩa là "không có bằng chứng"); có dấu hiệu nhưng chưa đủ → Uncertain; không xếp vào Uncertain chỉ vì thiếu chữ. Kết quả đánh giá được báo cáo riêng cho nhóm có chữ và nhóm chỉ chấm sao, vì nhóm chỉ chấm sao có ít bằng chứng hơn hẳn.
- **Đợt review thật** (khai trương, sự kiện, viral): f6 = 1 nhưng nội dung cụ thể và tài khoản đa dạng → có thể kết luận **Genuine**, ghi chú "đợt thật".
- **Lạc đề vô hại** (nhầm quán, hỏi đường…): không gắn f2, kết luận Genuine, ghi chú "lạc đề".
- **Review tiếng nước ngoài hoặc bản dịch máy:** đánh giá theo nội dung như bình thường, ghi chú "dịch".
- **Bị cắt cụt hoặc khó đọc:** đánh giá trên phần nhìn thấy; nếu không đủ thì để Uncertain.

## 5. Quy trình

1. **Pilot (người):** A và B gán độc lập **30 dòng đầu** của file mình vào `data/labels/annotator_A.csv` và `annotator_B.csv`, rồi chạy `python run_pipeline.py --stages agreement`. Nếu Kappa của `label` **< 0,6**: họp, xem các mục bất đồng, bổ sung ví dụ hoặc làm rõ tiêu chí trong file này (ghi lại thay đổi), rồi mới gán tiếp.
2. **Người gán tiếp** toàn bộ các dòng còn lại trong file (Nhóm B + 60 mục Nhóm A). Không sửa lại phần pilot.
3. **Hòa giải:** với các mục bất đồng, hai người thảo luận và ghi kết luận chung vào `data/labels/adjudication.csv` (cột `item_id,label`). Nếu vẫn không thống nhất thì nhờ người thứ ba quyết định.
4. **Claude (claude.ai)** gán toàn bộ Nhóm A *sau khi* hướng dẫn đã chốt ở bước 1 (để Claude dùng đúng bản hướng dẫn cuối):
   - `python run_pipeline.py --stages llm_export` → thư mục `data/labels/claude_ai/` gồm `PROMPT.md` và `batch_01.csv`, `batch_02.csv`…
   - Trên claude.ai: tạo một Project, dán nội dung `PROMPT.md` vào **Project instructions**. Mỗi batch mở **một chat mới**, upload `batch_XX.csv`, gửi "Gán nhãn file này". Tải file CSV Claude trả về vào `data/labels/claude_ai/ket_qua/` (hoặc dán nguyên câu trả lời vào một file `.txt` trong thư mục đó).
   - Không gửi nhãn của người cho Claude, không sửa tay kết quả của Claude.
   - `python run_pipeline.py --stages llm_import` → kiểm tra đủ mục và đúng định dạng, gộp thành `annotator_C.csv`. Nếu báo thiếu mục, gửi lại các dòng đó trong một chat mới.
5. `python run_pipeline.py --stages agreement` so Claude với người trên 60 mục Nhóm A chung. Nhãn Claude **chỉ được dùng** cho phần còn lại của Nhóm A khi κ(người–Claude) ≥ 0,6 **và** không thấp hơn κ(A–B) quá 0,1. Kết luận ghi ở `llm_validation.csv`. Nếu không đạt: người gán thêm, hoặc chỉnh hướng dẫn rồi cho Claude gán lại toàn bộ. Kể cả khi đạt, P@25 và P@50 vẫn tính **chỉ trên nhãn người**; P@100 và P@150 mới mở rộng sang nhãn Claude và được ghi rõ nguồn. Lý do: file gán nhãn đưa cho Claude đúng những bối cảnh mà M1a/M2/M3 dùng để chấm điểm, nên dựa chủ yếu vào nhãn Claude có nguy cơ "đo lại chính đặc trưng của hệ thống".
6. Báo cáo Kappa **trước** hòa giải, kèm % đồng ý. Kappa của cờ hiếm gặp có thể ≈ 0 dù hai người đồng ý gần như toàn bộ. `gold_final.csv` có cột `label_source` (human/llm) để báo cáo kết quả tách riêng theo nguồn nhãn.

**Định dạng:** cờ là `0`/`1`; `label` là `Genuine`, `Uncertain` hoặc `Spam`; `ghi_chu` là tự do, ngắn gọn. Không thêm, xóa hay đổi thứ tự cột.
