# ═══════════════════════════════════════════════════════════════════════════
# SEED DÙNG CHUNG CHO V40 / PHOBERT
# ───────────────────────────────────────────────────────────────────────────
# QUAN TRỌNG: đây là file DUY NHẤT định nghĩa danh sách seed. Cả train_v40.py,
# train_v40.py và train_phobert.py đều import từ đây (KHÔNG tự khai seed riêng
# trong mỗi file) để hai hệ thống luôn được đánh giá trên ĐÚNG 10 seed giống hệt
# nhau — điều kiện bắt buộc để paired test giữa các hệ thống có nghĩa (mỗi cặp
# điểm F1 phải đến từ cùng một lần chia Train/Calibration/Test).
#
# Điều kiện "cùng seed" là CẦN nhưng CHƯA ĐỦ: cùng seed mà khác hàm chia thì
# vẫn ra hai split khác nhau (đã xảy ra 2 lần — xem split_fingerprint.py).
# Dấu vân tay split được kiểm tra tự động ở mỗi lần chạy; đừng dựa vào việc
# ba file "cùng import file này" như một bảo đảm.
#
# ───────────────────────────────────────────────────────────────────────────
# VÌ SAO ĐỔI TỪ SEED RỜI RẠC SANG 10 SEED LIÊN TIẾP 42–51
#
# Bản trước: SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321], với
# lý do ghi trong comment là "rải đều, không theo một pattern số học dễ đoán".
# Mục tiêu "khó đoán" đó không mang lại lợi ích nào ở đây, và đánh đổi mất một
# thứ có giá trị thật:
#
#   1. KHÔNG có vấn đề kỹ thuật với seed liền nhau. Lo ngại "seed gần nhau sinh
#      chuỗi tương quan" đúng với một số bộ sinh tuyến tính đồng dư (LCG) cũ,
#      KHÔNG đúng với Mersenne Twister mà NumPy/scikit-learn dùng: seed đi qua
#      init_by_array để nạp toàn bộ 624 từ trạng thái, nên seed 42 và 43 cho hai
#      chuỗi độc lập không kém gì 42 và 31337. PyTorch (Philox/MT) cũng vậy.
#
#   2. Seed liên tiếp thì TỰ CHỨNG MINH là không chọn lọc. Đọc một danh sách rời
#      rạc, reviewer có quyền hỏi: "tại sao đúng 10 số này — có phải đã thử nhiều
#      bộ rồi giữ bộ cho kết quả đẹp nhất?" — và không có cách nào chứng minh là
#      không. Một khoảng liên tục range(42, 52) không còn chỗ cho lựa chọn: chỉ
#      cần nêu điểm bắt đầu và số lượng là toàn bộ thiết kế đã được xác định.
#      Đây là câu trả lời trực tiếp cho dòng 10 trong Bảng điểm yếu tiềm ẩn
#      ("Làm sao tôi biết các con số này không phải chọn lọc có lợi?").
#
#   3. Giữ 42 ở VỊ TRÍ ĐẦU TIÊN — không phải vì mê tín: train_phobert.py sao lưu
#      checkpoint đầy đủ riêng cho seed 42, và 42 là seed của lần chạy đơn đã
#      báo cáo trong Table 8 — giữ nó để còn đối chiếu ngược.
#
# CÂU DÙNG ĐƯỢC TRONG BÀI (Mục 4.4 — Quy trình đảm bảo độ tin cậy):
#   "Mọi thí nghiệm được lặp trên 10 seed liên tiếp (42–51), cố định trước khi
#    chạy và dùng chung cho cả hai hệ thống. Khoảng seed liên tục được chọn thay
#    cho một tập giá trị rời rạc để loại bỏ khả năng lựa chọn seed hậu nghiệm:
#    toàn bộ thiết kế được xác định bởi điểm bắt đầu và số lần lặp. Không có
#    seed nào bị loại khỏi báo cáo."
#
# NẾU CẦN TĂNG SỐ LẦN LẶP: mở rộng về PHÍA CUỐI (range(42, 62) cho 20 seed) và
# báo cáo TOÀN BỘ, không thay điểm bắt đầu và không bỏ seed nào ở giữa. Mở rộng
# về phía cuối giữ nguyên 10 seed cũ như một tập con, nên kết quả cũ vẫn dùng
# lại được thay vì phải chạy lại từ đầu.
# ═══════════════════════════════════════════════════════════════════════════

N_SEEDS = 10
SEED_START = 42

SEEDS = list(range(SEED_START, SEED_START + N_SEEDS))   # [42, 43, ..., 51]

assert len(SEEDS) == N_SEEDS and len(set(SEEDS)) == N_SEEDS, \
    f"SEEDS phải có đúng {N_SEEDS} giá trị, không trùng lặp."
assert SEEDS[0] == 42, \
    ("SEEDS[0] phải là 42 — bản sao checkpoint PhoBERT và lần chạy đơn "
     "đã báo cáo đều gắn với seed này.")

# ───────────────────────────────────────────────────────────────────────────
# CẢNH BÁO KHI ĐỔI FILE NÀY
# Đổi danh sách seed làm MỌI results_*.json đã có trở nên không so sánh được với
# lần chạy mới. Hai file kết quả (results_v40_multiseed.json,
# results_phobert_multiseed.json) phải được sinh lại
# TRỌN BỘ từ cùng một danh sách seed; compare_systems.py sẽ từ chối gộp bảng nếu
# phát hiện hai file khai báo các danh sách seed khác nhau.
# ───────────────────────────────────────────────────────────────────────────
