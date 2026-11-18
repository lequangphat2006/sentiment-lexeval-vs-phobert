#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# V40: AERS‑Pro Cải tiến  —  Khắc phục nhược điểm V38/V39
# ───────────────────────────────────────────────────────────────────────
# • Fix lỗi giá trị âm với ComplementNB → dùng scaler, bỏ CNB.
# • Stacking chuẩn: OOF riêng cho từng learner → meta‑features đầy đủ.
# • Sử dụng tập Calibration để chọn siêu tham số meta‑learner.
# • Tối ưu siêu tham số feature & base learner bằng Optuna.
# • Mở rộng lexicon, cải thiện trích xuất đặc trưng.
# • Đánh giá ổn định với nhiều seed (tuỳ chọn).
# ───────────────────────────────────────────────────────────────────────
# CHANGELOG v40.1 — xem lịch sử git (BUGFIX-1/2, FIX-3/4, FEAT-5/6, ARCH-7/8)
# CHANGELOG v40.2 — xem lịch sử git (LEAK-9, GROUPCV-10, FIX-11, LEXFIX-13,
#                   SCALE-14, INFER-15, SPLIT-16, CLEAN-18, ARCH-19)
# ───────────────────────────────────────────────────────────────────────
# CHANGELOG v40.3 (audit toàn diện dựa trên log 10 seed ngày 2026‑07‑20:
#                  F1=0.8667±0.0080, neutral P−R=+0.054 ở CẢ 10/10 seed):
#
# ── NHÓM A — SỬA LỖI ĐÚNG NGHĨA ────────────────────────────────────────
# [LEX-20]    ⛔ NGHIÊM TRỌNG NHẤT. extract_features() so khớp từ điển trên
#             văn bản THÔ (chỉ text.lower()), trong khi teencode/emoji chỉ
#             được chuẩn hoá ở preprocess() — một nhánh HOÀN TOÀN KHÁC. Với
#             review TMĐT tiếng Việt, "sp nay tot lam" / "ko hai long" /
#             "dep qua" TRƯỢT toàn bộ lexicon. Vì 9/14 cột thủ công phái
#             sinh từ matched_count, một lỗi kéo theo cả khối thành ma trận
#             gần‑0 — đây chính là gốc rễ của "ma trận thưa cho kết quả hạn
#             chế". Bằng chứng trong log: MI của ngưỡng neutral chỉ
#             0.044–0.060 và chọn đúng giá trị mặc định T=2.0 ở 8/10 seed.
#             → Tách normalize_for_lexicon(): chuẩn hoá emoji/teencode/NFC
#               nhưng GIỮ dấu câu và GIỮ stopword (luật CLAUSE_BOUNDARY và
#               phủ định cần chúng), rồi mới so khớp.
# [SEG-21]    ⛔ Phân đoạn từ là NO-OP. underthesea.word_tokenize() trả về
#             list các từ ghép CHỨA DẤU CÁCH (['chất lượng','tốt']), nên
#             ' '.join(...) dựng lại ĐÚNG chuỗi ban đầu. Phải dùng
#             format="text" → "chất_lượng tốt". Đây cũng là lý do Optuna
#             chọn ngram_range=(1,2) ở 9/10 seed: bigram ÂM TIẾT đang gánh
#             vai trò của bộ phân đoạn từ. (Log cũng cho thấy bước POS mất
#             0 ms ở cả 10 seed ⇒ HAS_POS=False ⇒ underthesea chưa cài.)
# [LEX-22]    INTENSIFIERS được khai báo như hệ số nhân (rất=1.5, hơi=0.7)
#             nhưng chỉ được dùng như ĐẾM TÚI TỪ — "hơi tệ" và "cực tệ" cho
#             cùng điểm −3. Thêm nữa `w in text_low` là so khớp CHUỖI CON:
#             "quá" khớp trong "quán", "cực" bị đếm 2 lần với "cực kỳ".
#             → Nhân thật vào điểm của từ cảm xúc kế tiếp, so khớp theo TOKEN.
# [LEX-23]    TEENCODE_MAP xung đột ngữ nghĩa với lexicon: 'tệ'(NEG=3)→'kém'
#             (NEG=2) mất 1 bậc cường độ; 'ok'(POS=1)→'được' mất trắng vì
#             'được' không có trong lexicon. Các xung đột này VÔ HẠI ở v40.2
#             (vì lexicon chạy trên text thô) nhưng sẽ CÓ HIỆU LỰC ngay khi
#             áp dụng [LEX-20] → phải dọn TRƯỚC, nếu không điểm có thể GIẢM.
# [NUM-24]    re.sub(r'\d+','') xoá sạch "1 sao"/"5 sao"/"10/10" — một trong
#             những n‑gram phân biệt mạnh nhất của review TMĐT.
#             → Ánh xạ thành token rating_1..rating_5 / score_x_y TRƯỚC khi xoá.
# [SCALE-25]  MinMaxScaler cực nhạy outlier: pos_s có đuôi dài, một review
#             dài khiến ~99% mẫu bị nén về [0,0.05]. [SCALE-14] clip [0,1]
#             chỉ CHE GIẤU chứ không sửa. → QuantileTransformer (theo thứ hạng).
# [TFIDF-26]  TF‑IDF thiếu min_df/max_df ⇒ giữ cả hapax (thuần nhiễu) — đúng
#             thứ mà lr_C≈10 sẽ ghi nhớ. → min_df/max_df, đưa min_df vào Optuna.
# [POS-27]    _global_pos_single() tính 4 tỉ lệ nhưng FeaturePipeline chỉ
#             dùng 2 (bỏ ratio_adv — cái tương quan MẠNH NHẤT với cường độ
#             cảm xúc trong tiếng Việt). POS còn chạy trên raw text (còn
#             emoji/teencode/URL) khiến gán nhãn sai hàng loạt.
#
# ── NHÓM B — KIẾN TRÚC ─────────────────────────────────────────────────
# ⚠ NGHIỆM THU [DECIDE-28]: lớp quyết định được thiết kế dựa trên độ lệch
#   P−R quan sát trong log v40.2 (neutral +0.054, nhất quán 10/10 seed). Nó
#   KHÔNG thể được nghiệm thu trên dữ liệu giả vì dữ liệu giả không có độ
#   lệch hệ thống nào để sửa. Hãy đọc `delta_decision_layer` trong
#   results_v40_multiseed.json sau khi chạy đủ 10 seed: nếu trung bình KHÔNG
#   dương, đặt Config.use_decision_layer=False. Mọi thành phần v40.3 khác
#   độc lập với nó.
#
# [DECIDE-28] ★ LỢI ÍCH LỚN NHẤT. np.argmax(prob) tối ưu ACCURACY, KHÔNG
#             tối ưu MACRO‑F1. Log 10 seed: neutral P=0.869/R=0.815 (P−R=
#             +0.054), negative P−R=−0.033 — CÙNG DẤU ở 10/10 seed, không
#             phải nhiễu: mô hình đẩy neutral sang negative có hệ thống.
#             → Quy tắc plug‑in ŷ = argmax_c w_c·p_c(x), w dò trên Cal.
#               Cơ sở: Koyejo et al. NeurIPS 2014; Narasimhan et al. ICML 2014.
#             ⚠ Bản "trần" của kỹ thuật này TỰ NÓ quá khớp (đã kiểm chứng:
#               đẩy w_neutral lên 2.18 trong khi đúng là ~1.8, lợi ích rơi
#               từ +0.0093 xuống +0.0013) → BẮT BUỘC bagging + co ngót.
# [STACK-29]  3 base learner (lr/svm/lgbm) nhìn CÙNG MỘT ma trận X ⇒ lr và
#             svm đều tuyến tính trên cùng không gian ⇒ tương quan dự đoán
#             rất cao. Theo phân rã Krogh–Vedelsby (NIPS 1995):
#                 E_ensemble = Ē_individual − Ā_ambiguity
#             Ā≈0 ⇒ ensemble ≈ learner đơn, chi phí gấp 3 lợi ích ≈ 0. Thêm
#             nữa LGBM trên ma trận thưa 52k chiều là lựa chọn tệ (cây chia
#             theo từng đặc trưng, mật độ <0.1%).
#             → STACKING ĐA GÓC NHÌN: word / char / NB‑weighted / SVD‑dense
#               / ordinal — mỗi learner một không gian khác nhau.
# [STACK-30]  LỆCH PHÂN BỐ meta_train vs meta_cal/test: [LEAK-9] fit pipeline
#             fold‑local (đúng về chống rò rỉ) nhưng cal/test lại đi qua
#             pipeline thứ 6 fit trên 100% train ⇒ OOF đến từ 5 TỪ VỰNG KHÁC
#             NHAU, meta học phân bố A rồi áp lên phân bố B. Đây rất có thể
#             là lý do C tối ưu của meta nhảy loạn giữa 0.01 và 10.0 qua các
#             seed. → BAGGED STACKING: cal/test = TRUNG BÌNH K mô hình fold.
#             Chi phí thêm ≈ 0 (K mô hình vốn đã train, v40.2 vứt đi).
# [NBSVM-31]  Thêm view NB‑weighted (log‑count‑ratio) — Wang & Manning,
#             ACL 2012. r = log((p/‖p‖₁)/(q/‖q‖₁)) chính là log tỉ số hợp lý
#             dưới NB đa thức; nhân vào TF‑IDF ≈ khởi tạo mô hình tuyến tính
#             tại nghiệm NB. Hiệu quả nhất khi tỉ lệ mẫu/chiều nhỏ — đúng
#             trường hợp này (15.000/52.782 ≈ 0.28).
# [ORD-32]    Nhãn có cấu trúc THỨ TỰ negative<neutral<positive nhưng mô hình
#             xem 3 lớp là PHẲNG, trong khi 100% khối lượng lỗi nằm ở ranh
#             giới neutral. → Phân rã Frank–Hall (ECML 2001): 2 bài nhị phân
#             tích luỹ P(y>neg), P(y>neu) rồi tái hợp; neutral trở thành GIAO
#             của hai quyết định dễ hơn thay vì phải tự bảo vệ ở giữa.
# [META-33]   Đầu vào meta chỉ có xác suất thô + cluster + lexicon; meta là
#             LR TUYẾN TÍNH nên không tự học được margin/entropy/bất đồng —
#             các đại lượng phi tuyến quan trọng nhất của stacking. → Cấp sẵn.
# [CHAR-34]   HashingVectorizer(char_wb) chiếm 62% số cột (32.768/52.782) mà
#             KHÔNG có IDF ⇒ n‑gram phổ biến ("ng","th") nhận trọng số ngang
#             n‑gram hiếm mang thông tin; alternate_sign=True còn gây triệt
#             tiêu do va chạm hash. → TfidfVectorizer(char_wb) có IDF+min_df.
# [CALIB-35]  Quy tắc [ARCH-8] fit VÀ đánh giá calibrator trên CÙNG tập Cal
#             ⇒ overfit. Bằng chứng 10 seed: ECE‑Test khi ÁP DỤNG = 0.0263
#             (n=6) còn TỆ HƠN khi LOẠI = 0.0237 (n=4). Ngoài ra LR trên
#             log‑odds = vector scaling 12 tham số, quá nhiều cho ~5.000 mẫu
#             Cal đã bị chia sẻ 3 mục đích. → Temperature scaling (ĐÚNG 1
#             tham số, Guo et al. ICML 2017) + chấp nhận qua K‑fold nội bộ.
# [CAL-36]    Tập Cal bị dùng lại 3 LẦN (dò T, chọn C meta, nhận/loại
#             calibrator) ⇒ best_f1_cal là ước lượng CHỆCH LÊN. Log xác nhận:
#             Cal F1 dao động 0.8530–0.8967 (4.4 điểm) trong khi khoảng cách
#             giữa các ứng viên C chỉ ~0.5 điểm ⇒ chọn C là tung đồng xu.
#             → T dò trên TRAIN; C meta chọn bằng CV group‑aware trên TRAIN;
#               Cal chỉ còn dùng cho temperature + decision layer.
#
# ───────────────────────────────────────────────────────────────────────
# CHANGELOG v40.4 (rà soát ngày 2026‑08‑09, khởi từ báo cáo "độ phủ lexicon
#                  đứng ở 54.x–55% dù đã cài underthesea"):
#
# ── CHẨN ĐOÁN: "ĐÃ CÀI UNDERTHESEA MÀ KHÔNG ĐỔI" LÀ ĐÚNG NHƯ KỲ VỌNG ──
# Nhánh tính độ phủ là _score() → normalize_for_lexicon() →
# _tokenize_for_lexicon(), KHÔNG hàm nào gọi word_tokenize(). underthesea chỉ
# được dùng ở _segment() (nhánh TF‑IDF) và pos_tag(). Đo trên cùng 20k mẫu:
#     chưa cài underthesea → 0.5457  |  ĐÃ cài (HAS_POS=True) → 0.5456
# ⇒ [SEG-21] đã sửa ĐÚNG, nhưng nó sửa cho TF‑IDF, không sửa cho lexicon.
# ⚠ ĐỪNG bật segmentation cho nhánh lexicon: đã thử, độ phủ GIẢM xuống
#   0.4759 vì underthesea gộp nhanh→nhanh_chóng, chậm→chậm_trễ… mà lexicon
#   70 mục không có các từ ghép đó.
#
# [LEX-47]  ★ NGUYÊN NHÂN THẬT. Lexicon v40.3 chỉ có 70 mục (34 POS+36 NEG).
#           PHAN_TICH_VA_CAI_TIEN_V40.md §[C6] đã ghi "cần 800–2.000 mục" và
#           §3.2 đề xuất mở rộng bán giám sát bằng PMI/delta‑IDF — nhưng §3.2
#           CHƯA HỀ được implement (grep 'pmi|delta_idf|expand_lexicon' trên
#           v40.3 → 0 kết quả). Lớp từ thiếu nặng nhất là TỪ GHÉP/TỪ LÁY:
#           lọc token phân biệt POS/NEG mạnh nhất mà lexicon không có
#           (df≥250) thì gần như toàn là ÂM TIẾT LẺ của từ ghép — thoải/mái,
#           hụt/hẫng, phũ/phàng, cồng/kềnh, loang/lổ, lạch/cạch, nồng/nặc.
#           → lexicon_v40_4.py: 197 mục hạt giống thủ công (KHÔNG dùng nhãn)
#             + delta_logodds_expand() fit trên TRAIN của TỪNG seed.
#           Đo (20k mẫu, seed 42, macro‑F1 của LR chỉ dùng 22 cột thủ công):
#             v40.3            70 mục  cov 0.5420/0.5495  F1 0.5264
#             +hạt giống      267 mục  cov 0.7831/0.7970  F1 0.6538
#             +mở rộng        667 mục  cov 0.9898/0.9915  F1 0.7140
#           Khe hở Train/Test = −0.0017 (Test CAO HƠN) ⇒ không quá khớp.
#
# ── BỐN LỖI CƠ HỌC TRONG NHÁNH LEXICON ─────────────────────────────────
# [E1-48]   ELONG_RE rút ký tự lặp về HAI ký tự (r'\1\1') nên "tốtttt"→"tốtt",
#           "ngonnnn"→"ngonn" — cả hai KHÔNG có trong lexicon (2,7% review).
#           Còn phá teencode: "khonggg"→"khongg" nên 'khong'→'không' không
#           bao giờ chạy (ELONG chạy TRƯỚC teencode). → rút về MỘT ký tự.
# [E2-49]   _tokenize_for_lexicon chỉ đệm khoảng trắng cho [.,;!?]. Mọi dấu
#           khác còn dính token: '(giao', 'sai)', 'chất_lượng:', 'tốt/xấu',
#           'đẹp-bền-rẻ', '"tuyệt_vời"' — tất cả TRƯỢT (7,1% review).
# [E3-50]   RATING_PATTERNS sinh score_9_10/score_10 nhưng lexicon không có
#           mục score_* nào ⇒ sinh token rồi bỏ.
# [E4-51]   ⛔ NẶNG NHẤT Ở NHÁNH TF‑IDF. preprocess() gọi re.sub(r'\d+',' ')
#           SAU khi rating_N đã được tạo, nên \d+ khớp luôn chữ số TRONG
#           'rating_5' ⇒ rating_1..rating_5 bị nghiền thành CÙNG một token
#           'rating_' (1.484/20.000 mẫu). "1 sao" và "5 sao" thành giống
#           nhau — toàn bộ [NUM-24] bị vô hiệu, đúng ngược với comment
#           "rating_* đã được giữ". → đổi thành token không chữ số
#           (rating_5→ratingfive) TRƯỚC khi xoá chữ số.
#
# ── LỖI KHÁC TÌM ĐƯỢC TRONG LÚC RÀ SOÁT ────────────────────────────────
# [DEAD-52] 6/22 cột thủ công gần như là ma trận 0 (đo trên 20k mẫu):
#           is_question 100,0% | pos_s_x_tagq 100,0% | neg_s_x_tagq 100,0%
#           quest_ratio 99,9% | excl_ratio 99,7% | is_tag_question 99,4%.
#           is_question dùng raw.endswith('?') — dữ liệu gần như không có
#           review kết thúc bằng '?'. → đổi thành "có '?' ở bất kỳ đâu".
# [FSTR-53] f-string ở load_data dùng {{...}} nên in ra CHỮ LITERAL
#           "Đã BỎ {_n_dropped}/{_n_before} dòng" thay vì giá trị. Đúng chỗ
#           mà theo comment ngay trên là tình huống làm đứt pairing giữa
#           V40/PhoBERT — mà lại không in được số.
# [INT-54]  pending_int giữ hệ số intensifier vô hạn tầm với:
#           "rất nhiều người nói sản phẩm này tốt" → pos_s = 3.0 (1.5×2),
#           trong khi 'rất' bổ nghĩa cho 'nhiều'. → giới hạn khoảng cách.
# [ABLA-55] Cờ ablation normalize_lexicon=False làm nhiễu CẢ nhánh TF‑IDF
#           (vì preprocess() gọi normalize_for_lexicon() và hàm này return
#           sớm) ⇒ dòng ablation "tắt [LEX-20]" đang đo HAI thay đổi cùng
#           lúc. → tách cờ riêng cho từng nhánh.
# [PAR-56]  CFG và lexicon KHÔNG lan sang worker của loky (worker import lại
#           module ⇒ nhận Config() mặc định và lexicon gốc). Vô hại khi mọi
#           cờ đều True, nhưng CHẾT khi lexicon được mở rộng theo fold.
#           → extract_features_batch chạy tuần tự (đo: 6,6s/68,7k mẫu, không
#             cần song song); CFG được đồng bộ tường minh cho worker.
# [SERV-57] Bundle không lưu lexicon ⇒ mô hình train với lexicon mở rộng mà
#           inference dùng lexicon gốc = training/serving skew âm thầm.
#           → lưu lexicon vào bundle và nạp lại trong predict_v40().
# ───────────────────────────────────────────────────────────────────────
# ── NHÓM C — TỐI ƯU SIÊU THAM SỐ ───────────────────────────────────────
# [OPT-37]    Optuna chạm BIÊN không gian tìm kiếm: lr_C sát cận trên 10.0 ở
#             4/10 seed (9.95/9.89/9.55/9.46), tfidf_max_feat chạm 20000 ở
#             2/10 seed. → Nới lr_C→[0.01,100], max_feat→[20k,80k].
# [OPT-38]    Objective đo F1 của MỘT LR đơn lẻ trong khi chỉ số cuối là F1
#             của cả stack ⇒ lệch hệ thống 3.3 điểm (CV 0.8335 vs Test
#             0.8667), và ngốn 65% thời gian mỗi seed (463s/711s).
#             → Thêm manual features vào objective cho gần mục tiêu cuối,
#               thêm MedianPruner, giảm n_trials mặc định.
# [OPT-39]    random_state=42 HARDCODE trong CV nội bộ của objective dù split
#             ngoài dùng `seed` ⇒ fold nội bộ giống hệt ở cả 10 seed.
#
# ── NHÓM D — CHẤT LƯỢNG MÃ & GIAO THỨC ─────────────────────────────────
# [CLEAN-40]  Xoá import chết: RandomForestClassifier, CatBoostClassifier,
#             SelectKBest, chi2, issparse, dataclass. Đặc biệt mỉa mai:
#             SelectKBest/chi2 được import nhưng CHƯA BAO GIỜ gọi — trong
#             khi lọc đặc trưng chính là thứ mô hình đang thiếu.
# [DATA-41]   load_data() ÂM THẦM tạo 3000 mẫu giả khi không thấy file (chỉ
#             warning). Chạy sai thư mục ⇒ script vẫn báo F1≈1.0 và ghi vào
#             results_*.json. Với đề tài học thuật phải RAISE.
# [SPLIT-42]  Nhánh fallback cuối của split_data_v40() KHÔNG group‑aware —
#             im lặng chấp nhận leakage. → RAISE nếu có group_col mà cả 2
#             cách group‑aware đều lỗi.
# [ECE-43]    ECE 10 bin ĐỀU trong khi confidence dồn vào [0.7,1.0] ⇒ 6/10
#             bin gần rỗng, phương sai ước lượng lớn. → Thêm ECE đều‑mật‑độ
#             và classwise‑ECE. GIỮ NGUYÊN `ece` bin‑đều để bảng đã công bố
#             (V40/PhoBERT) còn so sánh được.
# [REPORT-44] matplotlib/seaborn được import và thư mục reports/ được tạo
#             nhưng KHÔNG file nào được ghi. → Xuất ma trận nhầm lẫn, biểu đồ
#             tin cậy, chỉ số theo lớp, chẩn đoán ma trận thưa.
# [DIAG-45]   Ghi ĐỘ PHỦ LEXICON / mật độ ma trận / tỉ lệ OOV vào log+JSON —
#             ba con số trả lời trực tiếp câu hỏi "ma trận thưa còn hạn chế".
# [ABLA-46]   Toàn bộ thành phần mới đều có cờ bật/tắt trong Config để dựng
#             bảng ablation mà không sửa code.
#
# ⚠ LƯU Ý KÍCH THƯỚC MÔ HÌNH: bagged stacking [STACK-30] lưu K×V bộ trích
#   đặc trưng thay vì 1 ⇒ model phình. Đã bù bằng joblib compress=3 và
#   Config.max_features. Nếu ưu tiên kích thước hơn độ chính xác, đặt
#   Config.oof_mode='refit' để quay về hành vi v40.2 (nhỏ hơn ~5×).
# ═══════════════════════════════════════════════════════════════════════════

import os, sys, re, glob, json, time, random, warnings, logging, logging.handlers, unicodedata
import multiprocessing
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sp_hstack, csr_matrix
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GroupShuffleSplit, StratifiedGroupKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
# [CLEAN-40] đã xoá: RandomForestClassifier, CatBoostClassifier, SelectKBest,
# chi2, MinMaxScaler, HashingVectorizer, issparse, clone — không nơi nào dùng.
import joblib
from joblib import Parallel, delayed
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from underthesea import word_tokenize, pos_tag
    HAS_UNDER = True
    HAS_POS = True
except ImportError:
    try:
        from underthesea import word_tokenize
        HAS_UNDER = True
        HAS_POS = False
    except ImportError:
        HAS_UNDER = False
        HAS_POS = False

if HAS_UNDER:
    try:
        word_tokenize("khởi tạo")
        if HAS_POS:
            pos_tag("khởi tạo")
    except Exception:
        pass

# [LEX-47] Từ điển TMĐT tiếng Việt + mở rộng bán giám sát (§3.2 của tài liệu
# phân tích — phần chưa từng được implement trong v40.3).
import lexicon_v40_4 as LEX44

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# CONFIG & LOGGING
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
_log_file = os.path.join(LOG_DIR, f"train_v40_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger('SentimentV40')
logger.setLevel(logging.DEBUG)
_fh = logging.handlers.RotatingFileHandler(_log_file, maxBytes=15_000_000, backupCount=5, encoding='utf-8')
_ch = logging.StreamHandler(sys.stdout)
_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
for h in [_fh, _ch]:
    h.setFormatter(_fmt)
    logger.addHandler(h)

N_CORES = multiprocessing.cpu_count()
N_WORKERS = max(1, N_CORES - 1)
N_PARALLEL_PRE = N_WORKERS

VERSION = 'v40'
SUBVERSION = 'v40.4'
N_OOF_FOLDS = 5
JOBLIB_BACKEND = 'loky'
JOBLIB_COMPRESS = 3          # [STACK-30] bù kích thước cho bagged stacking

for d in ['models', 'models/production', 'models/archive', 'reports']:
    os.makedirs(d, exist_ok=True)


# [ABLA-46] Mọi thành phần mới đều bật/tắt được từ đây — dựng bảng ablation
# mà không phải sửa một dòng code nào.
@dataclass
class Config:
    # ── Nhóm A: chuẩn hoá & đặc trưng ──
    normalize_lexicon: bool = True     # [LEX-20] ★ sửa lỗi nghiêm trọng nhất
    word_segment: bool = True          # [SEG-21]
    apply_intensifier: bool = True     # [LEX-22]
    keep_rating_tokens: bool = True    # [NUM-24]
    quantile_scale: bool = True        # [SCALE-25]
    use_pos: bool = True               # [POS-27] dùng đủ 4 tỉ lệ
    # ── Nhóm A' (v40.4): độ phủ lexicon ──
    normalize_tfidf: bool = True       # [ABLA-55] tách khỏi normalize_lexicon
    seed_lexicon: bool = True          # [LEX-47] 197 mục hạt giống thủ công
    expand_lexicon: bool = True        # [LEX-47] ★ mở rộng bán giám sát §3.2
    lex_expand_top_k: int = 400        # [LEX-47]
    lex_expand_min_df: int = 30        # [LEX-47]
    lex_expand_min_logodds: float = 0.9
    fix_elong: bool = True             # [E1-48]
    fix_tokenizer_sep: bool = True     # [E2-49]
    fix_rating_tfidf: bool = True      # [E4-51]
    fix_dead_columns: bool = True      # [DEAD-52]
    intensifier_max_gap: int = 2       # [INT-54] 0 = vô hạn (hành vi v40.3)
    # ── Nhóm B: kiến trúc ──
    use_decision_layer: bool = True    # [DECIDE-28] ★ lợi ích lớn nhất
    view_word: bool = True             # [STACK-29]
    view_char: bool = True             # [CHAR-34]
    view_nbsvm: bool = True            # [NBSVM-31]
    view_svd_lgbm: bool = True         # [STACK-29]
    view_ordinal: bool = True          # [ORD-32]
    oof_mode: str = 'bagged'           # [STACK-30] 'bagged' | 'refit'
    meta_second_order: bool = True     # [META-33]
    use_cluster_features: bool = True
    use_temperature: bool = True       # [CALIB-35]
    # ── Siêu tham số cố định ──
    char_ngram: tuple = (2, 4)
    char_min_df: int = 3
    char_max_features: int = 60000
    svd_components: int = 200
    decision_bags: int = 15            # [DECIDE-28] chống quá khớp
    decision_shrink: float = 0.7
    decision_w_max: float = 1.5        # [DECIDE-28] chặn biên w
    # ── Tương thích ngược v40.2 ──
    include_svm: bool = True
    include_lgbm: bool = True

    def summary(self) -> str:
        on = [k for k, v in asdict(self).items() if v is True]
        off = [k for k, v in asdict(self).items() if v is False]
        return f"BẬT={on} | TẮT={off} | oof_mode={self.oof_mode}"


CFG = Config()

# ═══════════════════════════════════════════════════════════════════
# TOP‑LEVEL FUNCTIONS (bắt buộc cho pickle trên Windows)
# ═══════════════════════════════════════════════════════════════════
def _sync_cfg(cfg_dict):
    """[PAR-56] CFG là biến global của module. Với backend 'loky', worker là
    một TIẾN TRÌNH KHÁC: nó import lại module và nhận Config() MẶC ĐỊNH, không
    phải CFG mà train_v40() vừa đặt. Vô hại khi mọi cờ đều True (đúng cấu hình
    mặc định), nhưng làm MỌI dòng bảng ablation sai âm thầm. Truyền cfg tường
    minh và đồng bộ một lần cho mỗi worker."""
    global CFG
    if cfg_dict is None:
        return
    if asdict(CFG) != cfg_dict:
        CFG = Config(**cfg_dict)


def _global_extract_features_single(text: str, cfg=None) -> list:
    _sync_cfg(cfg)
    return TextPreprocessor.extract_features(text)

def _global_preprocess_single(text: str, cfg=None) -> str:
    _sync_cfg(cfg)
    return TextPreprocessor.preprocess(text)

def _global_extract_meta_single(text: str, cfg=None) -> list:
    return MetaFeatureExtractor.extract(text)

def _global_pos_single(text: str, cfg=None):
    """[POS-27] Trả về ĐỦ 4 tỉ lệ (v40.2 tính 4 nhưng chỉ dùng 2 — bỏ phí
    ratio_adv, vốn là tỉ lệ tương quan mạnh nhất với cường độ cảm xúc trong
    tiếng Việt). POS chạy trên văn bản ĐÃ chuẩn hoá nhẹ, không phải raw text
    còn nguyên emoji/teencode/URL (v40.2 gán nhãn sai hàng loạt vì điều này)."""
    _sync_cfg(cfg)
    empty = {'ratio_adj': 0.0, 'ratio_adv': 0.0, 'ratio_verb': 0.0, 'ratio_noun': 0.0}
    if not HAS_POS:
        return empty
    try:
        norm = TextPreprocessor.normalize_for_lexicon(text)   # [POS-27]
        tags = pos_tag(norm)
        total = max(len(tags), 1)
        c = Counter([t[1] for t in tags])
        return {'ratio_adj': c.get('A', 0) / total, 'ratio_adv': c.get('R', 0) / total,
                'ratio_verb': c.get('V', 0) / total, 'ratio_noun': c.get('N', 0) / total}
    except Exception:
        return empty

POS_KEYS = ['ratio_adj', 'ratio_adv', 'ratio_verb', 'ratio_noun']

def _pos_to_matrix(pos_cache):
    if pos_cache is None:
        return None
    return np.array([[d[k] for k in POS_KEYS] for d in pos_cache], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# 0. LEXICON MỞ RỘNG & NORMALIZER
# ═══════════════════════════════════════════════════════════════════
class TextNormalizerManager:
    def __init__(self, teencode_map: dict):
        sorted_entries = sorted(teencode_map.items(), key=lambda x: -len(x[0]))
        self._replacement_map = {k.lower(): v for k, v in sorted_entries}
        pattern_parts = [re.escape(k) for k, _ in sorted_entries]
        # (?<!\w)...(?!\w) thay cho \b: an toàn hơn với khoá nhiều từ tiếng Việt
        self._pattern = re.compile(r'(?<!\w)(' + '|'.join(pattern_parts) + r')(?!\w)',
                                   flags=re.IGNORECASE)

    def normalize_text(self, text: str) -> str:
        return self._pattern.sub(
            lambda m: self._replacement_map.get(m.group(0).lower(), m.group(0)), text)

_text_normalizer = None
_text_normalizer_lock = __import__('threading').Lock()

def get_text_normalizer() -> TextNormalizerManager:
    global _text_normalizer
    if _text_normalizer is None:
        with _text_normalizer_lock:
            if _text_normalizer is None:
                _text_normalizer = TextNormalizerManager(TextPreprocessor.TEENCODE_MAP)
    return _text_normalizer


# ═══════════════════════════════════════════════════════════════════
# 1. TEXT PREPROCESSOR NÂNG CAO
# ═══════════════════════════════════════════════════════════════════
class TextPreprocessor:
    EMOJI_MAP = {
        '😊': ' vui ', '😄': ' vui ', '😃': ' vui ', '🙂': ' vui ',
        '😍': ' yêu ', '🥰': ' yêu ', '❤️': ' yêu ', '❤': ' yêu ', '💕': ' yêu ',
        '👍': ' tốt ', '💯': ' hoàn hảo ', '✅': ' tốt ', '🔥': ' tuyệt ',
        '😞': ' buồn ', '😢': ' buồn ', '😭': ' rất buồn ', '☹': ' buồn ',
        '😡': ' bực mình ', '🤬': ' bực mình ', '👎': ' tệ ', '💩': ' rác ',
        '😤': ' bực mình ', '🙄': ' khó chịu ', '😒': ' khó chịu ',
    }

    # [LEX-23] TEENCODE ĐÃ DỌN XUNG ĐỘT với lexicon.
    # Nguyên tắc: chỉ ánh xạ VỀ DẠNG CHUẨN CÓ TRONG LEXICON, tuyệt đối không
    # ánh xạ một từ mạnh về một từ yếu hơn.
    #   v40.2 sai: 'tệ'(NEG=3)→'kém'(NEG=2)  ⇒ mất 1 bậc cường độ
    #              'ok'(POS=1)→'được'        ⇒ 'được' không có trong lexicon
    #              'đỉnh'(POS=3)→'rất tốt'   ⇒ 3 → 2
    # Các lỗi này VÔ HẠI ở v40.2 (lexicon chạy trên text thô nên teencode
    # chẳng bao giờ được áp) nhưng sẽ CÓ HIỆU LỰC ngay khi bật [LEX-20].
    TEENCODE_MAP = {
        # phủ định
        'ko': 'không', 'k': 'không', 'kg': 'không', 'khong': 'không',
        'hok': 'không', 'hem': 'không', 'kh': 'không', 'khg': 'không',
        'chag': 'chẳng', 'chăng': 'chẳng',
        # từ chức năng
        'dc': 'được', 'đc': 'được', 'j': 'gì', 'vs': 'với', 'r': 'rồi',
        'sp': 'sản phẩm', 'ship': 'giao hàng', 'shiper': 'giao hàng',
        'shipper': 'giao hàng', 'ib': 'nhắn tin', 'bt': 'bình thường',
        'bthg': 'bình thường', 'ntn': 'như thế nào', 'mn': 'mọi người',
        # cường độ — KHÔNG ánh xạ về từ cảm xúc, để INTENSIFIERS xử lý
        'vcl': 'cực', 'vl': 'cực', 'vc': 'cực',
        # biến thể KHÔNG DẤU của từ cảm xúc → dạng CÓ DẤU đúng trong lexicon
        'tot': 'tốt', 'dep': 'đẹp', 'nhanh': 'nhanh', 'ben': 'bền',
        'te': 'tệ', 'toi': 'tồi', 'cham': 'chậm', 'hong': 'hỏng',
        'kem': 'kém', 'xau': 'xấu', 'chan': 'chán', 'ngon': 'ngon',
        'that vong': 'thất vọng', 'hai long': 'hài lòng', 'tuyet': 'tuyệt',
        'tuyet voi': 'tuyệt vời', 'chat luong': 'chất lượng',
        'dang mua': 'đáng mua', 'kho chiu': 'khó chịu',
    }

    POSITIVE_WORDS = {
        'tốt': 2, 'tuyệt': 3, 'tuyệt vời': 3, 'tuyệt hảo': 3, 'hay': 2,
        'đẹp': 2, 'thích': 2, 'hài lòng': 2, 'ngon': 2, 'chất lượng': 2,
        'nhanh': 2, 'đỉnh': 3, 'xịn': 2, 'ok': 1, 'ổn': 1, 'bền': 2,
        'xuất sắc': 3, 'hoàn hảo': 3, 'rẻ': 1, 'đáng mua': 2, 'đáng tiền': 2,
        'yêu': 2, 'thân thiện': 2, 'nhiệt tình': 2, 'chu đáo': 2,
        'cẩn thận': 2, 'giá rẻ': 2, 'vừa vặn': 1, 'êm': 1, 'mượt': 2,
        'sắc nét': 2, 'vui': 2, 'rating_5': 3, 'rating_4': 2,
    }
    NEGATIVE_WORDS = {
        'tệ': 3, 'tệ hại': 3, 'xấu': 2, 'tồi': 3, 'chậm': 2, 'hỏng': 3,
        'thất vọng': 3, 'buồn': 2, 'rất buồn': 3, 'chán': 2, 'kém': 2,
        'fake': 3, 'giả': 3, 'rởm': 3, 'lừa': 3, 'lừa đảo': 3, 'rác': 3,
        'đắt': 2, 'lỗi': 2, 'treo': 3, 'bực mình': 3, 'khó chịu': 3,
        'không hài lòng': 2, 'móp': 2, 'méo': 2, 'trầy': 2, 'bể': 3,
        'thiếu': 2, 'sai': 2, 'bẩn': 2, 'hôi': 2, 'ẩu': 2,
        'vô trách nhiệm': 3, 'mất thời gian': 2, 'rating_1': 3, 'rating_2': 2,
    }
    # [LEX-47] ★ Gộp 197 mục hạt giống TMĐT (từ ghép/từ láy) + token score_*
    # [E3-50] + biến thể không dấu nhiều âm tiết. Toàn bộ khối này KHÔNG dùng
    # nhãn ⇒ không rò rỉ, dùng được cho Test và dữ liệu mới.
    # Hai bản gốc v40.3 được giữ nguyên ở trên để bảng ablation còn dựng được
    # (Config.seed_lexicon=False → quay về đúng 70 mục).
    _BASE_POSITIVE = dict(POSITIVE_WORDS)
    _BASE_NEGATIVE = dict(NEGATIVE_WORDS)
    POSITIVE_WORDS, NEGATIVE_WORDS = LEX44.build_extended_lexicon(
        POSITIVE_WORDS, NEGATIVE_WORDS)

    NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'chớ'}
    INTENSIFIERS = {
        'rất': 1.5, 'quá': 1.4, 'cực': 2.0, 'cực kỳ': 2.0, 'siêu': 1.8,
        'vô cùng': 1.8, 'thật': 1.3, 'lắm': 1.4, 'ghê': 1.3,
        'hơi': 0.6, 'khá': 1.2, 'tạm': 0.5, 'tương đối': 0.8, 'chút': 0.5,
    }
    CLAUSE_BOUNDARY = {'.', ',', ';', '!', '?', '...'}
    CONTRAST_MARKERS = {'nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà',
                        'thế mà', 'có điều', 'mỗi tội', 'đáng tiếc'}
    SENTENCE_FINAL_PARTICLES = {'vậy', 'ạ', 'nhỉ', 'nhé', 'à', 'hả', 'thế',
                                'đấy', 'chứ', 'ha', 'hử', 'cơ'}
    LOCATIVE_DAU_BEFORE = {'ở', 'tại', 'đi', 'đến', 'tới', 'về', 'từ', 'sang', 'qua'}
    LOCATIVE_DAU_AFTER = {'rồi', 'nào', 'nữa'}

    # [NUM-24] Mẫu điểm số PHẢI được bắt TRƯỚC khi xoá chữ số.
    RATING_PATTERNS = [
        (re.compile(r'(?<!\w)([1-5])\s*(?:sao|\*|star)(?!\w)', re.I), r' rating_\1 '),
        (re.compile(r'(?<!\w)(10|[0-9])\s*/\s*10(?!\w)'), r' score_\1_10 '),
        (re.compile(r'(?<!\w)([1-5])\s*/\s*5(?!\w)'), r' rating_\1 '),
        (re.compile(r'(?<!\w)([0-9]|10)\s*(?:điểm|diem)(?!\w)', re.I), r' score_\1 '),
    ]
    URL_RE = re.compile(r'http\S+|www\.\S+')
    REPEAT_PUNCT_RE = re.compile(r'([!?.,])\1+')
    ELONG_RE = re.compile(r'(\w)\1{2,}')

    # [E2-49] Dấu phân cách mà v40.3 KHÔNG tách → token dính dấu, trượt lexicon.
    SEP_RE = re.compile(r'[()\[\]{}<>:;/\\"\u201c\u201d\u2018\u2019`*+=|~@#%^&\u2026]')
    HYPHEN_RE = re.compile(r'(?<=\w)[-\u2013\u2014](?=\w)')
    # [E4-51] Token điểm số phải được chuyển sang dạng KHÔNG CHỮ SỐ trước khi
    # preprocess() xoá chữ số, nếu không rating_1..rating_5 bị nghiền thành
    # cùng một token 'rating_'.
    KEEP_NUM_RE = re.compile(r'(?:rating|score)_\d+(?:_\d+)?')
    NUM_WORD = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight',
                '9': 'nine', '10': 'ten'}

    STOPWORDS = {'và', 'của', 'một', 'những', 'các', 'đã', 'có', 'cho', 'về',
                 'từ', 'nếu', 'khi', 'thì', 'rằng', 'lại', 'ra', 'này', 'kia',
                 'đó', 'hay', 'hoặc', 'do', 'vì'}

    _MERGE_PATTERN = None

    # ───────────────────────────────────────────────────────────────
    # [LEX-20] ★ CHUẨN HOÁ DÀNH RIÊNG CHO SO KHỚP LEXICON
    # ───────────────────────────────────────────────────────────────
    @classmethod
    def _normalize_shared(cls, text: str, enabled: bool = True) -> str:
        """Thân chuẩn hoá dùng chung cho CẢ HAI nhánh (lexicon và TF‑IDF).
        `enabled` được truyền vào từ cờ RIÊNG của từng nhánh — xem [ABLA-55]:
        v40.3 để hai nhánh dùng chung một cờ nên bảng ablation bị nhiễu."""
        t = unicodedata.normalize('NFC', str(text))
        for emo, word in cls.EMOJI_MAP.items():
            t = t.replace(emo, word)
        t = cls.URL_RE.sub(' ', t)
        t = t.lower()
        if not enabled:
            return re.sub(r'\s+', ' ', t).strip()       # [ABLA-46] mô phỏng v40.2
        if CFG.keep_rating_tokens:                       # [NUM-24]
            for pat, repl in cls.RATING_PATTERNS:
                t = pat.sub(repl, t)
        # [E1-48] v40.3 rút về HAI ký tự (r'\1\1'): "ngonnnn"→"ngonn",
        # "tốtttt"→"tốtt" — cả hai KHÔNG có trong lexicon nên trượt sạch
        # (2,7% review). Còn phá teencode vì ELONG chạy TRƯỚC nó:
        # "khonggg"→"khongg" ⇒ mục 'khong'→'không' không bao giờ khớp.
        t = cls.ELONG_RE.sub(r'\1' if CFG.fix_elong else r'\1\1', t)
        t = get_text_normalizer().normalize_text(t)      # teencode
        t = cls.REPEAT_PUNCT_RE.sub(r'\1', t)
        return re.sub(r'\s+', ' ', t).strip()

    @classmethod
    def normalize_for_lexicon(cls, text: str) -> str:
        """Khác biệt then chốt so với v40.2: v40.2 chỉ gọi text.lower() trước
        khi so khớp từ điển, nên MỌI biến thể teencode/không dấu trượt sạch
        lexicon. Ở đây chuẩn hoá TRƯỚC, nhưng:
            • GIỮ NGUYÊN dấu câu   → luật CLAUSE_BOUNDARY còn hoạt động
            • GIỮ NGUYÊN stopword  → luật phủ định/tương phản cần chúng
            • KHÔNG phân đoạn từ   → extract_features() tự gộp cụm bằng '_'
        Đây là lý do phải TÁCH khỏi preprocess() (nhánh dùng cho TF‑IDF).

        ⚠ v40.4 — KHÔNG thêm word_tokenize() vào đây. Đã thử: độ phủ GIẢM từ
          0,5420 xuống 0,4759 vì underthesea gộp nhanh→nhanh_chóng,
          chậm→chậm_trễ, đẹp→đẹp_đẽ… mà lexicon không có các từ ghép đó.
          Cách đúng là mở rộng lexicon ([LEX-47]), không phải phân đoạn."""
        return cls._normalize_shared(text, enabled=CFG.normalize_lexicon)

    @classmethod
    def _build_multiword_assets(cls):
        cls._MERGED_POSITIVE = {k.replace(' ', '_'): v for k, v in cls.POSITIVE_WORDS.items()}
        cls._MERGED_NEGATIVE = {k.replace(' ', '_'): v for k, v in cls.NEGATIVE_WORDS.items()}
        cls._MERGED_INTENS = {k.replace(' ', '_'): v for k, v in cls.INTENSIFIERS.items()}
        cls._MERGED_CONTRAST = {m.replace(' ', '_') for m in cls.CONTRAST_MARKERS}
        multiword_keys = sorted(
            {k for k in cls.POSITIVE_WORDS if ' ' in k}
            | {k for k in cls.NEGATIVE_WORDS if ' ' in k}
            | {k for k in cls.INTENSIFIERS if ' ' in k}
            | {m for m in cls.CONTRAST_MARKERS if ' ' in m},
            key=len, reverse=True)
        cls._MERGE_PATTERN = (
            re.compile(r'(?<!\w)(' + '|'.join(re.escape(k) for k in multiword_keys) + r')(?!\w)')
            if multiword_keys else None)

    @classmethod
    def _tokenize_for_lexicon(cls, text_norm: str) -> list:
        if not hasattr(cls, '_MERGED_POSITIVE'):
            cls._build_multiword_assets()
        padded = re.sub(r'(\.\.\.)', r' \1 ', text_norm)
        padded = re.sub(r'([.,;!?])', r' \1 ', padded)
        if CFG.fix_tokenizer_sep:
            # [E2-49] v40.3 chỉ đệm [.,;!?] nên các dấu khác vẫn dính vào
            # token: '(giao', 'sai)', 'chất_lượng:', 'tốt/xấu', 'đẹp-bền-rẻ',
            # '"tuyệt_vời"' — tất cả TRƯỢT lexicon (7,1% review). Xoá thành
            # khoảng trắng (không giữ làm token vì chúng không phải ranh giới
            # mệnh đề như [.,;!?]).
            padded = cls.SEP_RE.sub(' ', padded)
            padded = cls.HYPHEN_RE.sub(' ', padded)
        if cls._MERGE_PATTERN is not None:
            padded = cls._MERGE_PATTERN.sub(lambda m: m.group(0).replace(' ', '_'), padded)
        return padded.split()

    # ───────────────────────────────────────────────────────────────
    # ĐẶC TRƯNG THỦ CÔNG
    # ───────────────────────────────────────────────────────────────
    FEATURE_NAMES = [
        'pos_s', 'neg_s', 'pos_minus_neg', 'sentiment_ratio',
        'intensity_mean', 'n_intensifier', 'n_negation',
        'excl_ratio', 'quest_ratio', 'avg_token_len', 'log_len',
        'last_match_pos_ratio', 'first_match_pos_ratio',
        'is_question', 'is_tag_question', 'pos_s_x_tagq', 'neg_s_x_tagq',
        'n_contrast', 'pos_after_contrast', 'neg_after_contrast',
        'has_lexicon_match', 'is_neutral',
    ]
    IDX_POS_MINUS_NEG = FEATURE_NAMES.index('pos_minus_neg')
    IDX_IS_NEUTRAL = FEATURE_NAMES.index('is_neutral')
    IDX_HAS_MATCH = FEATURE_NAMES.index('has_lexicon_match')
    NEUTRAL_THRESHOLD_DEFAULT = 2.0
    BEFORE_CONTRAST_WEIGHT = 0.5

    @classmethod
    def extract_features(cls, text: str) -> list:
        return cls._score(text)[0]

    @classmethod
    def _score(cls, text: str):
        """Trả về (vector đặc trưng, matched_count). matched_count dùng cho
        [DIAG-45] đo ĐỘ PHỦ LEXICON."""
        norm = cls.normalize_for_lexicon(text)          # ★ [LEX-20]
        tokens = cls._tokenize_for_lexicon(norm)
        n = max(len(tokens), 1)

        pos_s = neg_s = 0.0
        pos_after = neg_after = 0.0
        matched = n_int = n_contrast = 0
        first_idx = last_idx = -1
        int_values = []

        clause_buffer = []
        direct_mode = False
        neg_count = 0
        dist_since_neg = 0
        pending_int = 1.0          # [LEX-22] hệ số intensifier đang chờ áp dụng
        dist_since_int = 0         # [INT-54] khoảng cách kể từ intensifier
        detected_tag_question = False

        def flush_buffer(weight):
            nonlocal pos_s, neg_s
            for is_pos, amount in clause_buffer:
                if is_pos:
                    pos_s += amount * weight
                else:
                    neg_s += amount * weight
            clause_buffer.clear()

        def reset_clause():
            nonlocal neg_count, dist_since_neg, direct_mode, pending_int
            nonlocal dist_since_int
            flush_buffer(1.0)
            neg_count = dist_since_neg = 0
            direct_mode = False
            pending_int = 1.0
            dist_since_int = 0

        for i, tok in enumerate(tokens):
            if tok in cls.CLAUSE_BOUNDARY:
                reset_clause()
                continue
            if tok in cls._MERGED_CONTRAST:
                # [FEAT-5] vế trước giảm nửa trọng số, vế sau giữ nguyên
                flush_buffer(cls.BEFORE_CONTRAST_WEIGHT)
                direct_mode = True
                n_contrast += 1
                neg_count = dist_since_neg = 0
                pending_int = 1.0
                continue
            if tok in cls._MERGED_INTENS:
                # [LEX-22] GIỮ LẠI để nhân vào từ cảm xúc kế tiếp — v40.2 chỉ
                # đếm chúng như túi từ nên "hơi tệ" == "cực tệ".
                if CFG.apply_intensifier:
                    pending_int *= cls._MERGED_INTENS[tok]
                    dist_since_int = 0          # [INT-54]
                int_values.append(cls._MERGED_INTENS[tok])
                n_int += 1
                continue
            if tok in cls.NEGATION_WORDS:
                # [LEXFIX-13] "ở đâu"/"ship đâu rồi" là từ hỏi, không phải phủ định
                if tok == 'đâu':
                    prev_tok = tokens[i - 1] if i > 0 else None
                    next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
                    if (prev_tok in cls.LOCATIVE_DAU_BEFORE) or (next_tok in cls.LOCATIVE_DAU_AFTER):
                        continue
                # [FIX-4/FIX-11] "...không vậy?" / "...không ạ?" là câu hỏi đuôi
                j = i + 1
                while j < len(tokens) and tokens[j] in cls.SENTENCE_FINAL_PARTICLES:
                    j += 1
                if (j >= len(tokens)) or (tokens[j] in cls.CLAUSE_BOUNDARY):
                    if j > i + 1:
                        detected_tag_question = True
                    continue
                neg_count += 1
                dist_since_neg = 0
                continue

            base = None
            is_pos = None
            if tok in cls._MERGED_POSITIVE:
                is_pos, base = True, cls._MERGED_POSITIVE[tok]
            elif tok in cls._MERGED_NEGATIVE:
                is_pos, base = False, cls._MERGED_NEGATIVE[tok]

            if base is not None:
                matched += 1
                last_idx = i
                if first_idx < 0:
                    first_idx = i
                if neg_count == 0:
                    factor = 1.0
                else:
                    decay = max(1.0 - dist_since_neg * 0.15, 0.3)
                    # [FIX-3] lẻ lần phủ định → đảo dấu; chẵn → phục hồi yếu hơn
                    factor = -decay if (neg_count % 2 == 1) else 0.7 * decay
                # [INT-54] v40.3 giữ pending_int cho tới từ cảm xúc kế tiếp
                # BẤT KỂ cách bao xa: "rất nhiều người nói sản phẩm này tốt"
                # → pos_s = 3.0 (1.5×2) trong khi 'rất' bổ nghĩa cho 'nhiều'.
                # Giới hạn tầm với; 0 = vô hạn (giữ hành vi v40.3 cho ablation).
                if CFG.intensifier_max_gap and dist_since_int > CFG.intensifier_max_gap:
                    pending_int = 1.0
                amount = base * factor * pending_int      # ★ [LEX-22]
                pending_int = 1.0
                dist_since_int = 0
                if direct_mode:
                    if is_pos:
                        pos_s += amount
                        pos_after += amount
                    else:
                        neg_s += amount
                        neg_after += amount
                else:
                    clause_buffer.append((is_pos, amount))
            if neg_count > 0:
                dist_since_neg += 1
            if pending_int != 1.0:
                dist_since_int += 1          # [INT-54]

        flush_buffer(1.0)

        raw = str(text)
        raw_len = max(len(raw), 1)
        word_toks = [t for t in tokens if t not in cls.CLAUSE_BOUNDARY]
        pos_minus_neg = pos_s - neg_s
        is_tag_q = 1.0 if detected_tag_question else 0.0
        feats = [
            pos_s, neg_s, pos_minus_neg, matched / n,
            float(np.mean(int_values)) if int_values else 1.0,
            float(n_int), float(neg_count),
            raw.count('!') / raw_len, raw.count('?') / raw_len,
            float(np.mean([len(t) for t in word_toks])) if word_toks else 0.0,
            float(np.log1p(len(raw))),
            (last_idx / n) if last_idx >= 0 else -1.0,
            (first_idx / n) if first_idx >= 0 else -1.0,
            # [DEAD-52] v40.3 dùng raw.strip().endswith('?') → cột này bằng 0 ở
            # 100,0% mẫu (đo trên 20k) vì review TMĐT hầu như không kết thúc
            # bằng '?'. Ba cột phái sinh pos_s_x_tagq/neg_s_x_tagq cũng ~0 theo.
            (1.0 if ('?' in raw) else 0.0) if CFG.fix_dead_columns
            else (1.0 if raw.strip().endswith('?') else 0.0),
            is_tag_q, pos_s * is_tag_q, neg_s * is_tag_q,
            float(n_contrast), pos_after, neg_after,
            1.0 if matched else 0.0,
            # is_neutral — ngưỡng mặc định, bị ghi đè bằng T dò trên TRAIN [CAL-36]
            1.0 if abs(pos_minus_neg) < cls.NEUTRAL_THRESHOLD_DEFAULT else 0.0,
        ]
        return feats, matched

    # ───────────────────────────────────────────────────────────────
    # TIỀN XỬ LÝ CHO TF‑IDF
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def _segment(text: str) -> str:
        """[SEG-21] BUG GỐC trong v40.2:
               tokens = word_tokenize(text)     # → ['chất lượng', 'tốt']
               return ' '.join(tokens)          # → 'chất lượng tốt'  ← Y HỆT ĐẦU VÀO
           underthesea trả về list các từ ghép CHỨA DẤU CÁCH nên ' '.join()
           huỷ sạch kết quả phân đoạn. Phải dùng format="text"."""
        if not (HAS_UNDER and CFG.word_segment):
            return text
        try:
            return word_tokenize(text, format='text')
        except Exception:
            return text

    @staticmethod
    def _digitfree_rating(match) -> str:
        """[E4-51] rating_5 → ratingfive ; score_10_10 → scoretenten.
        Giữ được sự PHÂN BIỆT giữa "1 sao" và "5 sao" ở nhánh TF‑IDF mà không
        cần chữ số — nếu để chữ số thì re.sub(r'\\d+') ngay sau đó sẽ nghiền
        tất cả thành cùng một token 'rating_'."""
        parts = match.group(0).split('_')
        return parts[0] + ''.join(TextPreprocessor.NUM_WORD.get(p, p)
                                  for p in parts[1:])

    @classmethod
    def preprocess(cls, text: str) -> str:
        # [ABLA-55] v40.3 gọi trực tiếp normalize_for_lexicon(), nên khi tắt
        # Config.normalize_lexicon để dựng bảng ablation thì nhánh TF‑IDF CŨNG
        # mất teencode/ELONG/rating token ⇒ dòng ablation đó đo HAI thay đổi
        # cùng lúc. Giờ nhánh TF‑IDF có cờ riêng normalize_tfidf.
        t = cls._normalize_shared(text, enabled=CFG.normalize_tfidf)
        if CFG.fix_rating_tfidf:
            # [E4-51] ⛔ v40.3: re.sub(r'\d+',' ') khớp luôn chữ số TRONG
            # 'rating_5' ⇒ rating_1..rating_5 thành CÙNG token 'rating_'
            # (1.484/20.000 mẫu). Comment cũ ghi "rating_* đã được giữ" — sai.
            t = cls.KEEP_NUM_RE.sub(cls._digitfree_rating, t)
        t = re.sub(r'\d+', ' ', t)               # số còn lại
        t = cls._segment(t)                      # [SEG-21]
        toks = t.split()
        return ' '.join(w for w in toks if w not in cls.STOPWORDS and len(w) > 1)

    # ───────────────────────────────────────────────────────────────
    # BATCH
    # ───────────────────────────────────────────────────────────────
    @classmethod
    def _run_batch(cls, fn, raw_texts, sequential=False):
        """[PAR-56] Truyền asdict(CFG) tường minh: worker của backend 'loky' là
        một TIẾN TRÌNH KHÁC, nó import lại module nên nhận Config() MẶC ĐỊNH
        chứ không phải CFG mà train_v40() vừa đặt. Với cấu hình mặc định (mọi
        cờ True) điều đó vô hại, nhưng làm MỌI dòng bảng ablation sai âm thầm.
        `sequential=True` cho các hàm phụ thuộc TRẠNG THÁI ĐỘNG của lớp (lexicon
        đã mở rộng theo fold) — trạng thái đó không thể lan sang tiến trình con."""
        cfg = asdict(CFG)
        if sequential:
            return [fn(t, cfg) for t in raw_texts]
        try:
            with joblib.parallel_config(backend=JOBLIB_BACKEND, n_jobs=N_PARALLEL_PRE):
                return Parallel(n_jobs=N_PARALLEL_PRE)(
                    delayed(fn)(t, cfg) for t in raw_texts)
        except Exception:
            return [fn(t, cfg) for t in raw_texts]

    @classmethod
    def preprocess_batch(cls, raw_texts: list) -> list:
        return cls._run_batch(_global_preprocess_single, raw_texts)

    @classmethod
    def extract_features_batch(cls, raw_texts: list) -> np.ndarray:
        # [PAR-56] BẮT BUỘC tuần tự: sau [LEX-47] lexicon được mở rộng bằng nhãn
        # của TRAIN từng seed, tức là trạng thái ĐỘNG của lớp. Tiến trình con của
        # loky sẽ import lại module và dùng lexicon GỐC ⇒ đặc trưng thủ công của
        # Train/Cal/Test tính từ hai từ điển khác nhau mà không báo lỗi.
        # Chi phí: đo được 6,6s cho toàn bộ 68.661 mẫu — không cần song song.
        return np.array(cls._run_batch(_global_extract_features_single, raw_texts,
                                       sequential=True), dtype=np.float32)

    @classmethod
    def extract_pos_features_batch(cls, raw_texts: list):
        return cls._run_batch(_global_pos_single, raw_texts)

    @classmethod
    def lexicon_coverage(cls, raw_texts: list) -> dict:
        """[DIAG-45] ĐỘ PHỦ LEXICON — chỉ số quan trọng nhất để trả lời câu hỏi
        'ma trận thưa còn hạn chế ở đâu'. Nếu < 60% thì 9/22 cột thủ công phái
        sinh từ matched_count đang gần như là ma trận 0."""
        counts = np.array([cls._score(t)[1] for t in raw_texts])
        return {'coverage_rate': float((counts > 0).mean()),
                'mean_matches': float(counts.mean()),
                'p90_matches': float(np.percentile(counts, 90))}

    # ───────────────────────────────────────────────────────────────
    # [LEX-47] MỞ RỘNG LEXICON THEO TỪNG SEED
    # ───────────────────────────────────────────────────────────────
    @classmethod
    def lexicon_snapshot(cls) -> tuple:
        """Ảnh chụp lexicon TĨNH (hạt giống thủ công, không dùng nhãn)."""
        return (dict(cls.POSITIVE_WORDS), dict(cls.NEGATIVE_WORDS))

    @classmethod
    def restore_lexicon(cls, snapshot: tuple):
        """Gọi ở ĐẦU mỗi seed. Nếu không, seed n+1 thừa hưởng từ điển đã học
        từ tập Train của seed n — mỗi seed có split khác nhau nên đó là rò rỉ
        chéo seed, và nó làm hỏng đúng điều kiện của paired test."""
        cls.POSITIVE_WORDS = dict(snapshot[0])
        cls.NEGATIVE_WORDS = dict(snapshot[1])
        cls._rebuild_lexicon_assets()

    @classmethod
    def _rebuild_lexicon_assets(cls):
        for attr in ('_MERGED_POSITIVE', '_MERGED_NEGATIVE',
                     '_MERGED_INTENS', '_MERGED_CONTRAST'):
            if attr in cls.__dict__:
                delattr(cls, attr)
        cls._build_multiword_assets()

    @classmethod
    def expand_lexicon_from_train(cls, raw_train: list, labels_train: list) -> dict:
        """[LEX-47] ★ §3.2 của tài liệu phân tích — phần chưa từng được
        implement trong v40.3. Mở rộng lexicon bằng delta log‑odds có làm trơn
        trên CHÍNH tập Train.

        ⚠ HÀM NÀY DÙNG NHÃN. Bắt buộc gọi SAU split_data_v40() và CHỈ với dữ
          liệu Train. Gọi trên toàn bộ dataframe là rò rỉ nhãn sang Test."""
        if not hasattr(cls, '_MERGED_POSITIVE'):
            cls._build_multiword_assets()
        toks = (cls._tokenize_for_lexicon(cls.normalize_for_lexicon(t))
                for t in raw_train)
        already = (set(cls._MERGED_POSITIVE) | set(cls._MERGED_NEGATIVE)
                   | set(cls._MERGED_INTENS) | set(cls.NEGATION_WORDS))
        add_pos, add_neg, diag = LEX44.delta_logodds_expand(
            toks, labels_train, already=already,
            min_df=CFG.lex_expand_min_df,
            min_abs_logodds=CFG.lex_expand_min_logodds,
            top_k=CFG.lex_expand_top_k)
        pos = dict(cls.POSITIVE_WORDS); pos.update(add_pos)
        neg = dict(cls.NEGATIVE_WORDS); neg.update(add_neg)
        cls.POSITIVE_WORDS = pos
        cls.NEGATIVE_WORDS = neg
        cls._rebuild_lexicon_assets()
        diag['lexicon_size'] = len(pos) + len(neg)
        return diag


# [LEX-47] Hai ảnh chụp lexicon TĨNH, lấy MỘT LẦN lúc import — trước khi bất
# kỳ seed nào chạy. Phần mở rộng bằng nhãn (theo từng seed) luôn được đặt lên
# một trong hai ảnh này, không bao giờ cộng dồn qua các seed.
_LEXICON_SEED = TextPreprocessor.lexicon_snapshot()          # 267 mục (hạt giống)
_LEXICON_V403 = (dict(TextPreprocessor._BASE_POSITIVE),      # 70 mục (v40.3)
                 dict(TextPreprocessor._BASE_NEGATIVE))


def _active_lexicon_snapshot():
    """[ABLA-46] Config.seed_lexicon=False → quay về đúng 70 mục của v40.3."""
    return _LEXICON_SEED if CFG.seed_lexicon else _LEXICON_V403


# ═══════════════════════════════════════════════════════════════════
# 2. META‑FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════
class MetaFeatureExtractor:
    @classmethod
    def extract(cls, text: str) -> list:
        t = str(text).lower().split()
        n = max(len(t), 1)
        return [float(np.log1p(len(str(text)))), float(np.log1p(n)), len(set(t)) / n]

    @classmethod
    def extract_batch(cls, raw_texts: list) -> np.ndarray:
        return np.array(TextPreprocessor._run_batch(_global_extract_meta_single, raw_texts),
                        dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# 3. CÁC GÓC NHÌN ĐẶC TRƯNG  — [STACK-29][CHAR-34][NBSVM-31]
# ═══════════════════════════════════════════════════════════════════
class ManualScaler:
    """[SCALE-25] Thay MinMaxScaler.
    pos_s/neg_s có phân bố ĐUÔI DÀI; MinMax fit trên train khiến ~99% mẫu bị
    nén về [0,0.05] chỉ vì một review dài bất thường — rồi [SCALE-14] phải
    clip [0,1] để che giấu hậu quả. Biến đổi theo THỨ HẠNG miễn nhiễm outlier
    và luôn cho đầu ra trong [0,1] kể cả trên Test."""

    def __init__(self, seed=42):
        self.seed = seed
        self.scaler = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if CFG.quantile_scale:
            self.scaler = QuantileTransformer(
                n_quantiles=min(1000, max(10, X.shape[0])),
                output_distribution='uniform', subsample=200_000,
                random_state=self.seed)
        else:
            from sklearn.preprocessing import MinMaxScaler   # [ABLA-46] hành vi v40.2
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return np.clip(self.scaler.transform(np.asarray(X, dtype=np.float64)), 0.0, 1.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class NBFeatureWeighter:
    """[NBSVM-31] Trọng số log‑count‑ratio cho bài toán NHỊ PHÂN.
        p = α + Σ_{y=1} x ,  q = α + Σ_{y=0} x
        r = log( (p/‖p‖₁) / (q/‖q‖₁) ) ,  X' = X ⊙ r
    r_j là log tỉ số hợp lý của đặc trưng j dưới NB đa thức; nhân vào TF‑IDF
    tương đương khởi tạo mô hình tuyến tính tại nghiệm NB rồi để LR tinh chỉnh
    — chính quy hoá có tri thức tiên nghiệm, mạnh nhất khi mẫu/chiều nhỏ."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y_binary):
        Xb = csr_matrix(X).copy()
        Xb.data = np.ones_like(Xb.data)
        y = np.asarray(y_binary).ravel()
        p = np.asarray(Xb[y == 1].sum(axis=0)).ravel() + self.alpha
        q = np.asarray(Xb[y == 0].sum(axis=0)).ravel() + self.alpha
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        return self

    def transform(self, X):
        Xb = csr_matrix(X).copy()
        Xb.data = np.ones_like(Xb.data)
        return Xb.multiply(self.r_).tocsr()


class NBLinearClassifier(BaseEstimator, ClassifierMixin):
    """[NBSVM-31] Mở rộng NBSVM sang đa lớp theo lối một‑đối‑phần‑còn‑lại.
    Mỗi bài toán con có vector r RIÊNG nên số chiều vẫn là d (không phải 3d)."""

    def __init__(self, C=1.0, alpha=1.0, seed=42):
        self.C, self.alpha, self.seed = C, alpha, seed

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.weighters_, self.models_ = [], []
        for c in self.classes_:
            yb = (y == c).astype(int)
            w = NBFeatureWeighter(self.alpha).fit(X, yb)
            m = LogisticRegression(C=self.C, max_iter=1000, class_weight='balanced',
                                   random_state=self.seed)
            m.fit(w.transform(X), yb)
            self.weighters_.append(w)
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        P = np.column_stack([m.predict_proba(w.transform(X))[:, 1]
                             for w, m in zip(self.weighters_, self.models_)])
        return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)


class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """[ORD-32] Khai thác cấu trúc THỨ TỰ negative < neutral < positive.
    Thay vì 1 bài 3 lớp phẳng, giải C−1 = 2 bài NHỊ PHÂN tích luỹ:
        M₁: P(y>negative)      M₂: P(y>neutral)
    rồi tái hợp:
        P(neg)=1−P(y>neg) ; P(neu)=P(y>neg)−P(y>neu) ; P(pos)=P(y>neu)
    Vì sao đúng cho bài toán này: 100% khối lượng lỗi nằm ở ranh giới neutral
    (recall 0.815 so với 0.870/0.908). Ở dạng phẳng neutral phải tự bảo vệ
    đồng thời trước cả hai phía; ở dạng thứ tự nó là GIAO của hai quyết định
    đơn giản hơn, mỗi bộ nhị phân lại được train trên bài toán cân bằng hơn."""

    def __init__(self, C=1.0, seed=42):
        self.C, self.seed = C, seed

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.models_ = []
        for k in range(len(self.classes_) - 1):
            yb = (y > self.classes_[k]).astype(int)
            m = LogisticRegression(C=self.C, class_weight='balanced',
                                   max_iter=2000, random_state=self.seed)
            m.fit(X, yb)
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        cum = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        cum = np.minimum.accumulate(cum, axis=1)      # ép đơn điệu giảm
        P = np.zeros((cum.shape[0], len(self.classes_)))
        P[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, len(self.classes_) - 1):
            P[:, k] = cum[:, k - 1] - cum[:, k]
        P[:, -1] = cum[:, -1]
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)


class FeatureView:
    """Một 'góc nhìn': bộ trích đặc trưng + learner riêng.

    [STACK-29] v40.2 cho cả 3 learner nhìn CÙNG một ma trận nên độ bất đồng
    Ā ≈ 0 và stacking gần như không sinh lợi (Krogh–Vedelsby). Ở đây mỗi view
    là một KHÔNG GIAN KHÁC NHAU:
        word  — từ vựng      (LogisticRegression)
        char  — hình thái    (LinearSVC + Platt)  → chịu OOV/sai chính tả
        nbsvm — NB‑weighted  (NBLinearClassifier)
        svd   — ngữ nghĩa dày(LightGBM)           → cây mới đúng thế mạnh
        ord   — thứ tự       (Frank–Hall)
    """

    def __init__(self, name, seed=42, **params):
        self.name = name
        self.seed = seed
        self.params = params
        self.vectorizer = None
        self.extra_scaler = None
        self.model = None

    # ── xây ma trận đặc trưng của view ──
    def fit_transform_features(self, data, y):
        p = self.params
        if self.name == 'word':
            self.vectorizer = TfidfVectorizer(
                ngram_range=p.get('ngram', (1, 2)),
                min_df=p.get('min_df', 2), max_df=0.9,          # [TFIDF-26]
                max_features=p.get('max_features', 40000),
                sublinear_tf=p.get('sublinear_tf', True))
            return self.vectorizer.fit_transform(data['clean'])
        if self.name == 'char':
            # [CHAR-34] TF‑IDF thay Hashing: có IDF, có min_df, diễn giải được
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb', ngram_range=CFG.char_ngram,
                min_df=CFG.char_min_df, max_features=CFG.char_max_features,
                sublinear_tf=True)
            return self.vectorizer.fit_transform(data['clean'])
        if self.name == 'nbsvm':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=p.get('min_df', 2),
                max_features=p.get('max_features', 40000), sublinear_tf=True)
            return self.vectorizer.fit_transform(data['clean'])
        if self.name == 'ord':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=p.get('min_df', 2),
                max_features=p.get('max_features', 40000), sublinear_tf=True)
            Xt = self.vectorizer.fit_transform(data['clean'])
            self.extra_scaler = ManualScaler(self.seed).fit(data['manual'])
            return sp_hstack([Xt, csr_matrix(self.extra_scaler.transform(data['manual']))],
                             format='csr')
        if self.name == 'svd':
            # [STACK-29] LGBM trên ma trận thưa 52k chiều gần như vô dụng (cây
            # chia theo từng đặc trưng, mật độ <0.1%). Đưa về KHÔNG GIAN DÀY.
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                              sublinear_tf=True, max_features=40000)
            Xt = self.vectorizer.fit_transform(data['clean'])
            n_comp = min(CFG.svd_components, max(2, min(Xt.shape) - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            Z = self.svd.fit_transform(Xt)
            self.extra_scaler = ManualScaler(self.seed).fit(data['manual'])
            parts = [Z, self.extra_scaler.transform(data['manual']), data['meta']]
            if data.get('pos') is not None:
                parts.append(data['pos'])
            return np.hstack(parts)
        raise ValueError(f"View không xác định: {self.name}")

    def transform_features(self, data):
        if self.name in ('word', 'char', 'nbsvm'):
            return self.vectorizer.transform(data['clean'])
        if self.name == 'ord':
            Xt = self.vectorizer.transform(data['clean'])
            return sp_hstack([Xt, csr_matrix(self.extra_scaler.transform(data['manual']))],
                             format='csr')
        if self.name == 'svd':
            Z = self.svd.transform(self.vectorizer.transform(data['clean']))
            parts = [Z, self.extra_scaler.transform(data['manual']), data['meta']]
            if data.get('pos') is not None:
                parts.append(data['pos'])
            return np.hstack(parts)
        raise ValueError(f"View không xác định: {self.name}")

    # ── learner của view ──
    def build_model(self, y, groups=None):
        p = self.params
        C = p.get('C', 1.0)
        if self.name == 'word':
            return LogisticRegression(C=C, class_weight='balanced',
                                      max_iter=1000, random_state=self.seed)
        if self.name == 'char':
            base = LinearSVC(C=p.get('svm_C', 1.0), class_weight='balanced',
                             random_state=self.seed, max_iter=3000)
            cv = self._group_folds(y, groups, n=3)     # [GROUPCV-10]
            return CalibratedClassifierCV(base, method='sigmoid', cv=cv)
        if self.name == 'nbsvm':
            return NBLinearClassifier(C=C, seed=self.seed)
        if self.name == 'ord':
            return FrankHallOrdinalClassifier(C=C, seed=self.seed)
        if self.name == 'svd':
            return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  class_weight='balanced', random_state=self.seed, verbose=-1)
        raise ValueError(f"View không xác định: {self.name}")

    def _group_folds(self, y, groups, n=3):
        """[GROUPCV-10] group‑aware hoá CalibratedClassifierCV bằng cách truyền
        thẳng danh sách fold, không phụ thuộc API metadata‑routing."""
        if groups is None:
            return n
        try:
            n_splits = min(n, max(2, len(np.unique(groups))))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=self.seed)
            return list(sgkf.split(np.zeros(len(y)), y, groups))
        except Exception as e:
            logger.warning(f"[GROUPCV-10] StratifiedGroupKFold lỗi cho view '{self.name}' "
                           f"({e}) — fallback cv={n} (không group-aware).")
            return n

    def fit(self, data, y, groups=None):
        X = self.fit_transform_features(data, y)
        self.n_features_ = X.shape[1]
        self.model = self.build_model(y, groups)
        self.model.fit(X, y)
        return self

    def predict_proba(self, data):
        return self.model.predict_proba(self.transform_features(data))


def build_view_specs(best_params, seed):
    """Danh sách view sẽ tham gia stack, theo cờ trong Config."""
    specs = []
    common = dict(ngram=best_params.get('ngram_range', (1, 2)),
                  min_df=best_params.get('min_df', 2),
                  max_features=best_params.get('tfidf_max_feat', 40000),
                  sublinear_tf=best_params.get('sublinear_tf', True),
                  C=best_params.get('lr_C', 4.0))
    if CFG.view_word:
        specs.append(('word', common))
    if CFG.view_char and CFG.include_svm:
        specs.append(('char', dict(common, svm_C=1.0)))
    if CFG.view_nbsvm:
        specs.append(('nbsvm', common))
    if CFG.view_ordinal:
        specs.append(('ord', common))
    if CFG.view_svd_lgbm and CFG.include_lgbm and HAS_LGBM:
        specs.append(('svd', common))
    if not specs:
        raise ValueError("Không có view nào được bật — kiểm tra Config.")
    return [FeatureView(name, seed=seed, **params) for name, params in specs]


# ═══════════════════════════════════════════════════════════════════
# 4. BAGGED MULTI-VIEW STACKING  — [STACK-30]
# ═══════════════════════════════════════════════════════════════════
class MultiViewStack:
    """Sinh OOF cho Train và dự đoán cho Cal/Test.

    ★ Điểm sửa quan trọng nhất so với v40.2:
        v40.2:  meta_train ← mô hình fold (train 80%, TỪ VỰNG RIÊNG của fold)
                meta_test  ← mô hình fit lại trên 100% train (từ vựng khác hẳn)
                ⇒ meta học phân bố A rồi được áp lên phân bố B.
        v40.3:  meta_test ← TRUNG BÌNH K mô hình fold (cùng loại mô hình đã
                sinh ra meta_train) ⇒ hai phân bố KHỚP nhau. K mô hình fold
                vốn đã được huấn luyện rồi — v40.2 đơn giản là vứt đi.
    Đặt Config.oof_mode='refit' để quay lại hành vi v40.2 (model nhỏ hơn ~5×).
    """

    def __init__(self, best_params, n_folds=N_OOF_FOLDS, seed=42):
        # Lưu THAM SỐ chứ không phải closure: một lambda/hàm lồng nhau sẽ khiến
        # joblib.dump() ném PicklingError khi lưu bundle.
        self.best_params = dict(best_params)
        self.n_folds = n_folds
        self.seed = seed
        self.fold_views_ = []      # [bagged] list[list[FeatureView]]
        self.full_views_ = []      # [refit]  list[FeatureView]

    def _folds(self, y, groups):
        if groups is not None:
            cv = StratifiedGroupKFold(self.n_folds, shuffle=True, random_state=self.seed)
            return list(cv.split(np.zeros(len(y)), y, groups))
        cv = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        return list(cv.split(np.zeros(len(y)), y))

    @staticmethod
    def _slice(data, idx):
        out = {}
        for k, v in data.items():
            if v is None:
                out[k] = None
            elif isinstance(v, np.ndarray):
                out[k] = v[idx]
            else:
                out[k] = [v[i] for i in idx]
        return out

    def fit_transform(self, data_train, y_train, eval_datas, groups_train=None):
        """Trả về (oof_train, [pred_eval...]) — mỗi cái là ma trận
        (n, n_views × n_classes)."""
        y_train = np.asarray(y_train)
        n_cls = len(np.unique(y_train))
        folds = self._folds(y_train, groups_train)
        probe = build_view_specs(self.best_params, self.seed)
        self.view_names_ = [v.name for v in probe]
        n_views = len(self.view_names_)

        oof = np.zeros((len(y_train), n_views * n_cls))
        acc = [np.zeros((len(d['clean']), n_views * n_cls)) for d in eval_datas]

        for fi, (tr_idx, va_idx) in enumerate(folds, 1):
            t0 = time.time()
            d_tr = self._slice(data_train, tr_idx)
            d_va = self._slice(data_train, va_idx)
            g_tr = groups_train[tr_idx] if groups_train is not None else None
            views = build_view_specs(self.best_params, self.seed)
            for vi, view in enumerate(views):
                sl = slice(vi * n_cls, (vi + 1) * n_cls)
                view.fit(d_tr, y_train[tr_idx], g_tr)
                oof[va_idx, sl] = view.predict_proba(d_va)
                if CFG.oof_mode == 'bagged':
                    for k, d in enumerate(eval_datas):
                        acc[k][:, sl] += view.predict_proba(d)
            if CFG.oof_mode == 'bagged':
                self.fold_views_.append(views)
            logger.debug(f"  OOF fold {fi}/{len(folds)}: {n_views} view "
                         f"({', '.join(self.view_names_)}) — {time.time()-t0:.1f}s")

        if CFG.oof_mode == 'bagged':
            K = len(folds)
            eval_mats = [a / K for a in acc]
        else:
            # Hành vi v40.2: fit lại trên 100% train rồi dự đoán cal/test.
            logger.info("  [STACK-30] oof_mode='refit' — fit lại view trên toàn bộ Train "
                        "(nhỏ hơn nhưng meta_train/meta_test LỆCH phân bố).")
            self.full_views_ = build_view_specs(self.best_params, self.seed)
            for view in self.full_views_:
                view.fit(data_train, y_train, groups_train)
            eval_mats = [np.hstack([v.predict_proba(d) for v in self.full_views_])
                         for d in eval_datas]
        return oof, eval_mats

    def predict_proba_block(self, data):
        """Dùng ở inference — tái tạo ĐÚNG cách meta_test được sinh lúc train."""
        if CFG.oof_mode == 'bagged' or self.fold_views_:
            n = len(data['clean'])
            out = None
            for views in self.fold_views_:
                block = np.hstack([v.predict_proba(data) for v in views])
                out = block if out is None else out + block
            return out / len(self.fold_views_)
        return np.hstack([v.predict_proba(data) for v in self.full_views_])


def build_meta_features(prob_block, n_views, n_classes, manual_scaled=None, onehot=None):
    """[META-33] Bổ sung đặc trưng BẬC HAI của xác suất.

    Meta của v40.2 là LogisticRegression TUYẾN TÍNH trên các cột xác suất thô
    nên không thể tự học được các đại lượng phi tuyến quan trọng nhất của
    stacking. Bốn nhóm dưới đây rẻ và thường cho +0.3…+0.8 F1:
        margin       = p₍₁₎ − p₍₂₎     — độ chắc chắn
        entropy      = −Σ p log p       — độ bất định
        disagreement = std giữa các view — khi nào nên tin ensemble
        polarity     = p_pos − p_neg    — TRỤC THỨ TỰ của bài toán
    """
    parts = [prob_block]
    if CFG.meta_second_order:
        P = prob_block.reshape(len(prob_block), n_views, n_classes)
        mean_p = P.mean(axis=1)
        srt = np.sort(mean_p, axis=1)
        margin = (srt[:, -1] - srt[:, -2]).reshape(-1, 1)
        entropy = (-(mean_p * np.log(np.clip(mean_p, 1e-12, None))).sum(axis=1)).reshape(-1, 1)
        disagree = P.std(axis=1)
        polarity = (mean_p[:, -1] - mean_p[:, 0]).reshape(-1, 1)
        parts += [mean_p, margin, entropy, disagree, polarity]
    if onehot is not None and onehot.shape[1] > 0:
        parts.append(onehot)
    if manual_scaled is not None:
        parts.append(manual_scaled)          # [ARCH-7]
    return np.hstack(parts)


# ═══════════════════════════════════════════════════════════════════
# 5. META CLUSTERER, META‑LEARNER, HIỆU CHỈNH, LỚP QUYẾT ĐỊNH
# ═══════════════════════════════════════════════════════════════════
class MetaClusterer:
    def __init__(self, n_clusters=4, seed=42):
        self.nc = n_clusters
        self.sc = StandardScaler()
        self.km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)

    def fit_predict(self, X):
        return self.km.fit_predict(self.sc.fit_transform(X))

    def transform(self, X):
        return self.km.predict(self.sc.transform(X))

    def to_onehot(self, cids):
        oh = np.zeros((len(cids), self.nc), dtype=np.float32)
        oh[np.arange(len(cids)), cids] = 1.0
        return oh


class StackingMetaLearner:
    def __init__(self, C=1.0, seed=42):
        self.model = LogisticRegression(C=C, class_weight='balanced',
                                        max_iter=2000, random_state=seed)

    def fit(self, meta_train, y):
        self.model.fit(meta_train, y)
        return self

    def predict_proba(self, meta_test):
        return self.model.predict_proba(meta_test)


class TemperatureScaler:
    """[CALIB-35] Temperature scaling — ĐÚNG MỘT tham số.

    v40.2 dùng LogisticRegression trên log‑odds, tức VECTOR SCALING với ma
    trận 3×3 + bias = 12 tham số, fit trên tập Cal vốn đã bị chia sẻ cho 3
    mục đích khác. Sai về đánh đổi độ chệch–phương sai, và log đã cho thấy
    hậu quả: ECE‑Test khi ÁP DỤNG = 0.0263 (n=6) còn TỆ HƠN khi LOẠI = 0.0237
    (n=4).  p' = softmax(log p / T).  (Guo et al., ICML 2017)

    Ghi chú: temperature scaling ĐƠN ĐIỆU nên KHÔNG đổi argmax — nó chỉ sửa
    độ tự tin. Vì vậy nó độc lập hoàn toàn với lớp quyết định bên dưới."""

    def __init__(self):
        self.T_ = 1.0

    @staticmethod
    def _apply(logp, T):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, proba, y):
        logp = np.log(np.clip(proba, 1e-12, 1.0))
        y = np.asarray(y).ravel()

        def nll(logT):
            p = self._apply(logp, np.exp(logT))
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method='bounded')
        self.T_ = float(np.exp(res.x))
        return self

    def predict_proba(self, proba):
        return self._apply(np.log(np.clip(proba, 1e-12, 1.0)), self.T_)


def accept_temperature_cv(proba_cal, y_cal, n_splits=5, seed=42):
    """[CALIB-35] Quyết định NHẬN/LOẠI bằng CV NỘI BỘ trên tập Cal.
    v40.2 fit VÀ đánh giá trên CÙNG tập ⇒ quyết định overfit."""
    y_cal = np.asarray(y_cal).ravel()
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    before, after = [], []
    for tr, va in cv.split(proba_cal, y_cal):
        ts = TemperatureScaler().fit(proba_cal[tr], y_cal[tr])
        before.append(ece_equal_mass(y_cal[va], proba_cal[va]))
        after.append(ece_equal_mass(y_cal[va], ts.predict_proba(proba_cal[va])))
    e0, e1 = float(np.mean(before)), float(np.mean(after))
    return (e1 < e0), TemperatureScaler().fit(proba_cal, y_cal), e0, e1


class MacroF1DecisionLayer:
    """[DECIDE-28] ★ Quy tắc plug‑in tối đa macro‑F1:  ŷ = argmax_c w_c·p_c(x)

    CƠ SỞ LÝ THUYẾT: np.argmax thuần tối ưu ACCURACY chứ không tối ưu
    MACRO‑F1. Với các độ đo tuyến tính‑phân thức như F‑measure, bộ phân loại
    tối ưu Bayes có dạng NGƯỠNG trên xác suất hậu nghiệm (Koyejo et al.,
    NeurIPS 2014; Narasimhan et al., ICML 2014). Với đa lớp, dạng ngưỡng đó
    tương đương một vector trọng số w ∈ R^C₊ áp lên xác suất trước argmax.

    BẰNG CHỨNG TỪ LOG 10 SEED CỦA CHÍNH DỰ ÁN NÀY:
        neutral   P=0.869  R=0.815  →  P−R = +0.054   (dự đoán THIẾU)
        negative  P=0.837  R=0.870  →  P−R = −0.033   (dự đoán THỪA)
    Lệch CÙNG DẤU ở 10/10 seed ⇒ hệ thống, không phải nhiễu.

    ⚠ CHỐNG QUÁ KHỚP — BẮT BUỘC: leo dốc toạ độ "trần" trên ~5.000 mẫu Cal
    quá khớp nặng và thường VƯỢT QUÁ mức hiệu chỉnh cần thiết. Kiểm chứng
    trên mô phỏng đúng chế độ lỗi này (w đúng ≈ 1.8):
        leo dốc trần    w_neu = 2.181  →  ΔF1 = +0.0013
        bagging+co ngót w_neu = 1.615  →  ΔF1 = +0.0093
    Đó chính là lỗi [CAL-36] mà ta phê phán ở v40.2 — không được lặp lại.

    BA LỚP BẢO VỆ:
      1. BAGGING — ước lượng w trên B mẫu bootstrap, lấy trung bình HÌNH HỌC
         (trung bình của log w). Giảm phương sai ước lượng.
      2. CO NGÓT — w = exp(λ·mean_b log w_b), kéo w về phía 1 (λ=0.7).
      3. KIỂM CHỨNG CV GIỮ RIÊNG CÓ TIÊU CHÍ THỐNG KÊ — điểm mấu chốt.
         Ước lượng độ lợi THẬT bằng K-fold trên Cal (fit w trên K−1 fold, đo
         ΔF1 trên fold còn lại), rồi CHỈ áp dụng nếu
                 mean(ΔF1) − SE(ΔF1) > min_cv_gain
         tức độ lợi phải vượt sai số chuẩn của chính nó, không chỉ dương.
         (Kiểm tra "w có cải thiện trên chính tập đã fit không" là VÔ NGHĨA —
          nó luôn đúng theo định nghĩa vì đó chính là hàm mục tiêu. Bản thử
          nghiệm đầu của v40.3 mắc đúng lỗi này: +0.0255 trên Cal, −0.0131
          trên Test. Ngay cả CV trung bình dương cũng chưa đủ khi Cal nhỏ:
          lần thử thứ hai cho CV +0.0183 mà Test vẫn −0.0131. Cùng tinh thần
          với accept_temperature_cv ở trên.)
      4. CHẶN BIÊN w ∈ [1/w_max, w_max] — giới hạn thiệt hại tối đa nếu Cal
         và Test bất đồng. Không có chặn này, w có thể rơi xuống 0.61 và bóp
         nghẹt cả một lớp.
    """

    def __init__(self, n_rounds=4, n_bags=None, shrink=None, min_gain=1e-3,
                 n_cv=5, min_cv_gain=0.0, w_max=None, seed=42):
        self.n_rounds = n_rounds
        self.n_bags = CFG.decision_bags if n_bags is None else n_bags
        self.shrink = CFG.decision_shrink if shrink is None else shrink
        self.min_gain = min_gain
        self.n_cv = n_cv
        self.min_cv_gain = min_cv_gain
        self.w_max = CFG.decision_w_max if w_max is None else w_max
        self.seed = seed
        self.grid = np.exp(np.linspace(-0.9, 0.9, 25))

    def _ascent(self, proba, y):
        n_cls = proba.shape[1]
        w = np.ones(n_cls)
        best = f1_score(y, np.argmax(proba * w, axis=1), average='macro')
        for _ in range(self.n_rounds):
            improved = False
            for c in range(n_cls):
                cur = w[c]
                for g in self.grid:
                    w[c] = g
                    s = f1_score(y, np.argmax(proba * w, axis=1), average='macro')
                    if s > best + self.min_gain:
                        best, cur, improved = s, g, True
                w[c] = cur
            w = w / w[0]
            if not improved:
                break
        return w

    def _fit_weights(self, proba, y):
        """Bagging + co ngót — KHÔNG kiểm chứng (dùng nội bộ)."""
        rng = np.random.default_rng(self.seed)
        n = len(y)
        logs = []
        for _ in range(max(1, self.n_bags)):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < proba.shape[1]:
                continue
            logs.append(np.log(self._ascent(proba[idx], y[idx])))
        log_w = np.mean(logs, axis=0) if logs else np.zeros(proba.shape[1])
        w = np.exp(self.shrink * log_w)
        w = w / w[0]
        return np.clip(w, 1.0 / self.w_max, self.w_max)      # lớp bảo vệ 4

    def fit(self, proba_val, y_val):
        y_val = np.asarray(y_val).ravel()

        # ── Lớp bảo vệ 3: ước lượng độ lợi THẬT bằng CV giữ riêng ──
        gains = []
        try:
            cv = StratifiedKFold(self.n_cv, shuffle=True, random_state=self.seed)
            for tr, va in cv.split(proba_val, y_val):
                w_tr = self._fit_weights(proba_val[tr], y_val[tr])
                base = f1_score(y_val[va], np.argmax(proba_val[va], axis=1), average='macro')
                tuned = f1_score(y_val[va], np.argmax(proba_val[va] * w_tr, axis=1),
                                 average='macro')
                gains.append(tuned - base)
        except Exception as e:
            logger.warning(f"[DECIDE-28] CV kiểm chứng lỗi ({e}) — từ chối áp dụng w.")
        self.cv_gain_ = float(np.mean(gains)) if gains else -1.0
        self.cv_se_ = (float(np.std(gains, ddof=1) / np.sqrt(len(gains)))
                       if len(gains) > 1 else float('inf'))

        self.base_score_ = f1_score(y_val, np.argmax(proba_val, axis=1), average='macro')
        if (self.cv_gain_ - 2.0 * self.cv_se_) > self.min_cv_gain:
            self.weights_ = self._fit_weights(proba_val, y_val)
            self.applied_ = True
        else:
            self.weights_ = np.ones(proba_val.shape[1])
            self.applied_ = False
        self.best_score_ = f1_score(y_val, np.argmax(proba_val * self.weights_, axis=1),
                                    average='macro')
        return self

    def predict(self, proba):
        return np.argmax(proba * self.weights_, axis=1)


def tune_neutral_threshold(pos_minus_neg, y, candidates=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)):
    """[FEAT-6] Dò ngưỡng T cho is_neutral bằng mutual information.
    [CAL-36] Từ v40.3 dò trên TRAIN (không phải Cal) — đây là lựa chọn kỹ
    thuật đặc trưng, dùng nhãn train là hợp lệ và không rò rỉ; đồng thời giải
    phóng tập Cal khỏi một trong ba lần bị dùng lại."""
    best_t, best_mi = candidates[0], -1.0
    for t in candidates:
        flag = (np.abs(pos_minus_neg) < t).astype(np.float64).reshape(-1, 1)
        try:
            mi = mutual_info_classif(flag, y, discrete_features=[True], random_state=42)[0]
        except Exception:
            mi = 0.0
        if mi > best_mi:
            best_mi, best_t = mi, t
    return best_t, best_mi


# ═══════════════════════════════════════════════════════════════════
# 6. DATA LOADING & SPLIT
# ═══════════════════════════════════════════════════════════════════
def load_data(input_dir='input_data', group_col=None):
    files = glob.glob(os.path.join(input_dir, '*.csv')) + glob.glob(os.path.join(input_dir, '*.xlsx'))
    if not files:
        # [DATA-41] v40.2 âm thầm tạo 3000 mẫu giả ở đây (chỉ warning). Chạy
        # sai thư mục ⇒ script vẫn báo F1≈1.0 và ghi vào results_*.json. Với
        # một đề tài học thuật điều đó là không chấp nhận được.
        raise FileNotFoundError(
            f"Không tìm thấy file .csv/.xlsx nào trong {input_dir!r}. "
            "v40.3 KHÔNG tạo dữ liệu giả thay thế (xem [DATA-41]).")
    dfs = []
    for f in files:
        df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
        df.rename(columns=lambda c: 'comment' if c.lower() in ('comment', 'text')
                  else ('sentiment' if c.lower() in ('sentiment', 'label') else c),
                  inplace=True)
        cols = ['comment', 'sentiment']
        if group_col and group_col in df.columns and group_col not in cols:
            cols.append(group_col)
        if 'comment' in df.columns and 'sentiment' in df.columns:
            dfs.append(df[cols])
    if not dfs:
        raise ValueError(f"Không file nào trong {input_dir!r} có đủ 2 cột comment/sentiment.")
    out = pd.concat(dfs, ignore_index=True)
    # [ALIGN-2026-08] ĐỒNG BỘ VỚI load_data() CỦA train_v40.py
    # Loại dòng có comment/sentiment rỗng hoặc NaN rồi reset_index (cả V40 và
    # PhoBERT dùng cùng khối này). Trên dữ liệu hiện tại không có dòng nào như vậy,
    # nên đây là biện pháp phòng xa: chỉ cần một dòng bị loại ở hệ thống này mà
    # không bị loại ở hệ thống kia là split theo VỊ TRÍ trượt đi, và pairing giữa
    # các hệ thống đứt mà không có bất kỳ dấu hiệu nào trong log.
    _n_before = len(out)
    out = out[out['comment'].notna() & out['sentiment'].notna()].copy()
    out = out[out['comment'].astype(str).str.strip().str.len() > 0].copy()
    _n_dropped = _n_before - len(out)
    if _n_dropped:
        # [FSTR-53] v40.3 viết {{_n_dropped}} nên f-string in ra CHỮ LITERAL
        # "Đã BỎ {_n_dropped}/{_n_before} dòng" — đúng chỗ mà theo comment trên
        # là tình huống làm đứt pairing V40/PhoBERT, mà lại không in số.
        logger.warning(
            f"⚠️  [ALIGN-2026-08] Đã BỎ {_n_dropped}/{_n_before} dòng có comment/sentiment rỗng hoặc NaN "
            f"khi nạp từ '{input_dir}' — giống hệt load_data() của train_v40.py.")
    if len(out) == 0:
        raise ValueError(f"load_data(): sau khi loại dòng rỗng/NaN, không còn dòng hợp lệ trong '{input_dir}'.")
    return out.reset_index(drop=True)


def split_data_v40(df, y, group_col=None, seed=42):
    """[SPLIT-16] Lấy 1 fold từ StratifiedGroupKFold (group‑aware VÀ xấp xỉ
    cân bằng lớp) — sklearn chưa có StratifiedGroupShuffleSplit.
    [SPLIT-42] Nếu CÓ group_col mà cả hai cách group‑aware đều lỗi thì RAISE
    thay vì im lặng rơi về split ngẫu nhiên (v40.2 chỉ ghi warning rồi chạy
    tiếp với leakage)."""
    idx_all = np.arange(len(df))
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].values
        try:
            sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            idx_tr_cal, idx_te = next(iter(sgkf1.split(idx_all, y, groups)))
            sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            rel_tr, rel_cal = next(iter(sgkf2.split(idx_tr_cal, y[idx_tr_cal], groups[idx_tr_cal])))
            logger.info(f"[SPLIT-16] Chia bằng StratifiedGroupKFold (1 fold, 60/20/20) theo "
                        f"cột '{group_col}' — group-aware VÀ xấp xỉ cân bằng lớp.")
            return idx_tr_cal[rel_tr], idx_tr_cal[rel_cal], idx_te
        except Exception as e:
            logger.warning(f"[SPLIT-16] StratifiedGroupKFold lỗi ({e}) — thử GroupShuffleSplit.")
            try:
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                idx_tr_cal, idx_te = next(gss1.split(idx_all, y, groups))
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                rel_tr, rel_cal = next(gss2.split(idx_tr_cal, y[idx_tr_cal], groups[idx_tr_cal]))
                logger.info(f"Chia bằng GroupShuffleSplit theo '{group_col}' (không stratify).")
                return idx_tr_cal[rel_tr], idx_tr_cal[rel_cal], idx_te
            except Exception as e2:
                raise RuntimeError(
                    f"[SPLIT-42] Có group_col='{group_col}' nhưng CẢ HAI cách chia "
                    f"group-aware đều lỗi ({e} | {e2}). v40.3 từ chối rơi về split "
                    f"ngẫu nhiên vì điều đó tạo leakage train/test âm thầm.") from e2

    logger.warning("KHÔNG có cột group hợp lệ → stratified split ngẫu nhiên. CẢNH BÁO: "
                   "nếu dữ liệu có nhiều biến thể gần giống nhau, F1/accuracy đo được "
                   "sẽ lạc quan hơn thực tế do leakage.")
    idx_tr_cal, idx_te = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=seed)
    idx_tr, idx_cal = train_test_split(idx_tr_cal, test_size=0.25,
                                       stratify=y[idx_tr_cal], random_state=seed)
    return idx_tr, idx_cal, idx_te


# ═══════════════════════════════════════════════════════════════════
# 6b. CALIBRATION METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_calibration_metrics(y_true, probs, n_bins=10):
    """ECE (bin đều) + Brier — GIỮ NGUYÊN công thức v40.2 để bảng đã công bố
    (V40/PhoBERT) còn so sánh được. Quy ước biên: bin đầu [lo,hi], các
    bin sau (lo,hi] — thống nhất giữa cả hai file từ 2026‑07.
    [ECE-43] Bổ sung ece_em (đều mật độ) và ece_cw (classwise) làm chỉ số
    PHỤ, chính xác hơn nhưng KHÔNG thay thế `ece` trong bảng cũ."""
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == 0 \
            else (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())

    return {'ece': float(ece), 'brier': brier,
            'ece_em': ece_equal_mass(y_true, probs),
            'ece_cw': classwise_ece(y_true, probs)}


def ece_equal_mass(y_true, probs, n_bins=15):
    """[ECE-43] ECE với bin ĐỀU MẬT ĐỘ. Confidence dồn vào [0.7,1.0] nên bin
    đều-theo-giá-trị để 6/10 bin gần rỗng ⇒ phương sai ước lượng lớn."""
    y_true = np.asarray(y_true).ravel()
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classwise_ece(y_true, probs, n_bins=15):
    """[ECE-43] ECE trung bình trên TỪNG cột xác suất. Quan trọng ở đây: ECE
    top‑1 có thể trông đẹp trong khi cột `neutral` — đúng lớp đang có vấn đề —
    bị hiệu chỉnh rất tệ."""
    y_true = np.asarray(y_true).ravel()
    total = 0.0
    for c in range(probs.shape[1]):
        p, t = probs[:, c], (y_true == c).astype(float)
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        e = 0.0
        for i in range(n_bins):
            m = (p > edges[i]) & (p <= edges[i + 1])
            if m.sum():
                e += m.sum() / len(p) * abs(t[m].mean() - p[m].mean())
        total += e
    return float(total / probs.shape[1])


# ═══════════════════════════════════════════════════════════════════
# 6c. BÁO CÁO & HÌNH  — [REPORT-44]
# ═══════════════════════════════════════════════════════════════════
def _library_versions():
    """
    [V50 #3] Phiên bản các thư viện có thể làm đổi kết quả giữa hai phiên chạy.

    V40 trước đây không ghi lại gì về môi trường. Một lần chạy lại trên Kaggle
    từng cho thấy hậu quả: image Kaggle đổi giữa hai phiên và kết quả lệch 0,011 F1
    (p = 3e-05) mà không có cảnh báo nào. Ghi lại để so được về sau.
    """
    import importlib
    out = {}
    for name in ('sklearn', 'numpy', 'scipy', 'lightgbm', 'optuna'):
        try:
            out[name] = getattr(importlib.import_module(name), '__version__', 'unknown')
        except Exception:
            out[name] = 'absent'
    out['python'] = '.'.join(str(v) for v in sys.version_info[:3])
    return out


# ═══════════════════════════════════════════════════════════════════
# [V50] PHÂN TÍCH THEO NHÓM HIỆN TƯỢNG NGÔN NGỮ
#
# Bản ĐÃ SỬA hai lỗi khớp chuỗi con. Hai hệ thống phải dùng đúng cùng một
# định nghĩa nhóm thì bảng phân nhóm mới đặt cạnh nhau được.
#
# Hai lỗi đã sửa trong bản gốc, ghi lại để không ai "tối ưu" nó về như cũ:
#   • 'mà' khớp SUBSTRING vào bên trong 'màu' → MỌI câu nhắc màu sắc bị xếp vào
#     nhóm trai_chieu. Trên corpus này là 288 dòng thừa, toàn câu dễ, đủ để đẩy
#     f1_macro của nhóm lên 0,03 điểm một cách giả tạo.
#   • 'tạm' khớp vào 'tạm biệt' → lời chào bị đếm thành hedge.
# Bản dưới đây khớp theo TOKEN và BIGRAM, không khớp chuỗi con.
#
# LƯU Ý: bản trong train_phobert.py vẫn CHƯA sửa. Số phân nhóm trong
# results_phobert_multiseed.json vì thế KHÔNG so trực tiếp được với V40 —
# phải tính lại ngoại tuyến từ tệp xác suất bằng đúng hàm này.
# ═══════════════════════════════════════════════════════════════════
_V50_NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}
_V50_HEDGE_WORDS = {'tạm', 'tạm được', 'cũng được', 'bình thường', 'cũng ổn',
                    'tàm tạm', 'không đến nỗi', 'cũng tạm'}
_V50_CONTRAST_PHRASES = ('nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà')


def _tag_phenomena(text: str) -> dict:
    """Gán nhãn bốn hiện tượng ngôn ngữ cho MỘT câu. Khớp theo token/bigram."""
    t = str(text).lower()
    # Tách dấu câu ra khỏi token TRƯỚC khi so khớp, nếu không thì 'không,' và
    # 'chưa.' đều trượt và nhóm phu_dinh bị đếm thiếu.
    t_spaced = re.sub(r'([.,;:!?()\[\]"\'])', r' \1 ', t)
    toks = [w for w in t_spaced.split() if w]
    bigrams = set(zip(toks, toks[1:]))

    has_neg = any(w in _V50_NEGATION_WORDS for w in toks)
    has_hedge = (
        any(h in toks for h in _V50_HEDGE_WORDS if ' ' not in h)
        or any(tuple(h.split()) in bigrams for h in _V50_HEDGE_WORDS if ' ' in h)
    ) and 'tạm biệt' not in t
    has_contrast = (
        any(c in toks for c in _V50_CONTRAST_PHRASES if ' ' not in c)
        or any(tuple(c.split()) in bigrams for c in _V50_CONTRAST_PHRASES if ' ' in c)
    )
    is_short = len(toks) <= 3
    return {'phu_dinh': has_neg, 'hedge': has_hedge,
            'trai_chieu': has_contrast, 'cau_ngan': is_short}


def error_analysis_by_subgroup(raw_texts, y_true, y_pred, le, min_n=5):
    """
    Chia tập test theo hiện tượng ngôn ngữ và báo cáo f1_macro từng nhóm.

    Nhóm nào có ít hơn min_n mẫu thì BỎ QUA thay vì báo cáo — f1_macro trên 3
    mẫu không mang thông tin gì và chỉ tạo cột số trông có vẻ chắc chắn.
    """
    out = {}
    try:
        tags = [_tag_phenomena(t) for t in raw_texts]
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        for key in ('phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan'):
            mask = np.array([t[key] for t in tags], dtype=bool)
            if mask.sum() < min_n:
                continue
            out[key] = {
                'n': int(mask.sum()),
                'accuracy': float(accuracy_score(y_true[mask], y_pred[mask])),
                'f1_macro': float(f1_score(y_true[mask], y_pred[mask],
                                           average='macro', zero_division=0)),
            }
        for key, v in out.items():
            logger.info(f"   [nhóm] {key:<12} n={v['n']:<6} acc={v['accuracy']:.4f} "
                        f"f1_macro={v['f1_macro']:.4f}")
    except Exception as e:
        logger.warning(f"[V50] Không tính được phân nhóm: {type(e).__name__}: {e}")
    return out


def dump_probabilities(seed, split_name, row_index, y_true, prob, le,
                       out_dir='reports/v40_probs', version='v40'):
    """
    [V50 #1] Ghi xác suất TỪNG MẪU ra CSV.

    Đây là thay đổi quan trọng nhất của bản này. V40 trước đây chỉ lưu số tổng
    hợp, nên là hệ DUY NHẤT trong ba hệ không có xác suất mức mẫu — và vì thiếu
    nó mà V40 không vào được McNemar, bootstrap bắt cặp, đường risk–coverage,
    hay bảng phân nhóm ba cột. Mọi phân tích đó cần xác suất mức mẫu và KHÔNG
    cái nào tính lại được từ F1/ECE.

    LƯỢC ĐỒ — cố ý trùng với reports/v40_probs/ và reports/phobert_logits/ để
    một hàm nạp duy nhất đọc được cả hai hệ:

        seed, split, row_index, y_true, prob_0, prob_1, prob_2

    row_index là chỉ số dòng trong DataFrame gốc (idx_te/idx_cal), KHÔNG phải
    thứ tự trong mảng con — nhờ vậy join ngược về câu văn và về xác suất của hệ
    khác luôn đúng. Đã kiểm chứng: join V40 với PhoBERT theo khoá này cho 7.982
    dòng với y_true trùng khớp 100%.

    Xác suất ghi ra là prob_te SAU temperature scaling — cùng đại lượng dùng để
    tính ECE/Brier trong bảng chính, nên hai nguồn số luôn nhất quán.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        prob = np.asarray(prob, dtype=float)
        row_index = np.asarray(row_index).ravel()
        y_true = np.asarray(y_true).ravel()
        if not (len(row_index) == len(y_true) == prob.shape[0]):
            logger.warning(
                f"[V50 #1] Bỏ qua dump xác suất ({split_name}): độ dài không khớp — "
                f"row_index={len(row_index)}, y_true={len(y_true)}, prob={prob.shape}.")
            return None
        cols = {'seed': int(seed), 'split': split_name,
                'row_index': row_index, 'y_true': y_true}
        for k in range(prob.shape[1]):
            cols[f'prob_{k}'] = prob[:, k]
        path = os.path.join(out_dir, f"{version}_{split_name}_seed_{seed}.csv")
        pd.DataFrame(cols).to_csv(path, index=False)
        meta_path = os.path.join(out_dir, 'classes.json')
        if not os.path.exists(meta_path):
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({'classes_in_order': [str(c) for c in le.classes_]},
                          f, ensure_ascii=False, indent=1)
        logger.info(f"💾 [V50 #1] Đã lưu xác suất {split_name} (n={len(row_index)}) → {path}")
        return path
    except Exception as e:
        # Không bao giờ để việc ghi file phụ làm hỏng một lần train 33 phút.
        logger.warning(f"[V50 #1] Không ghi được xác suất {split_name}: {type(e).__name__}: {e}")
        return None


def save_reports(y_true, y_pred, probs, class_names, out_dir, tag=''):
    """v40.2 import matplotlib/seaborn và tạo thư mục reports/ nhưng KHÔNG ghi
    file nào. Ở đây xuất ma trận nhầm lẫn + biểu đồ tin cậy + chỉ số theo lớp."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax[0])
        ax[0].set_title(f'Ma trận nhầm lẫn {tag}')
        ax[0].set_xlabel('Dự đoán'); ax[0].set_ylabel('Thực tế')

        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(float)
        edges = np.quantile(conf, np.linspace(0, 1, 11))
        xs, ys = [], []
        for i in range(10):
            m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
            if m.sum():
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
        ax[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Hiệu chỉnh hoàn hảo')
        ax[1].plot(xs, ys, 'o-', label='Mô hình')
        ax[1].set_xlabel('Độ tự tin trung bình'); ax[1].set_ylabel('Độ chính xác')
        ax[1].set_title('Biểu đồ tin cậy (bin đều mật độ)'); ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'report{tag}.png'), dpi=130)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"[REPORT-44] Không vẽ được hình: {e}")

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class = {str(c): {'precision': float(p[i]), 'recall': float(r[i]),
                          'f1': float(f[i]), 'support': int(s[i]),
                          'p_minus_r': float(p[i] - r[i])}
                 for i, c in enumerate(class_names)}
    with open(os.path.join(out_dir, f'per_class{tag}.json'), 'w', encoding='utf-8') as fh:
        json.dump(per_class, fh, ensure_ascii=False, indent=2)
    return per_class


# ═══════════════════════════════════════════════════════════════════
# 7. OPTUNA OBJECTIVE  — [OPT-37][OPT-38][OPT-39]
# ═══════════════════════════════════════════════════════════════════
def optuna_objective(trial, clean_train, y_train, groups_train=None,
                     manual_train=None, seed=42):
    """[LEAK-9] Fit một bộ trích đặc trưng MỚI cho từng fold, chỉ trên phần
    train của fold đó — mọi bước fit() phải nằm trong vòng lặp CV.

    [OPT-37] Không gian tìm kiếm đã NỚI: v40.2 chặn lr_C ở [0.1,10] và Optuna
    chạm sát cận trên ở 4/10 seed (9.95/9.89/9.55/9.46); tfidf_max_feat chạm
    biên 20000 ở 2/10 seed. Thêm min_df vào không gian ([TFIDF-26]).

    [OPT-38] Objective vẫn là proxy (một LR) chứ không phải cả stack — chấp
    nhận có chủ đích để giữ chi phí. Bù lại: (a) nối thêm manual features cho
    gần mục tiêu cuối, (b) MedianPruner cắt trial tệ từ fold đầu. Khoảng cách
    proxy↔cuối ở v40.2 là 3.3 điểm (CV 0.8335 vs Test 0.8667) — hãy đọc
    'Best CV F1' như một ĐẠI LƯỢNG SO SÁNH TƯƠNG ĐỐI giữa các cấu hình, KHÔNG
    phải ước lượng hiệu năng.

    [OPT-39] random_state của CV nội bộ giờ theo `seed` (v40.2 hardcode 42 dù
    split ngoài dùng seed ⇒ fold nội bộ giống hệt ở cả 10 seed)."""
    max_feat = trial.suggest_int('tfidf_max_feat', 20000, 80000, step=10000)
    ngram_choice = trial.suggest_categorical('ngram_range', [(1, 2), (1, 3)])
    min_df = trial.suggest_categorical('min_df', [2, 3, 5])
    lr_C = trial.suggest_float('lr_C', 0.01, 100.0, log=True)
    sublinear = trial.suggest_categorical('sublinear_tf', [True, False])

    if groups_train is not None:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(y_train)), y_train, groups_train))
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(y_train)), y_train))

    fold_f1 = []
    for step, (tr_idx, va_idx) in enumerate(splits):
        vec = TfidfVectorizer(ngram_range=ngram_choice, min_df=min_df, max_df=0.9,
                              max_features=max_feat, sublinear_tf=sublinear)
        Xtr = vec.fit_transform([clean_train[i] for i in tr_idx])
        Xva = vec.transform([clean_train[i] for i in va_idx])
        if manual_train is not None:
            sc = ManualScaler(seed).fit(manual_train[tr_idx])
            Xtr = sp_hstack([Xtr, csr_matrix(sc.transform(manual_train[tr_idx]))], format='csr')
            Xva = sp_hstack([Xva, csr_matrix(sc.transform(manual_train[va_idx]))], format='csr')
        m = LogisticRegression(C=lr_C, class_weight='balanced', max_iter=1000,
                               random_state=seed)
        m.fit(Xtr, y_train[tr_idx])
        fold_f1.append(f1_score(y_train[va_idx], m.predict(Xva), average='macro'))
        trial.report(float(np.mean(fold_f1)), step)          # [OPT-38] pruner
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_f1))


# ═══════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE V40
# ═══════════════════════════════════════════════════════════════════
def train_v40(input_dir='input_data', n_trials=25, seed=42, group_col=None,
              include_svm=True, include_lgbm=True, use_cluster_features=True,
              config: Config = None):
    """[ABLA-46] `config` cho phép bật/tắt từng thành phần v40.3 để dựng bảng
    ablation mà không sửa code. Ba tham số cũ (include_svm/include_lgbm/
    use_cluster_features) được giữ để tương thích ngược."""
    global CFG
    if config is not None:
        CFG = config
    CFG.include_svm = include_svm
    CFG.include_lgbm = include_lgbm
    CFG.use_cluster_features = use_cluster_features

    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    # ── 1. Nạp dữ liệu ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"🚀 BẮT ĐẦU HUẤN LUYỆN V40 ({SUBVERSION}, seed={seed})")
    logger.info(f"   Config: {CFG.summary()}")
    logger.info("=" * 60)
    df = load_data(input_dir, group_col=group_col)
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    raw_texts = df['comment'].astype(str).tolist()
    logger.info(f"Đã nạp {len(df)} mẫu. Nhãn: {le.classes_}")
    if len(le.classes_) != 3:
        logger.warning(f"⚠️  Kỳ vọng 3 lớp nhưng LabelEncoder thấy {len(le.classes_)}: "
                       f"{list(le.classes_)} — kiểm tra chính tả/viết hoa nhãn.")
    if group_col is not None and group_col not in df.columns:
        logger.warning(f"Cột group_col='{group_col}' không tồn tại — bỏ qua group-aware split.")
        group_col = None

    if not HAS_UNDER:
        logger.warning("[SEG-21] underthesea CHƯA CÀI → không phân đoạn từ, không POS. "
                       "Chạy `pip install underthesea` để bật. (Log v40.2 cho thấy bước "
                       "POS mất 0 ms ở cả 10 seed — đúng triệu chứng này.) "
                       "LƯU Ý: việc này KHÔNG ảnh hưởng độ phủ lexicon — xem [LEX-47].")
    else:
        logger.info(f"[SEG-21] underthesea sẵn sàng (phân đoạn từ cho TF‑IDF"
                    f"{', POS' if HAS_POS else ''}). Không tác động tới độ phủ "
                    f"lexicon — độ phủ do KÍCH THƯỚC LEXICON quyết định [LEX-47].")

    # ── 2. Tiền xử lý ─────────────────────────────────────────────
    logger.info("Tiền xử lý văn bản...")
    clean_texts = TextPreprocessor.preprocess_batch(raw_texts)

    # ── 3. Chia Train / Calibration / Test (60/20/20) ─────────────
    idx_tr, idx_cal, idx_te = split_data_v40(df, y, group_col=group_col, seed=seed)
    # [ALIGN-2026-08] KIỂM TRA DẤU VÂN TAY SPLIT — xem split_fingerprint.py
    # Dừng NGAY nếu seed này đã được một hệ thống khác chạy với cách chia khác.
    try:
        from split_fingerprint import check_split_fingerprint
        check_split_fingerprint('v40', seed, idx_tr, idx_cal, idx_te, len(df))
    except ImportError:
        logger.warning("[ALIGN-2026-08] Không tìm thấy split_fingerprint.py cạnh file train — "
                       "BỎ QUA kiểm tra lệch split. Copy file đó vào cùng thư mục để bật lại.")
    sub = lambda lst, idx: [lst[i] for i in idx]
    raw_tr, raw_cal, raw_te = sub(raw_texts, idx_tr), sub(raw_texts, idx_cal), sub(raw_texts, idx_te)
    clean_tr, clean_cal, clean_te = sub(clean_texts, idx_tr), sub(clean_texts, idx_cal), sub(clean_texts, idx_te)
    y_tr, y_cal, y_te = y[idx_tr], y[idx_cal], y[idx_te]
    logger.info(f"Train: {len(y_tr)} | Calibration: {len(y_cal)} | Test: {len(y_te)}")
    groups_tr = df[group_col].values[idx_tr] if group_col is not None else None

    # ── 4a. [LEX-47] ★ MỞ RỘNG LEXICON TRÊN TRAIN CỦA SEED NÀY ────
    # Đặt ở ĐÚNG chỗ này là bắt buộc: SAU split (để không rò rỉ nhãn sang
    # Cal/Test) và TRƯỚC khi trích đặc trưng thủ công (để cả hai tập đi qua
    # cùng một từ điển).
    TextPreprocessor.restore_lexicon(_active_lexicon_snapshot())  # xoá vết seed trước
    cov_before = TextPreprocessor.lexicon_coverage(raw_tr)['coverage_rate']
    lex_diag = {'lexicon_size': len(TextPreprocessor.POSITIVE_WORDS)
                                + len(TextPreprocessor.NEGATIVE_WORDS)}
    if CFG.expand_lexicon:
        lex_diag = TextPreprocessor.expand_lexicon_from_train(
            raw_tr, [str(c) for c in le.inverse_transform(y_tr)])
        logger.info(f"[LEX-47] Mở rộng lexicon trên TRAIN (seed={seed}): "
                    f"+{lex_diag['n_added_pos']} POS, +{lex_diag['n_added_neg']} NEG "
                    f"trên {lex_diag['n_candidates']} ứng viên → lexicon "
                    f"{lex_diag['lexicon_size']} mục.")
        logger.debug(f"[LEX-47] 20 mục điểm cao nhất: {lex_diag['top_20']}")
    else:
        logger.info(f"[LEX-47] TẮT mở rộng lexicon (ablation) — giữ "
                    f"{lex_diag['lexicon_size']} mục hạt giống.")

    # ── 4b. Đặc trưng thủ công + POS ──────────────────────────────
    logger.info("Trích xuất manual lexicon features...")
    manual_tr = TextPreprocessor.extract_features_batch(raw_tr)
    manual_cal = TextPreprocessor.extract_features_batch(raw_cal)
    manual_te = TextPreprocessor.extract_features_batch(raw_te)

    # [DIAG-45] ĐỘ PHỦ LEXICON — con số trả lời trực tiếp câu hỏi "ma trận
    # thưa còn hạn chế ở đâu". Đo trên Train, ghi vào log + JSON kết quả.
    cov = float((manual_tr[:, TextPreprocessor.IDX_HAS_MATCH] > 0).mean())
    cov_te = float((manual_te[:, TextPreprocessor.IDX_HAS_MATCH] > 0).mean())
    logger.info(f"[DIAG-45] Độ phủ lexicon Train = {cov:.1%} | Test = {cov_te:.1%} "
                f"(trước mở rộng: {cov_before:.1%}) "
                f"({'ĐẠT' if cov >= 0.6 else '⚠ THẤP'})")
    if cov < 0.6:
        # [LEX-47] Ghi rõ hướng chẩn đoán để không lặp lại chuyện đi cài lại
        # underthesea: nhánh này KHÔNG dùng word_tokenize, nên thư viện phân
        # đoạn từ không liên quan gì tới con số ở trên.
        logger.warning(
            "[LEX-47] ⚠ Độ phủ thấp → kiểm tra KÍCH THƯỚC LEXICON "
            f"({lex_diag['lexicon_size']} mục) và Config.expand_lexicon, KHÔNG "
            "phải underthesea: nhánh _score()/normalize_for_lexicon()/"
            "_tokenize_for_lexicon() không gọi word_tokenize() ở bất kỳ đâu.")
    if abs(cov - cov_te) > 0.05:
        logger.warning(f"[LEX-47] ⚠ Lệch độ phủ Train−Test = {cov - cov_te:+.3f} "
                       f"(>0.05) — dấu hiệu từ điển mở rộng đang khớp riêng Train.")

    if CFG.use_pos and HAS_POS:
        logger.info("Trích xuất POS features (4 tỉ lệ, trên văn bản đã chuẩn hoá)...")
        pos_tr = _pos_to_matrix(TextPreprocessor.extract_pos_features_batch(raw_tr))
        pos_cal = _pos_to_matrix(TextPreprocessor.extract_pos_features_batch(raw_cal))
        pos_te = _pos_to_matrix(TextPreprocessor.extract_pos_features_batch(raw_te))
    else:
        pos_tr = pos_cal = pos_te = None

    # [CAL-36] T dò trên TRAIN (v40.2 dò trên Cal — một trong ba lần Cal bị
    # dùng lại). Dùng nhãn train cho lựa chọn kỹ thuật đặc trưng là hợp lệ.
    idx_pmn = TextPreprocessor.IDX_POS_MINUS_NEG
    idx_neu = TextPreprocessor.IDX_IS_NEUTRAL
    best_T, best_mi = tune_neutral_threshold(manual_tr[:, idx_pmn], y_tr)
    logger.info(f"[FEAT-6][CAL-36] Ngưỡng neutral T tối ưu trên TRAIN: T={best_T} (MI={best_mi:.4f})")
    for mat in (manual_tr, manual_cal, manual_te):
        mat[:, idx_neu] = (np.abs(mat[:, idx_pmn]) < best_T).astype(np.float32)

    meta_feat_tr = MetaFeatureExtractor.extract_batch(raw_tr)
    meta_feat_cal = MetaFeatureExtractor.extract_batch(raw_cal)
    meta_feat_te = MetaFeatureExtractor.extract_batch(raw_te)

    data_tr = {'raw': raw_tr, 'clean': clean_tr, 'manual': manual_tr,
               'meta': meta_feat_tr, 'pos': pos_tr}
    data_cal = {'raw': raw_cal, 'clean': clean_cal, 'manual': manual_cal,
                'meta': meta_feat_cal, 'pos': pos_cal}
    data_te = {'raw': raw_te, 'clean': clean_te, 'manual': manual_te,
               'meta': meta_feat_te, 'pos': pos_te}

    # ── 5. Optuna ─────────────────────────────────────────────────
    logger.info(f"Tối ưu siêu tham số với Optuna ({n_trials} trials, có MedianPruner)...")
    study = optuna.create_study(
        direction='maximize', sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1))     # [OPT-38]
    study.optimize(
        lambda t: optuna_objective(t, clean_tr, y_tr, groups_train=groups_tr,
                                   manual_train=manual_tr, seed=seed),
        n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best = study.best_params
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logger.info(f"Best params: {best}")
    logger.info(f"Best CV F1 (proxy 1 LR, KHÔNG phải ước lượng hiệu năng — [OPT-38]): "
                f"{study.best_value:.4f}  | trial bị cắt: {n_pruned}/{n_trials}")

    # ── 6-8. Bagged multi-view stacking ───────────────────────────
    logger.info(f"Tạo OOF {N_OOF_FOLDS} folds — multi-view [STACK-29], "
                f"oof_mode='{CFG.oof_mode}' [STACK-30]...")
    stack = MultiViewStack(best, N_OOF_FOLDS, seed)
    oof_tr, (prob_cal_block, prob_te_block) = stack.fit_transform(
        data_tr, y_tr, [data_cal, data_te], groups_train=groups_tr)
    view_names = stack.view_names_
    n_views, n_classes = len(view_names), len(le.classes_)
    logger.info(f"  Các view tham gia: {view_names}")

    # ── 9. Cluster one-hot + đặc trưng meta ───────────────────────
    clusterer = MetaClusterer(seed=seed)
    if CFG.use_cluster_features:
        oh_tr = clusterer.to_onehot(clusterer.fit_predict(meta_feat_tr))
        oh_cal = clusterer.to_onehot(clusterer.transform(meta_feat_cal))
        oh_te = clusterer.to_onehot(clusterer.transform(meta_feat_te))
    else:
        oh_tr = np.zeros((len(y_tr), 0)); oh_cal = np.zeros((len(y_cal), 0))
        oh_te = np.zeros((len(y_te), 0))

    manual_scaler = ManualScaler(seed).fit(manual_tr)        # [ARCH-7][SCALE-25]
    meta_train = build_meta_features(oof_tr, n_views, n_classes,
                                     manual_scaler.transform(manual_tr), oh_tr)
    meta_cal = build_meta_features(prob_cal_block, n_views, n_classes,
                                   manual_scaler.transform(manual_cal), oh_cal)
    meta_te = build_meta_features(prob_te_block, n_views, n_classes,
                                  manual_scaler.transform(manual_te), oh_te)
    logger.info(f"  Kích thước đầu vào meta: {meta_train.shape[1]} cột "
                f"({n_views} view × {n_classes} lớp"
                f"{' + bậc 2' if CFG.meta_second_order else ''}"
                f" + {oh_tr.shape[1]} cluster + {manual_tr.shape[1]} lexicon)")

    # ── 10. Meta-learner: chọn C bằng CV trên TRAIN [CAL-36] ──────
    logger.info("Chọn C của meta-learner bằng CV group-aware trên TRAIN ([CAL-36])...")
    if groups_tr is not None:
        cv_meta = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        meta_splits = list(cv_meta.split(meta_train, y_tr, groups_tr))
    else:
        cv_meta = StratifiedKFold(3, shuffle=True, random_state=seed)
        meta_splits = list(cv_meta.split(meta_train, y_tr))
    best_c, best_f1_cv = None, -1.0
    for C in [0.01, 0.1, 1.0, 5.0, 10.0]:
        scores = []
        for tr_i, va_i in meta_splits:
            m = StackingMetaLearner(C=C, seed=seed).fit(meta_train[tr_i], y_tr[tr_i])
            scores.append(f1_score(y_tr[va_i], np.argmax(m.predict_proba(meta_train[va_i]), 1),
                                   average='macro'))
        s = float(np.mean(scores))
        logger.debug(f"  C={C:<6} → F1-macro CV(train) = {s:.4f}")
        if s > best_f1_cv:
            best_f1_cv, best_c = s, C
    logger.info(f"Chọn C={best_c} với F1-CV(train)={best_f1_cv:.4f}")
    meta = StackingMetaLearner(C=best_c, seed=seed).fit(meta_train, y_tr)

    # ── 11. Temperature scaling trên Cal [CALIB-35] ───────────────
    prob_cal_raw = meta.predict_proba(meta_cal)
    calib_cal_raw = compute_calibration_metrics(y_cal, prob_cal_raw)
    f1_cal_raw = f1_score(y_cal, np.argmax(prob_cal_raw, 1), average='macro')
    temperature = None
    if CFG.use_temperature:
        ok, ts, e0, e1 = accept_temperature_cv(prob_cal_raw, y_cal, seed=seed)
        logger.info(f"[CALIB-35] Temperature scaling: T={ts.T_:.4f} | "
                    f"ECE(CV nội bộ trên Cal) {e0:.4f} → {e1:.4f} → "
                    f"{'ÁP DỤNG' if ok else 'LOẠI'}")
        if ok:
            temperature = ts
    prob_cal = temperature.predict_proba(prob_cal_raw) if temperature else prob_cal_raw

    # ── 12. Lớp quyết định cost-sensitive trên Cal [DECIDE-28] ────
    decision = None
    if CFG.use_decision_layer:
        decision = MacroF1DecisionLayer(seed=seed).fit(prob_cal, y_cal)
        logger.info(f"[DECIDE-28] ΔF1 giữ-riêng (CV trên Cal) = {decision.cv_gain_:+.4f} "
                    f"± {decision.cv_se_:.4f} (SE) → tiêu chí mean−2·SE > 0 → "
                    f"{'ÁP DỤNG' if decision.applied_ else 'LOẠI (không đủ bằng chứng)'}")
        logger.info(f"[DECIDE-28] w={np.round(decision.weights_, 3)} | "
                    f"F1-Cal {decision.base_score_:.4f} → {decision.best_score_:.4f} "
                    f"(bagging={CFG.decision_bags}, co ngót λ={CFG.decision_shrink})")

    # ── 13. Đánh giá trên Test ────────────────────────────────────
    logger.info("Đánh giá trên tập Test...")
    prob_te_raw = meta.predict_proba(meta_te)
    prob_te = temperature.predict_proba(prob_te_raw) if temperature else prob_te_raw
    pred_argmax = np.argmax(prob_te, axis=1)
    pred_final = decision.predict(prob_te) if decision else pred_argmax

    f1_argmax = f1_score(y_te, pred_argmax, average='macro')
    acc_argmax = accuracy_score(y_te, pred_argmax)
    f1_test = f1_score(y_te, pred_final, average='macro')
    acc_test = accuracy_score(y_te, pred_final)
    # ECE/Brier đo trên xác suất ĐÃ hiệu chỉnh (chất lượng xác suất), tách
    # khỏi F1 sau lớp quyết định (chất lượng quyết định) — hai câu hỏi khác nhau.
    calib = compute_calibration_metrics(y_te, prob_te)

    logger.info(f"✅ F1-Macro TEST = {f1_test:.4f}   (argmax thuần: {f1_argmax:.4f}, "
                f"Δ lớp quyết định = {f1_test - f1_argmax:+.4f})")
    logger.info(f"✅ Accuracy      = {acc_test:.4f}   (argmax thuần: {acc_argmax:.4f})")
    logger.info(f"✅ ECE={calib['ece']:.4f} (bin đều, so sánh được với bảng cũ) | "
                f"ECE_em={calib['ece_em']:.4f} | ECE_cw={calib['ece_cw']:.4f} | "
                f"Brier={calib['brier']:.4f}")
    logger.info("\n" + classification_report(y_te, pred_final,
                                             target_names=[str(c) for c in le.classes_]))

    # [REPORT-44] xuất hình + chỉ số theo lớp
    rep_dir = os.path.join('reports', VERSION, f'seed_{seed}')
    per_class = save_reports(y_te, pred_final, prob_te,
                             [str(c) for c in le.classes_], rep_dir)
    for c, m in per_class.items():
        logger.info(f"   {c:<10s} P={m['precision']:.3f} R={m['recall']:.3f} "
                    f"F1={m['f1']:.3f}  P−R={m['p_minus_r']:+.3f}")

    # ── [V50] Phân nhóm hiện tượng ngôn ngữ ───────────────────────
    # Dùng pred_final (sau lớp quyết định) cho nhất quán với per_class ở trên.
    # Con số so sánh ba hệ vẫn là f1_comparable = f1_argmax; bảng phân nhóm là
    # phân tích chẩn đoán riêng, không trộn vào bảng chính.
    logger.info("Phân tích lỗi theo nhóm hiện tượng ngôn ngữ...")
    subgroup_results = error_analysis_by_subgroup(raw_te, y_te, pred_final, le)

    # ── [V50 #1] Lưu xác suất per-sample ──────────────────────────
    dump_probabilities(seed, 'test', idx_te, y_te, prob_te, le)
    dump_probabilities(seed, 'calib', idx_cal, y_cal, prob_cal, le)

    # ── 14. Lưu mô hình ───────────────────────────────────────────
    prod_dir = os.path.join('models', 'production', VERSION, f'seed_{seed}')
    os.makedirs(prod_dir, exist_ok=True)
    bundle = {
        'stack': stack, 'meta': meta, 'clusterer': clusterer if CFG.use_cluster_features else None,
        'manual_scaler': manual_scaler, 'le': le, 'view_names': view_names,
        'temperature': temperature, 'decision': decision, 'best_T': float(best_T),
        'config': asdict(CFG), 'n_views': n_views, 'n_classes': n_classes,
        'subversion': SUBVERSION,
        # [SERV-57] Lexicon là THAM SỐ ĐÃ HỌC kể từ [LEX-47] (mở rộng bằng nhãn
        # của Train). Không lưu nó thì inference dùng lexicon gốc trong khi mô
        # hình được train với lexicon mở rộng — training/serving skew âm thầm,
        # đúng loại lỗi mà [INFER-15] đã gộp 8 file .pkl thành 1 để tránh.
        'lexicon': TextPreprocessor.lexicon_snapshot(),
        'lexicon_size': int(lex_diag['lexicon_size']),
    }
    # [STACK-30] compress=3 bù kích thước cho bagged stacking (K×V bộ trích
    # đặc trưng thay vì 1). Đổi Config.oof_mode='refit' nếu ưu tiên kích thước.
    bundle_path = os.path.join(prod_dir, 'v40_bundle.pkl')
    joblib.dump(bundle, bundle_path, compress=JOBLIB_COMPRESS)
    model_size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
    training_time_sec = time.time() - t_start
    logger.info(f"📁 Đã lưu {bundle_path}  ({model_size_mb:.2f} MB, nén mức {JOBLIB_COMPRESS})")
    logger.info(f"⏱️  Tổng thời gian: {training_time_sec:.1f}s")

    return {
        # [ALIGN-2026-08] [F] 'f1_comparable' = chỉ số DUY NHẤT được dùng trong bảng so sánh
        # 2 hệ thống. Với V40 nó là f1_argmax (KHÔNG phải 'f1'), vì 'f1' của V40 đã đi
        # qua MacroF1DecisionLayer — một lớp tinh chỉnh quyết định mà PhoBERT
        # không có. Lợi ích của lớp đó vẫn được báo cáo riêng qua 'delta_decision_layer'.
        'f1_comparable': float(f1_argmax),
        'f1': float(f1_test), 'accuracy': float(acc_test),
        'f1_argmax': float(f1_argmax), 'accuracy_argmax': float(acc_argmax),
        'delta_decision_layer': float(f1_test - f1_argmax),
        'ece': calib['ece'], 'ece_em': calib['ece_em'], 'ece_cw': calib['ece_cw'],
        'brier': calib['brier'],
        'training_time_sec': float(training_time_sec),
        'model_size_mb': float(model_size_mb),
        'best_params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
        'meta_C': float(best_c), 'meta_f1_cv_train': float(best_f1_cv),
        'neutral_threshold_T': float(best_T), 'neutral_threshold_mi': float(best_mi),
        'lexicon_coverage_train': cov,
        # [LEX-47] ba con số cần cho bảng ablation về độ phủ
        'lexicon_coverage_test': cov_te,
        'lexicon_coverage_before_expand': float(cov_before),
        'lexicon_size': int(lex_diag['lexicon_size']),
        'temperature_applied': bool(temperature is not None),
        'temperature_T': float(temperature.T_) if temperature else 1.0,
        'decision_weights': ([float(x) for x in decision.weights_] if decision else None),
        'decision_applied': bool(decision.applied_) if decision else False,
        'decision_cv_gain': float(decision.cv_gain_) if decision else 0.0,
        'decision_cv_se': float(decision.cv_se_) if decision else 0.0,
        'f1_cal_raw': float(f1_cal_raw), 'ece_cal_raw': calib_cal_raw['ece'],
        'view_names': view_names, 'per_class': per_class,
        # [V50 #2] Cùng khoá, cùng cấu trúc với results_phobert_multiseed.json. CẢNH BÁO khi đọc:
        # hàm _tag_phenomena ở đây là bản ĐÃ SỬA; bản trong train_phobert.py
        # thì CHƯA. Không đặt hai cột cạnh nhau nếu chưa tính lại PhoBERT ngoại
        # tuyến bằng cùng một hàm.
        'subgroup_results': subgroup_results,
        # [V50 #3] Ghi lại môi trường để lần sau so được. V40 trước đây không
        # lưu gì về phiên bản thư viện.
        'lib_versions': _library_versions(),
        'subversion': SUBVERSION,
    }


# ═══════════════════════════════════════════════════════════════════
# 8b. INFERENCE  — [INFER-15]
# ═══════════════════════════════════════════════════════════════════
def load_v40_bundle(prod_dir):
    """Nạp bundle từ models/production/v40/seed_{seed}/.
    v40.3 gộp toàn bộ artifact vào MỘT file thay vì 8 file rời — không còn
    khả năng nạp thiếu neutral_threshold.pkl rồi lặng lẽ dùng ngưỡng mặc định
    (training/serving skew)."""
    path = os.path.join(prod_dir, 'v40_bundle.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không thấy {path}. (Model v40.2 cũ dùng 8 file .pkl rời "
                                f"— hãy train lại bằng v40.3.)")
    return joblib.load(path)


def predict_v40(raw_texts, bundle):
    """Trả về (nhãn dự đoán, ma trận xác suất). `raw_texts` là văn bản THÔ —
    hàm tự làm toàn bộ các bước THEO ĐÚNG THỨ TỰ lúc train."""
    global CFG
    if 'config' in bundle:
        CFG = Config(**bundle['config'])      # tái tạo đúng cấu hình lúc train
    # [SERV-57] Nạp lại ĐÚNG lexicon lúc train. Bundle v40.3 không có khoá này
    # nên báo lỗi rõ ràng thay vì lặng lẽ dùng lexicon mặc định.
    if 'lexicon' in bundle:
        TextPreprocessor.restore_lexicon(bundle['lexicon'])
    elif bundle.get('subversion', '') >= 'v40.4':
        raise KeyError("Bundle thiếu khoá 'lexicon' — hãy train lại bằng v40.4.")
    else:
        logger.warning("[SERV-57] Bundle cũ (%s) không lưu lexicon → dùng lexicon "
                       "mặc định. Nếu bundle được train với expand_lexicon=True "
                       "thì dự đoán sẽ LỆCH so với lúc train.",
                       bundle.get('subversion', '?'))

    clean = TextPreprocessor.preprocess_batch(raw_texts)
    manual = TextPreprocessor.extract_features_batch(raw_texts)
    manual[:, TextPreprocessor.IDX_IS_NEUTRAL] = (
        np.abs(manual[:, TextPreprocessor.IDX_POS_MINUS_NEG]) < bundle['best_T']
    ).astype(np.float32)
    meta_f = MetaFeatureExtractor.extract_batch(raw_texts)
    pos = (_pos_to_matrix(TextPreprocessor.extract_pos_features_batch(raw_texts))
           if (CFG.use_pos and HAS_POS) else None)
    data = {'raw': list(raw_texts), 'clean': clean, 'manual': manual,
            'meta': meta_f, 'pos': pos}

    prob_block = bundle['stack'].predict_proba_block(data)
    if bundle['clusterer'] is not None:
        oh = bundle['clusterer'].to_onehot(bundle['clusterer'].transform(meta_f))
    else:
        oh = np.zeros((len(raw_texts), 0))
    meta_in = build_meta_features(prob_block, bundle['n_views'], bundle['n_classes'],
                                  bundle['manual_scaler'].transform(manual), oh)
    prob = bundle['meta'].predict_proba(meta_in)
    if bundle.get('temperature') is not None:
        prob = bundle['temperature'].predict_proba(prob)
    pred_idx = (bundle['decision'].predict(prob) if bundle.get('decision') is not None
                else np.argmax(prob, axis=1))
    return bundle['le'].inverse_transform(pred_idx), prob


# ═══════════════════════════════════════════════════════════════════
# 9. CHẠY CHÍNH — ĐA SEED
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from seeds_config import SEEDS

    GROUP_COL = 'near_dup_group_id'
    # [OPT-38] Giảm từ 30 → 25 vì đã có MedianPruner (cắt ~30–40% trial tệ từ
    # fold đầu). Nếu cần khớp CHÍNH XÁC Table 8 của bài đã công bố thì đặt lại
    # 30 và tắt pruner — nhưng lưu ý không gian tìm kiếm đã đổi ([OPT-37]) nên
    # con số vốn đã không so sánh trực tiếp được với bảng cũ.
    # [ALIGN-2026-08] [B] 25 → 30: bài báo (Table 8) ghi 30 trial/hệ thống.
    # Ngân sách tìm kiếm lệch nhau thì phần chênh F1 giữa V40 không quy được về
    # kiến trúc — reviewer sẽ hỏi ngay. Chi phí thêm ~20% thời gian mỗi seed.
    N_TRIALS_MULTISEED = 30

    logger.info("#" * 60)
    logger.info(f"CHẠY ĐA SEED V40 ({SUBVERSION}) — {len(SEEDS)} seed: {SEEDS} "
                f"— group_col='{GROUP_COL}'")
    logger.info("#" * 60)

    METRIC_KEYS = ['f1_comparable', 'f1', 'accuracy', 'f1_argmax', 'delta_decision_layer',
                   'ece', 'ece_em', 'ece_cw', 'brier',
                   'training_time_sec', 'model_size_mb', 'lexicon_coverage_train',
                   'lexicon_coverage_test', 'lexicon_coverage_before_expand',
                   'lexicon_size']
    results, summary = {}, {}
    for i, s in enumerate(SEEDS, 1):
        logger.info(f"\n########## [V40] SEED {i}/{len(SEEDS)} = {s} ##########")
        metrics = train_v40(input_dir='input_data', n_trials=N_TRIALS_MULTISEED,
                            seed=s, group_col=GROUP_COL)
        results[s] = metrics
        summary = {}
        for key in METRIC_KEYS:
            vals = [results[s2][key] for s2 in results if key in results[s2]]
            if vals:
                summary[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        # Lưu ngay sau MỖI seed — mất kết nối giữa chừng vẫn giữ được phần đã chạy.
        with open('results_v40_multiseed.json', 'w', encoding='utf-8') as f:
            json.dump({'version': VERSION, 'subversion': SUBVERSION, 'seeds': SEEDS,
                       'group_col': GROUP_COL, 'n_trials': N_TRIALS_MULTISEED,
                       'config': asdict(CFG),
                       'metrics_by_seed': {str(k): v for k, v in results.items()},
                       'summary': summary, 'completed_seeds': len(results),
                       'total_seeds': len(SEEDS)},
                      f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ V40 ({SUBVERSION}) — qua {len(SEEDS)} seed:")
    for key in METRIC_KEYS:
        if key in summary:
            logger.info(f"   {key:<24s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    logger.info("📁 Đã lưu results_v40_multiseed.json")
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # [V50 #4] KIỂM TRA TÁI LẬP so với lần chạy trước
    #
    # Lần chạy này CHỈ được phép thêm artifact (xác suất per-sample, phân
    # nhóm), tuyệt đối không đổi con số — vì PhoBERT đã chạy xong và
    # bảng so sánh hai hệ đã dựng trên số V40 cũ. Nếu F1 lệch thì hoặc môi
    # trường Kaggle đã đổi, hoặc một thay đổi tưởng là vô hại hoá ra không.
    # Phát hiện ngay ở đây rẻ hơn nhiều so với phát hiện lúc viết bài.
    #
    # Đặt results_v40_multiseed_REFERENCE.json cạnh script để bật kiểm tra.
    # ══════════════════════════════════════════════════════════════
    REF_PATH = 'results_v40_multiseed_REFERENCE.json'
    if os.path.exists(REF_PATH):
        try:
            ref = json.load(open(REF_PATH, encoding='utf-8'))['metrics_by_seed']
            logger.info("🔁 [V50 #4] KIỂM TRA TÁI LẬP so với lần chạy trước:")
            drift, missing = [], []
            for s in SEEDS:
                k = str(s)
                if k not in ref or k not in results and s not in results:
                    missing.append(s); continue
                new = results[s]['f1_comparable']
                old = ref[k].get('f1_comparable')
                if old is None:
                    missing.append(s); continue
                d = new - old
                flag = '' if abs(d) < 1e-6 else ('  ← LỆCH' if abs(d) > 1e-4 else '  (lệch nhỏ)')
                logger.info(f"     seed {s}: cũ {old:.6f} → mới {new:.6f}  Δ={d:+.6f}{flag}")
                if abs(d) > 1e-4:
                    drift.append((s, old, new, d))
            if drift:
                logger.error("=" * 70)
                logger.error(f"❌ [V50 #4] {len(drift)}/{len(SEEDS)} seed KHÔNG tái lập được "
                             f"(ngưỡng 1e-4 trên f1_comparable).")
                logger.error("   Bản V50 chỉ THÊM việc ghi tệp, không đụng vào bất kỳ phép")
                logger.error("   tính nào — nên lệch số gần như chắc chắn đến từ MÔI TRƯỜNG,")
                logger.error("   không phải từ mã. So requirements_pinned.txt của hai phiên.")
                logger.error("   HỆ QUẢ: bảng so sánh ba hệ đang dùng số V40 CŨ. Nếu giữ số")
                logger.error("   mới thì phải cập nhật cả bảng; nếu giữ số cũ thì tệp xác")
                logger.error("   suất mới KHÔNG khớp với số cũ và không dùng chung được.")
                logger.error("   Phải chọn một, và ghi rõ lựa chọn vào nhật ký thí nghiệm.")
                logger.error("=" * 70)
            elif not missing:
                logger.info(f"   ✓ Cả {len(SEEDS)}/{len(SEEDS)} seed tái lập khớp — "
                            f"tệp xác suất mới dùng chung được với bảng so sánh hiện có.")
            if missing:
                logger.warning(f"   ⚠ Không đối chiếu được {len(missing)} seed: {missing}")
        except Exception as e:
            logger.warning(f"[V50 #4] Không đọc được {REF_PATH}: {type(e).__name__}: {e}")
    else:
        logger.info(f"ℹ️  [V50 #4] Không thấy {REF_PATH} → bỏ qua kiểm tra tái lập. "
                    f"Đặt tệp results_v40_multiseed.json của lần chạy trước vào đây "
                    f"(đổi tên) để bật kiểm tra.")

    # ══════════════════════════════════════════════════════════════
    # [V50] BÁO CÁO HẬU KIỂM — bắt thiếu sót ngay tại chỗ
    # ══════════════════════════════════════════════════════════════
    logger.info("─" * 60)
    n_sub = sum(1 for m in results.values() if m.get('subgroup_results'))
    logger.info(f"   Seed có 'subgroup_results' : {n_sub}/{len(results)}"
                + ("  ✓" if n_sub == len(results) and results else "  ← THIẾU"))
    n_p = len(glob.glob('reports/v40_probs/v40_test_seed_*.csv'))
    logger.info(f"   Tệp xác suất test          : {n_p}/{len(results)}"
                + ("  ✓" if n_p == len(results) else "  ← THIẾU"))
    if summary.get('f1_comparable'):
        sd = summary['f1_comparable']['std']
        logger.info(f"   Độ lệch chuẩn f1_comparable: {sd:.4f}"
                    + ("  ✓" if sd <= 0.004 else "  ← CAO BẤT THƯỜNG"))
        if sd > 0.004:
            logger.warning("     V40 lần chạy trước đạt 0,0018 và PhoBERT 0,0017. "
                           "Std cao hơn nhiều thường có nghĩa 10 seed KHÔNG chạy cùng "
                           "một điều kiện — kiểm tra trước khi báo cáo.")
    logger.info("")
    logger.info("⚠️  TRƯỚC KHI ĐÓNG PHIÊN KAGGLE, lưu lại:")
    logger.info("      • results_v40_multiseed.json")
    logger.info("      • split_fingerprints.json")
    logger.info("      • reports/v40_probs/        ← xác suất per-sample")
    logger.info("      • models/production/v40/    ← cần cho gold test ở bước sau")
    logger.info("=" * 60)
