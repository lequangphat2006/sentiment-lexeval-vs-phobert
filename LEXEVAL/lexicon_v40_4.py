#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# lexicon_v40_4.py — TỪ ĐIỂN CẢM XÚC TMĐT TIẾNG VIỆT + MỞ RỘNG BÁN GIÁM SÁT
# ───────────────────────────────────────────────────────────────────────────
# Module này giải quyết [LEX-47] — nguyên nhân đã xác minh của việc độ phủ
# lexicon đứng ở 54,x% trong v40.3.
#
# BỐI CẢNH (đo trên input_data/train.csv, 68.661 mẫu):
#
#   • Nhánh tính độ phủ là  _score() → normalize_for_lexicon()
#     → _tokenize_for_lexicon(), và KHÔNG hàm nào gọi word_tokenize().
#     underthesea chỉ được dùng ở _segment() (nhánh TF-IDF) và pos_tag().
#     Đo trên cùng 20k mẫu: chưa cài underthesea → 0,5457
#                           ĐÃ cài (HAS_POS=True) → 0,5456
#     ⇒ Cài underthesea KHÔNG THỂ làm đổi độ phủ. Đây là chẩn đoán sai
#       phổ biến nhất với con số này, nên ghi lại ở đây để lần sau không
#       mất thời gian đi cài lại thư viện.
#
#   • Nguyên nhân thật: lexicon v40.3 chỉ có 70 mục (34 POS + 36 NEG).
#     PHAN_TICH_VA_CAI_TIEN_V40.md §[C6] đã ghi "cần 800–2.000 mục" và §3.2
#     đề xuất mở rộng bán giám sát bằng PMI/delta-IDF — nhưng §3.2 chưa hề
#     được implement (grep 'pmi|delta_idf|expand_lexicon' trên train_v40.py
#     v40.3 → 0 kết quả). Module này chính là phần còn thiếu đó.
#
#   • Lớp từ thiếu nặng nhất là TỪ GHÉP / TỪ LÁY cảm xúc — chiếm phần lớn
#     review TMĐT tiếng Việt. Bằng chứng: lọc các token phân biệt POS/NEG
#     mạnh nhất mà lexicon KHÔNG có (df ≥ 250) thì gần như toàn bộ là ÂM
#     TIẾT LẺ của từ ghép: thoải/mái, hụt/hẫng, phũ/phàng, cồng/kềnh,
#     loang/lổ, lạch/cạch, nồng/nặc, y/hệt…
#
# ⚠ CẢNH BÁO QUAN TRỌNG — ĐỪNG "SỬA" BẰNG CÁCH BẬT SEGMENTATION CHO LEXICON.
#   Đã thử: độ phủ GIẢM từ 0,5420 xuống 0,4759. Lý do: underthesea gộp
#   nhanh→nhanh_chóng, chậm→chậm_trễ, đẹp→đẹp_đẽ… và các từ ghép đó không
#   có trong lexicon 70 mục. Segmentation chỉ có ích SAU KHI lexicon đã
#   được viết ở dạng từ ghép — tức là sau module này.
#
# ───────────────────────────────────────────────────────────────────────────
# KẾT QUẢ ĐO ĐƯỢC (20.000 mẫu, seed 42, split group-aware; macro-F1 là F1
# của một LogisticRegression CHỈ dùng 22 cột thủ công — proxy nhanh cho chất
# lượng khối lexicon, KHÔNG phải F1 của cả hệ thống):
#
#   v40.3 nguyên bản                 70 mục   cov 0,5420 / 0,5495   F1 0,5264
#   + E1–E4 + hạt giống thủ công    267 mục   cov 0,7831 / 0,7970   F1 0,6538
#   + mở rộng bán giám sát          667 mục   cov 0,9898 / 0,9915   F1 0,7140
#
#   Khe hở Train/Test ở bước cuối = −0,0017 (Test CAO HƠN Train) ⇒ mức tăng
#   độ phủ này không phải do quá khớp chỉ số.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import collections
import logging

import numpy as np

logger = logging.getLogger('SentimentV40')

# ═══════════════════════════════════════════════════════════════════
# 1. HẠT GIỐNG THỦ CÔNG — TỪ GHÉP / TỪ LÁY CẢM XÚC TMĐT
# ───────────────────────────────────────────────────────────────────
# Phần này KHÔNG dùng nhãn ⇒ không gây rò rỉ, dùng được cho Test và dữ liệu
# mới, và là phần an toàn để mô tả trong báo cáo ("từ điển do người soạn").
# Thang điểm giữ đúng quy ước v40.3: 1 = yếu, 2 = trung bình, 3 = mạnh.
# ═══════════════════════════════════════════════════════════════════
SEED_POSITIVE = {
    # ── cảm giác / chất liệu ──
    'thoải mái': 2, 'dễ chịu': 2, 'êm ái': 2, 'mềm mại': 2, 'mượt mà': 2,
    'dày dặn': 2, 'chắc chắn': 2, 'gọn gàng': 2, 'nhẹ nhàng': 1,
    'thoáng mát': 2, 'dịu nhẹ': 2, 'thơm': 2, 'thơm nhẹ': 2,
    'thoang thoảng': 1,
    # ── thẩm mỹ ──
    'xinh': 2, 'xinh xắn': 2, 'sang trọng': 2, 'tinh tế': 2, 'hài hoà': 2,
    'hài hòa': 2, 'bắt mắt': 2, 'sắc sảo': 2, 'xịn sò': 3,
    # ── khớp mô tả (trục quan trọng nhất của review TMĐT) ──
    'y hệt': 2, 'giống hình': 2, 'giống ảnh': 2, 'đúng mô tả': 2,
    'đúng size': 1, 'đúng hình': 2, 'chuẩn': 2, 'chuẩn size': 2,
    'vừa ý': 2, 'như mong đợi': 2, 'chính hãng': 2, 'auth': 2,
    # ── dịch vụ / giao hàng ──
    'giao nhanh': 2, 'đóng gói kỹ': 2, 'bọc kỹ': 2, 'kỹ càng': 2,
    'tỉ mỉ': 2, 'phản hồi nhanh': 2, 'tư vấn kỹ': 2, 'nhanh gọn': 2,
    # ── giá trị ──
    'hợp lý': 1, 'phải chăng': 1, 'đáng đồng tiền': 3, 'tiết kiệm': 1,
    # ── hành vi mua lại / khuyến nghị ──
    'ủng hộ': 2, 'sẽ mua lại': 3, 'mua lại': 2, 'recommend': 3, 'khen': 2,
    'đáng thử': 2, 'hết ý': 3, 'quá đẹp': 3, 'siêu đẹp': 3,
    'tuyệt cú mèo': 3, 'good': 2, 'nice': 2, 'perfect': 3, 'ưng': 2,
    'ưng ý': 2,
}

SEED_NEGATIVE = {
    # ── hư hại vật lý ──
    'trầy xước': 2, 'sứt sẹo': 2, 'bong tróc': 3, 'nứt nẻo': 3, 'nứt': 3,
    'vỡ': 3, 'gãy': 3, 'rách': 3, 'bung': 2, 'tuột': 2, 'móp méo': 2,
    'biến dạng': 3, 'ọp ẹp': 3, 'lỏng lẻo': 2, 'rời rạc': 2,
    'lạch cạch': 2, 'lung lay': 2, 'thô ráp': 2, 'lởm chởm': 2,
    'nhăn nhúm': 2,
    # ── mùi / vệ sinh / màu ──
    'mùi hắc': 3, 'nồng nặc': 3, 'hôi rình': 3, 'có mùi': 2, 'mùi lạ': 2,
    'mốc': 3, 'ẩm mốc': 3, 'bụi bẩn': 2, 'loang lổ': 2, 'xỉn màu': 2,
    'phai màu': 2, 'ố': 2, 'nhoè': 2, 'nhòe': 2, 'mờ': 2,
    # ── lệch mô tả ──
    'không đúng mô tả': 3, 'khác mô tả': 3, 'khác hình': 3, 'khác ảnh': 3,
    'không giống': 2, 'sai màu': 2, 'sai size': 2, 'sai hàng': 3,
    'thiếu hàng': 3, 'hàng tồn': 3, 'hàng dạt': 3, 'lởm': 3, 'dởm': 3,
    # ── cảm xúc ──
    'hụt hẫng': 3, 'phũ phàng': 3, 'ức chế': 3, 'bức xúc': 3, 'bực': 3,
    'hối hận': 3, 'chán ghét': 3, 'thất vọng nặng': 3, 'tệ nhất': 3,
    'kinh khủng': 3, 'thảm hoạ': 3, 'thảm họa': 3, 'ám ảnh': 2,
    # ── dịch vụ / giá trị ──
    'cẩu thả': 3, 'sơ sài': 2, 'lôm côm': 3, 'chậm chạp': 2, 'lề mề': 2,
    'trễ hẹn': 2, 'mất hút': 3, 'không phản hồi': 3, 'chặt chém': 3,
    'mất tiền': 3, 'phí tiền': 3, 'không đáng': 3, 'không đáng tiền': 3,
    'tiền mất tật mang': 3, 'cảnh báo': 3, 'tẩy chay': 3, 'report': 3,
    'không mua nữa': 3, 'bad': 3, 'terrible': 3,
    # ── kích thước / cảm giác dùng ──
    'cồng kềnh': 2, 'chật': 2, 'chật cứng': 2, 'bí nóng': 2,
    'nóng bức': 2, 'xù': 2, 'nhạt': 2, 'không ổn': 2, 'khó dùng': 2,
    'khó lắp': 2,
}

# ═══════════════════════════════════════════════════════════════════
# 2. [E3] TOKEN ĐIỂM SỐ — RATING_PATTERNS sinh score_* nhưng lexicon v40.3
#    KHÔNG có mục nào tên score_* ⇒ sinh token rồi bỏ (26 lần trên tập này).
# ═══════════════════════════════════════════════════════════════════
SCORE_POSITIVE = {'score_8': 2, 'score_9': 3, 'score_10': 3,
                  'score_8_10': 2, 'score_9_10': 3, 'score_10_10': 3}
SCORE_NEGATIVE = {'score_0': 3, 'score_1': 3, 'score_2': 3, 'score_3': 2,
                  'score_0_10': 3, 'score_1_10': 3, 'score_2_10': 3,
                  'score_3_10': 2}

# ═══════════════════════════════════════════════════════════════════
# 3. BIẾN THỂ KHÔNG DẤU — CHỈ dạng NHIỀU ÂM TIẾT
# ───────────────────────────────────────────────────────────────────
# 9,0% review trong train.csv được viết hoàn toàn không dấu, nên cần lớp này.
# NHƯNG tự sinh biến thể không dấu cho MỌI mục là một cái bẫy: nó đẩy độ phủ
# từ 54,6% lên 61,3% mà phần lớn là khớp SAI. Đo trên train.csv:
#
#   qua (từ 'quá')  : 5.795 review — thực tế là "qua" (đi qua)
#   em  (từ 'êm')   : 4.817 review — thực tế là "em" (đại từ)
#   ban (từ 'bẩn')  : 1.876 review — thực tế là "bạn" / "bán"
#   gia (từ 'giả')  : 1.497 review — thực tế là "giá"
#   loi (từ 'lỗi')  :   563 review — thực tế là "lời"
#   lam (từ 'lắm')  :   325 review — thực tế là "làm"
#   yeu (từ 'yêu')  :   141 review, 138 trong đó là NEGATIVE — thực tế là
#                       "yếu", tức NGƯỢC DẤU hoàn toàn
#
# Vì vậy chỉ giữ dạng nhiều âm tiết, không đụng từ thông dụng nghĩa khác.
# ═══════════════════════════════════════════════════════════════════
NODIAC_POSITIVE = {
    'tuyet': 3, 'tuyet voi': 3, 'xuat sac': 3, 'hoan hao': 3,
    'hai long': 2, 'chat luong': 2, 'dang mua': 2, 'dang tien': 2,
    'nhiet tinh': 2, 'chu dao': 2, 'can than': 2, 'than thien': 2,
    'sac net': 2, 'vua van': 1, 'thoai mai': 2, 'de chiu': 2,
    'dung mo ta': 2, 'giao nhanh': 2, 'chinh hang': 2, 'dong goi ky': 2,
}
NODIAC_NEGATIVE = {
    'te hai': 3, 'that vong': 3, 'kho chiu': 3, 'buc minh': 3,
    'lua dao': 3, 'vo trach nhiem': 3, 'mat thoi gian': 2,
    'khong hai long': 2, 'cau tha': 3, 'so sai': 2, 'hut hang': 3,
    'cong kenh': 2, 'khac mo ta': 3, 'sai size': 2, 'sai mau': 2,
    'khong dung mo ta': 3, 'tray xuoc': 2, 'phi tien': 3,
}


def build_extended_lexicon(base_positive: dict, base_negative: dict):
    """Gộp lexicon gốc v40.3 với các khối hạt giống thủ công.
    Trả về (positive, negative) MỚI, không sửa dict đầu vào."""
    pos = dict(base_positive)
    neg = dict(base_negative)
    for src in (SEED_POSITIVE, SCORE_POSITIVE, NODIAC_POSITIVE):
        pos.update(src)
    for src in (SEED_NEGATIVE, SCORE_NEGATIVE, NODIAC_NEGATIVE):
        neg.update(src)
    # An toàn: một mục không được vừa POS vừa NEG. Ưu tiên bản gốc v40.3.
    for w in list(pos):
        if w in neg and w not in base_positive:
            del pos[w]
    return pos, neg


# ═══════════════════════════════════════════════════════════════════
# 4. MỞ RỘNG BÁN GIÁM SÁT — §3.2 (phần chưa từng được implement)
# ═══════════════════════════════════════════════════════════════════
# Danh sách chặn: từ chức năng và danh từ chủ đề trung tính. Giữ chúng ngoài
# lexicon để "từ điển cảm xúc" không biến thành "bộ đếm từ khoá chủ đề".
BLOCKLIST = set("""
là này mà hàng thì màu mua mình cho shop một được lần có nhận giao đúng cái rồi đặt ngày ra lại
bên như về để và chưa sau nói thêm đơn giá kèm tay nữa bán đã ai đi qua với phải luôn tôi nên
người mấy hôm em thấy mới ảnh tui cả hơn size vào tiền bị ngoài tớ mọi đợt con nhưng theo vừa
nhiều nước thôi mất chỉ khi nhà tuần gì cửa chủ chắc nhỏ báo cũng sinh hết vậy khác tả ngay còn
túi nào da lên kỹ đầu thế sẽ mang cập kiểu tin sơ sale shopee lazada tiki order link inbox
mã voucher freeship tháng năm giờ phút lúc trước nay mai kia ạ nhé nha ha vâng dạ
""".split())


def delta_logodds_expand(tokenized_docs, labels, *, already: set,
                         min_df=30, min_abs_logodds=0.9, top_k=400,
                         pos_label='positive', neg_label='negative',
                         blocklist=None):
    """Mở rộng lexicon bằng delta log-odds CÓ LÀM TRƠN
    (Monroe, Colaresi & Quinn, *Political Analysis* 2008) — bản có hiệu
    chỉnh tần suất của PMI mà §3.2 đề xuất. Chọn công thức này thay vì PMI
    thuần vì PMI thuần thiên vị nặng về từ tần số thấp, đúng thứ sinh ra
    lexicon toàn từ hiếm nhiễu.

        score(w) = log( (df_pos(w)+α) / N_pos ) − log( (df_neg(w)+α) / N_neg )

    Xếp hạng theo |score| × log(1+df) để KHÔNG nhặt từ hiếm có |score| lớn
    một cách ngẫu nhiên.

    ⚠ HÀM NÀY DÙNG NHÃN. Chỉ được truyền vào tập TRAIN, và phải gọi lại cho
      MỖI seed (mỗi seed có split khác nhau). Gọi trên toàn bộ dataframe là
      rò rỉ nhãn sang Test và sẽ làm F1 báo cáo cao giả.

    Tham số
    -------
    tokenized_docs : iterable[list[str]]  — token ĐÃ qua _tokenize_for_lexicon
    labels         : iterable[str]        — nhãn tương ứng
    already        : set[str]             — mục đã có (POS ∪ NEG ∪ INTENS ∪ NEG_WORDS)

    Trả về (add_positive: dict, add_negative: dict, diagnostics: dict)
    """
    blocklist = BLOCKLIST if blocklist is None else blocklist
    doc_freq = collections.defaultdict(collections.Counter)
    n_doc = collections.Counter()

    for toks, label in zip(tokenized_docs, labels):
        label = str(label)
        n_doc[label] += 1
        doc_freq[label].update(
            {w for w in toks if len(w) > 1 and not w.isdigit()})

    if not n_doc[pos_label] or not n_doc[neg_label]:
        logger.warning("[LEX-47] Không thấy lớp '%s'/'%s' trong Train — "
                       "bỏ qua mở rộng lexicon.", pos_label, neg_label)
        return {}, {}, {'n_candidates': 0, 'n_added': 0}

    vocab = set().union(*[set(c) for c in doc_freq.values()])
    cands = []
    for w in vocab:
        if w in already or w in blocklist or w.replace('_', ' ') in blocklist:
            continue
        p = doc_freq[pos_label][w]
        q = doc_freq[neg_label][w]
        total = sum(doc_freq[c][w] for c in doc_freq)
        if total < min_df:
            continue
        lo = (np.log((p + 0.5) / n_doc[pos_label])
              - np.log((q + 0.5) / n_doc[neg_label]))
        if abs(lo) < min_abs_logodds:
            continue
        cands.append((w, float(lo), int(total)))

    cands.sort(key=lambda r: -abs(r[1]) * np.log1p(r[2]))

    add_pos, add_neg = {}, {}
    for w, lo, _tot in cands[:top_k]:
        score = 3 if abs(lo) > 2.5 else (2 if abs(lo) > 1.5 else 1)
        (add_pos if lo > 0 else add_neg)[w] = score

    diag = {
        'n_candidates': len(cands),
        'n_added': len(add_pos) + len(add_neg),
        'n_added_pos': len(add_pos),
        'n_added_neg': len(add_neg),
        'top_20': [(w, round(lo, 3), tot) for w, lo, tot in cands[:20]],
    }
    return add_pos, add_neg, diag


# ═══════════════════════════════════════════════════════════════════
# 5. LƯU Ý KHI VIẾT BÁO CÁO
# ═══════════════════════════════════════════════════════════════════
REPORTING_NOTE = """\
Danh sách do delta_logodds_expand() sinh ra CẦN ĐƯỢC DUYỆT TAY trước khi mô
tả trong báo cáo. Một phần là từ cảm xúc thật (thoải_mái, hụt_hẫng,
loang_lổ, bức_xúc, cẩu_thả) nhưng một phần là đặc thù ngữ liệu chứ không
phải từ cảm xúc — trên tập này có 'photoshop', 'review', 'mưa', 'cao_cấp',
'du_lịch'. Quy trình lọc nên nêu rõ: (a) giữ lại chỉ mục được pos_tag gán
nhãn A (tính từ) hoặc R (trạng từ) — underthesea đã cài nên làm được;
(b) ghi số mục trước và sau khi lọc; (c) nêu rõ từ điển được học TRÊN TẬP
TRAIN CỦA TỪNG SEED, không phải trên toàn bộ dữ liệu."""
