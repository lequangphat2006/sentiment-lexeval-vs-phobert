#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# PHOBERT BASELINE — cho AERS-Pro V40 (so sánh với SOTA transformer)
# ───────────────────────────────────────────────────────────────────────────
# Mục tiêu: fine-tune vinai/phobert-base(-v2) trên CÙNG một cách chia dữ liệu,
# CÙNG bộ hàm phân tích (subgroup, distribution check, entropy/confidence,
# ECE/Brier) như train_v40.py, để kết quả có thể đưa thẳng vào bảng so sánh
# V40 / PhoBERT mà không lệch protocol đánh giá.
#
# BẢN CẬP NHẬT NÀY: chạy ĐA SEED (10 seed, dùng CHUNG seeds_config.py với
# V40) thay vì chỉ 1 seed như bản trước — để có mean±std thật sự có thể
# đặt cạnh Table so sánh V40 (vốn đã chạy đủ 10 seed). Cấu trúc vòng lặp
# đa seed, lưu kết quả sau MỖI seed, và format results_*.json copy Y HỆT
# pattern trong train_v40.py (xem phần "CHẠY CHÍNH — ĐA SEED"
# ở cuối 2 file đó) để 2 file JSON có thể gộp thẳng vào cùng một bảng.
#
# ĐÃ ĐIỀU CHỈNH CHO MÁY THẬT SẼ DÙNG ĐỂ CHẠY: Kaggle Notebook, GPU T4 x2
# (2 x NVIDIA Tesla T4, 16GB VRAM/GPU, kiến trúc Turing TU104 — CÓ Tensor
# Core, khác hẳn GTX 16-series). Không còn ràng buộc VRAM 4GB nữa nên "bung
# lụa": tăng batch/max_length, tắt gradient_checkpointing mặc định để tối đa
# tốc độ. Hệ quả cụ thể cho lựa chọn tham số dưới đây:
#   • 2 GPU: HF Trainer tự phát hiện torch.cuda.device_count()==2 và dùng
#     DataParallel (không cần torchrun/DDP trong Kaggle Notebook — chạy
#     `python train_phobert.py` bình thường là đủ). per_device_batch_size
#     áp dụng CHO MỖI GPU, nên effective batch = batch_size * 2 * grad_accum.
#   • fp16 mixed precision: T4 CÓ Tensor Core nên fp16 vừa tiết kiệm VRAM
#     vừa TĂNG TỐC thật (khác hẳn GTX 1650 Ti trước đây) — luôn bật.
#   • gradient_checkpointing=False làm MẶC ĐỊNH: với 16GB/GPU, phobert-base
#     ở max_length=128, batch=32 không cần đánh đổi tốc độ lấy VRAM nữa. Vẫn
#     giữ flag --gradient_checkpointing nếu bạn tăng batch/max_length quá tay
#     và gặp OOM.
#   • batch_size=32 (mỗi GPU) + grad_accum=1 (effective batch = 32*2*1 = 64)
#     làm MẶC ĐỊNH — tận dụng cả 16GB x2, KHÔNG phải cấu hình dè dặt nữa.
#   • max_length=192 (ĐO THỰC TẾ trên dataset_cleaned.csv, 25289 dòng): p50=17
#     từ, p95=65, p97=76, p99=127 từ (word-level, chưa tính BPE) — 128 sẽ cắt
#     cụt ngay ở đuôi p99, trong khi VRAM T4x2 còn dư nhiều nên nâng lên 192
#     để có biên an toàn qua cả bước tách từ (underthesea) + BPE (PhoBERT).
#   • torch.backends.cudnn.benchmark=True: input luôn padding cố định
#     max_length nên bật benchmark giúp cuDNN chọn thuật toán phù hợp.
#   • PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + TOKENIZERS_PARALLELISM=false
#     (set TRƯỚC khi import torch/transformers): giảm phân mảnh bộ nhớ CUDA
#     khi chạy liên tục 10 seed, và tránh cảnh báo/deadlock tokenizer đa luồng
#     hay gặp trong môi trường notebook (Kaggle) khi kết hợp với dataloader
#     nhiều worker.
#
# HỖ TRỢ CHẠY TRÊN KAGGLE (phiên có giới hạn thời gian, /kaggle/working
# KHÔNG tự động giữ qua các phiên trừ khi bạn lưu lại làm output/dataset):
#   • Resume tự động: nếu results_phobert_multiseed.json đã có sẵn TRONG
#     phiên hiện tại (bị ngắt giữa chừng nhưng notebook chưa restart), script
#     tự bỏ qua seed đã xong. Dùng --no_resume nếu muốn ép chạy lại từ đầu.
#   • --resume_from <path>: nếu bạn bắt đầu một PHIÊN KAGGLE MỚI (session cũ
#     đã hết giờ/bị đóng), tải results_phobert_multiseed.json của phiên trước
#     lên làm Kaggle Input Dataset rồi trỏ --resume_from vào đường dẫn đó
#     (vd. /kaggle/input/phobert-results-seed-p1/results_phobert_multiseed.json)
#     — script sẽ nạp lại làm điểm xuất phát rồi chạy tiếp seed còn thiếu.
#     ⚠️ Nhớ tải results_phobert_multiseed.json + thư mục logs/ về máy (hoặc
#     "Save Version" làm Output) TRƯỚC KHI phiên Kaggle hết giờ, nếu không sẽ
#     mất toàn bộ tiến độ của phiên đó.
#   • Sau mỗi seed, thư mục checkpoint tạm của Trainer
#     (models/phobert/checkpoints/seed_*, dùng để resume epoch giữa chừng)
#     bị xoá để không tích luỹ dung lượng qua 10 lần chạy — dùng
#     --keep_checkpoints nếu muốn giữ lại để debug. Đây KHÔNG phải model
#     cuối cùng, chỉ là checkpoint huấn luyện tạm thời của HF Trainer.
#   • Model production (models/production/phobert/) vẫn bị GHI ĐÈ sau mỗi
#     seed (Y HỆT train_v40.py, để giữ đúng schema
#     monitoring_baseline.json) — NHƯNG ngay sau khi ghi, toàn bộ nội dung
#     được sao lưu riêng, ĐẦY ĐỦ (model + tokenizer + label encoder +
#     baseline giám sát) vào models/production/phobert_seed{seed}/ cho CẢ
#     10 SEED, không chỉ seed=42 như bản trước. Mục tiêu: giữ lại đủ 10
#     checkpoint để dùng lại sau này (ensemble, soi lỗi theo seed...),
#     không chỉ để tính mean±std. Ước tính dung lượng: ~500-540MB/seed x 10
#     ≈ 5-5.5GB. Nếu đĩa giới hạn (Kaggle), dùng --session/--seeds bên dưới
#     để chia thành 5 phiên x 2 seed, tải model+kết quả ra ngoài giữa các
#     phiên rồi xoá bớt trước khi chạy phiên kế tiếp.
#   • --session N (N=1..5) hoặc --seeds "42,43": chỉ chạy ĐÚNG 2 seed của
#     phiên đó thay vì toàn bộ 10 seed còn thiếu. Kết hợp với --resume_from
#     để nối kết quả của các phiên trước vào results_phobert_multiseed.json
#     — mỗi phiên chỉ cần train 2/10 seed nên nhẹ hơn nhiều cho giới hạn
#     thời gian/đĩa của Kaggle. Không truyền cờ nào thì giữ hành vi cũ:
#     chạy hết SEEDS còn thiếu trong một lần.
#
# QUAN TRỌNG — Word segmentation cho PhoBERT:
#   PhoBERT được train trên văn bản tiếng Việt đã TÁCH TỪ (word-segmented,
#   nối bằng "_"), khác với TF-IDF ở V40 vốn hoạt động ở mức token thô.
#   Nếu bỏ qua bước này, hiệu năng PhoBERT sẽ tụt rất nhiều so với số liệu
#   công bố trong paper gốc. Script này dùng underthesea.word_tokenize để
#   tách từ trước khi đưa vào tokenizer của PhoBERT.
#
#   FIX (2026-07, phát hiện trên dataset_cleaned.csv thực tế): ~2.8%
#   (699/25289) comment là review nhiều-trường kiểu Shopee ghép bằng '\n'
#   KHÔNG có khoảng trắng quanh dấu xuống dòng (vd "...tuyệt lắm\nChất
#   liệu:tốt..."). underthesea.word_tokenize không coi '\n' là ranh giới từ
#   như dấu cách nên sẽ dính 2 từ ở "dòng" trước/sau thành 1 token rác
#   ('tuyệt_Đúng') không có trong vocab PhoBERT. Đã thêm _normalize_whitespace()
#   để gộp mọi chuỗi whitespace (\n, \r, \t, nbsp) thành một dấu cách TRƯỚC
#   khi tách từ, tránh dính chữ ở đúng nhóm review nhiều-trường này.
# ═══════════════════════════════════════════════════════════════════════════

import os
# Phải set TRƯỚC khi import torch/transformers để có hiệu lực.
# - expandable_segments: giảm phân mảnh bộ nhớ CUDA khi chạy liên tục 10 seed
#   trong cùng 1 tiến trình dài (mỗi seed nạp/giải phóng model riêng).
# - TOKENIZERS_PARALLELISM=false: tránh cảnh báo/deadlock tokenizer đa luồng
#   khi kết hợp với dataloader_num_workers>0 trong môi trường notebook.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import re
import sys, glob, json, time, random, shutil, argparse, logging, logging.handlers
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback,
                           set_seed)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from underthesea import word_tokenize
    HAS_UNDER = True
except ImportError:
    HAS_UNDER = False

VERSION = 'phobert'

if torch.cuda.is_available():
    # Input luôn padding cố định về max_length -> kích thước tensor không đổi
    # giữa các batch, nên bật benchmark an toàn và có lợi cho tốc độ trên T4.
    torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════
# LOGGING (cùng định dạng với train_v40.py) — MỘT file log
# duy nhất cho toàn bộ phiên chạy đa seed (dễ theo dõi ETA/tổng thời gian).
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
_log_file = os.path.join(LOG_DIR, f"train_{VERSION}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger('SentimentPhoBERT')
logger.setLevel(logging.DEBUG)
_fh = logging.handlers.RotatingFileHandler(_log_file, maxBytes=15_000_000, backupCount=5, encoding='utf-8')
_ch = logging.StreamHandler(sys.stdout)
_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
for h in [_fh, _ch]:
    h.setFormatter(_fmt)
    logger.addHandler(h)

for d in ['models', 'models/production', 'reports']:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING — GIỮ NGUYÊN LOGIC NHƯ train_v40.py (không đổi để so
#    sánh công bằng: cùng file input, cùng cách nhận diện cột). Có thêm
#    group_col để đọc cột 'near_dup_group_id' — ĐỒNG BỘ với train_v40.py/
#    train_v40.py (Trục 1, mục 2.2), bắt buộc để so sánh 2 hệ thống trên
#    cùng một split sạch, không tái phạm leakage đã audit trước đó.
# ═══════════════════════════════════════════════════════════════════
def load_data(input_dir='input_data', group_col=None):
    files = glob.glob(os.path.join(input_dir, '*.csv')) + glob.glob(os.path.join(input_dir, '*.xlsx'))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy .csv/.xlsx nào trong '{input_dir}'. "
                                 f"Copy đúng dữ liệu đã dùng cho train_v40 vào đây để so sánh công bằng.")
    dfs = []
    for f in files:
        df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
        df.rename(columns=lambda c: 'comment' if c.lower() in ('comment', 'text')
                   else ('sentiment' if c.lower() in ('sentiment', 'label') else c), inplace=True)
        cols = ['comment', 'sentiment']
        if group_col and group_col in df.columns and group_col not in cols:
            cols.append(group_col)
        if 'comment' in df.columns and 'sentiment' in df.columns:
            dfs.append(df[cols])
    if not dfs:
        raise ValueError("Không file nào có đủ cột 'comment' và 'sentiment'.")
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
        logger.warning(
            f"⚠️  [ALIGN-2026-08] Đã BỎ {{_n_dropped}}/{{_n_before}} dòng có comment/sentiment rỗng hoặc NaN "
            f"khi nạp từ '{{input_dir}}' — giống hệt load_data() của train_v40.py.")
    if len(out) == 0:
        raise ValueError(f"load_data(): sau khi loại dòng rỗng/NaN, không còn dòng hợp lệ trong '{{input_dir}}'.")
    return out.reset_index(drop=True)


def split_data_phobert(df, y, group_col=None, seed=42):
    """
    [SPLIT-16 / PATCH] ĐỒNG BỘ với split_data_v40() HIỆN TẠI (train_v40.py).

    LỊCH SỬ: bản trước của hàm này gọi GroupShuffleSplit và docstring ghi
    "ĐỒNG BỘ với split_data_v40()" — đúng tại thời điểm đó.
    Nhưng sau đó split_data_v40() được vá theo note [SPLIT-16]: GroupShuffleSplit
    group-aware nhưng KHÔNG hỗ trợ stratify theo nhãn, nên V40 đổi sang lấy 1
    fold từ StratifiedGroupKFold (group-aware VÀ xấp xỉ cân bằng lớp).
    split_data_phobert() KHÔNG được cập nhật theo — hệ quả: với cùng 1 seed,
    V40 và PhoBERT sinh ra 3 tập Train/Calibration/Test có kích thước khác
    nhau (đối chiếu log gốc, seed=42: V40 Train=14849/Cal=5125/Test=5315 vs
    PhoBERT (bản cũ) Train=15111/Cal=5003/Test=5175 — 0/10 seed khớp nhau).
    Điều này làm câu "exact same train/calibration/test partition" ở Mục 3.3
    của bài báo không đúng với thực tế đã chạy.

    BẢN VÁ NÀY: copy nguyên logic StratifiedGroupKFold(n_splits=5 rồi 4) từ
    split_data_v40() hiện tại — cùng thứ tự gọi, cùng random_state=seed, cùng
    2 tầng fallback (GroupShuffleSplit rồi stratified train_test_split) — để
    V40 và PhoBERT lại sinh ra đúng cùng 1 cách chia trên cùng seed, khôi phục
    đúng tuyên bố "exact same partition". Cần chạy lại toàn bộ 10 seed sau khi
    áp bản vá này; số liệu PhoBERT hiện có (results_phobert_multiseed.json)
    được sinh ra từ bản GroupShuffleSplit cũ nên KHÔNG dùng chung được với
    lần chạy mới.
    """
    idx_all = np.arange(len(df))
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].values
        try:
            sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            idx_tr_cal, idx_te = next(iter(sgkf1.split(idx_all, y, groups)))
            sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            rel_tr, rel_cal = next(iter(sgkf2.split(idx_tr_cal, y[idx_tr_cal], groups[idx_tr_cal])))
            idx_tr, idx_cal = idx_tr_cal[rel_tr], idx_tr_cal[rel_cal]
            logger.info(f"[SPLIT-16] Chia dữ liệu bằng StratifiedGroupKFold (lấy 1 fold, 60/20/20) theo "
                        f"cột '{group_col}' — group-aware VÀ xấp xỉ cân bằng lớp.")
            return idx_tr, idx_cal, idx_te
        except Exception as e:
            logger.warning(f"[SPLIT-16] StratifiedGroupKFold lỗi ({e}) — fallback GroupShuffleSplit "
                            f"(group-aware nhưng KHÔNG stratify).")
            try:
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                idx_tr_cal, idx_te = next(gss1.split(idx_all, y, groups))
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                rel_tr, rel_cal = next(gss2.split(idx_tr_cal, y[idx_tr_cal], groups[idx_tr_cal]))
                idx_tr, idx_cal = idx_tr_cal[rel_tr], idx_tr_cal[rel_cal]
                logger.info(f"Chia dữ liệu bằng GroupShuffleSplit theo cột '{group_col}' (Trục 1, mục 2.2) "
                            f"— tránh 2 biến thể cùng nguồn rơi vào 2 tập khác nhau.")
                return idx_tr, idx_cal, idx_te
            except Exception as e2:
                logger.warning(f"GroupShuffleSplit cũng lỗi ({e2}) — fallback stratified split ngẫu nhiên.")

    logger.warning("KHÔNG có cột group/near_dup_group_id hợp lệ → dùng stratified split ngẫu nhiên như "
                    "trước đây. CẢNH BÁO (Trục 1, mục 2.2): số liệu này sẽ KHÔNG so sánh công bằng được "
                    "với V40 nếu hai bản đó đã chạy group-aware — chạy dedup_clustering.py trước.")
    idx_tr_cal, idx_te = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=seed)
    idx_tr, idx_cal = train_test_split(idx_tr_cal, test_size=0.25, stratify=y[idx_tr_cal], random_state=seed)
    return idx_tr, idx_cal, idx_te


_NEWLINE_RE = re.compile(r'[\r\n]+')
_WHITESPACE_RE = re.compile(r'\s+')


def _normalize_whitespace(t: str) -> str:
    """Chuẩn hoá whitespace TRƯỚC khi tách từ.
    PHÁT HIỆN TỪ dataset_cleaned.csv THỰC TẾ (~2.8% / 699/25289 dòng): nhiều
    comment là review Shopee ghép nhiều field liên tiếp bằng '\\n', vd:
        "Chất lượng sản phẩm:gất tuyệt\\nĐúng với mô tả:đúng\\nMũ siêu đẹp..."
    Hai vấn đề với underthesea.word_tokenize khi gặp '\\n':
      (1) nếu KHÔNG có khoảng trắng quanh '\\n' (chữ dính sát), 2 từ ở 2 "dòng"
          liền kề bị dính thành 1 token rác không có trong vocab PhoBERT
          (vd 'tuyệt\\nĐúng' -> 'tuyệt_Đúng' nếu chỉ thay '\\n' bằng dấu cách).
      (2) ngay cả khi CÓ khoảng trắng, thiếu dấu câu ở ranh giới giữa 2 field
          khiến underthesea coi 2 field là một câu liền mạch và vẫn có thể
          ghép nhầm compound qua ranh giới field (đã kiểm chứng thực nghiệm).
    → Thay '\\n'/'\\r' bằng " . " (chấm + khoảng trắng hai bên) để tạo ranh
    giới câu rõ ràng cho underthesea, sau đó gộp các whitespace còn lại
    (tab, nbsp, nhiều dấu cách liên tiếp) về một dấu cách."""
    t = _NEWLINE_RE.sub(' . ', str(t))
    return _WHITESPACE_RE.sub(' ', t).strip()


def word_segment_batch(texts):
    """Tách từ tiếng Việt bằng underthesea trước khi đưa vào PhoBERT tokenizer.
    Nếu thiếu underthesea, cảnh báo rõ vì hiệu năng PhoBERT sẽ giảm đáng kể."""
    if not HAS_UNDER:
        logger.warning("KHÔNG có underthesea → PhoBERT sẽ nhận văn bản CHƯA tách từ, "
                        "hiệu năng có thể thấp hơn số liệu công bố trong paper gốc PhoBERT. "
                        "Cài đặt: pip install underthesea")
        return [_normalize_whitespace(t) for t in texts]
    out = []
    for t in texts:
        try:
            out.append(word_tokenize(_normalize_whitespace(t), format="text"))
        except Exception:
            out.append(_normalize_whitespace(t))
    return out


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                    max_length=max_length, return_tensors=None)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'f1_macro': f1_score(labels, preds, average='macro'),
        'accuracy': accuracy_score(labels, preds),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. PHÂN TÍCH MỞ RỘNG — SAO CHÉP Y HỆT train_v40.py ĐỂ SO SÁNH ĐÚNG
#    PROTOCOL (subgroup, phân phối dự đoán, entropy/confidence)
# ═══════════════════════════════════════════════════════════════════
NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}
HEDGE_WORDS = {'tạm', 'tạm được', 'cũng được', 'bình thường', 'cũng ổn', 'tàm tạm', 'không đến nỗi', 'cũng tạm'}


def _tag_phenomena(text: str) -> dict:
    t = str(text).lower()
    toks = t.split()
    has_neg = any(w in NEGATION_WORDS for w in toks)
    has_hedge = any(h in t for h in HEDGE_WORDS)
    has_contrast = any(m in t for m in ('nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà'))
    is_short = len(toks) < 5
    return {'phu_dinh': has_neg, 'hedge': has_hedge, 'trai_chieu': has_contrast, 'cau_ngan': is_short}


def error_analysis_by_subgroup(raw_texts, y_true, y_pred, le):
    tags = [_tag_phenomena(t) for t in raw_texts]
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    logger.info("── Phân tích lỗi theo phân nhóm ngữ nghĩa (đồng bộ Trục 5 của LexEval) ──")
    results = {}
    for key in ['phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan']:
        mask = np.array([tg[key] for tg in tags])
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
        results[key] = {'n': int(mask.sum()), 'accuracy': float(acc), 'f1_macro': float(f1)}
        logger.info(f"  {key:12s}: n={mask.sum():4d}  acc={acc:.3f}  f1_macro={f1:.3f}")
    return results


def prediction_distribution_check(y_pred, y_train, le, warn_threshold=0.10):
    logger.info("── Kiểm tra lệch phân phối dự đoán so với train ──")
    y_pred, y_train = np.asarray(y_pred), np.asarray(y_train)
    flagged = []
    for ci, cname in enumerate(le.classes_):
        p_pred = np.mean(y_pred == ci)
        p_train = np.mean(y_train == ci)
        diff = p_pred - p_train
        flag = abs(diff) > warn_threshold
        if flag:
            flagged.append(str(cname))
        logger.info(f"  Lớp {cname}: train={p_train:.1%}  predict(test)={p_pred:.1%}  "
                    f"lệch={diff:+.1%}{'  ⚠️' if flag else ''}")
    if flagged:
        logger.warning(f"  Lệch phân phối > {warn_threshold:.0%} ở lớp: {flagged}")
    return flagged


def confidence_entropy_stats(probs):
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    confidence = np.max(probs, axis=1)
    stats = {
        'mean_entropy': float(np.mean(entropy)),
        'mean_confidence': float(np.mean(confidence)),
        'std_confidence': float(np.std(confidence)),
    }
    logger.info("── Entropy/độ tin cậy trung bình ──")
    logger.info(f"  mean_entropy={stats['mean_entropy']:.4f}  mean_confidence={stats['mean_confidence']:.4f} "
                f"(±{stats['std_confidence']:.4f})")
    return stats


def compute_ece_brier(probs, y_true, n_bins=10):
    """Tính ECE (Expected Calibration Error) và Brier score — CÙNG quy ước bin
    với train_v40.py (thống nhất 2026-07: bin đầu [lo,hi], các
    bin sau (lo,hi]) để so sánh calibration công bằng giữa cả 2 hệ thống."""
    y_true = np.asarray(y_true)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[y_true]
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))

    logger.info(f"── Calibration: ECE={ece:.4f}  Brier={brier:.4f} ──")
    return {'ece': float(ece), 'brier': float(brier)}


def save_diagnostic_plots(y_te, pred_te, le, out_dir='reports'):
    try:
        os.makedirs(out_dir, exist_ok=True)
        cm = confusion_matrix(y_te, pred_te)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Dự đoán'); ax.set_ylabel('Thực tế'); ax.set_title(f'Confusion Matrix — Test ({VERSION})')
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'confusion_matrix_{VERSION}.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info(f"📊 Đã lưu confusion matrix vào {out_path}")
    except Exception as e:
        logger.warning(f"Không lưu được confusion matrix: {e}")


def export_phobert_logits(seed, idx_cal, y_cal, logits_cal, probs_cal,
                           idx_te, y_te, logits_te, probs_te,
                           out_dir='reports/phobert_logits'):
    """
    [EXPORT-LOGITS PATCH — Bước 2 của lộ trình nâng cấp] Xuất logits/xác suất
    per-sample cho tập Calibration và Test, dùng làm input cho
    fit_temperature_scaling.py (Temperature Scaling, Bước 5) và cho reliability
    diagram / confusion matrix đối chiếu với V40 (Bước 8).

    QUAN TRỌNG: dùng logits THẬT lấy từ trainer.predict(...).predictions
    (không suy ngược từ prob qua log()), vì log(softmax(z)) chỉ bằng z sai
    khác một hằng số cộng theo mỗi sample — không ảnh hưởng tới việc fit T
    nhưng để đúng đặc tả cột "logit_*" thì nên ghi logits gốc của model.

    row_index là chỉ số dòng TRONG DataFrame gốc đã nạp bởi load_data()
    (tức phần tử của idx_cal/idx_te trả về từ split_data_phobert()) — dùng để
    tra ngược lại văn bản gốc khi làm "phân tích lỗi đối sánh" ở Bước 8.

    Ghi 2 file mỗi seed, đúng đặc tả:
        reports/phobert_logits/phobert_calib_seed_<seed>.csv
        reports/phobert_logits/phobert_test_seed_<seed>.csv
    với các cột: seed, split, row_index, y_true, logit_0, logit_1, logit_2,
    prob_0, prob_1, prob_2.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_classes = probs_cal.shape[1]

    def _save(split_name, idx_orig, y_true, logits, probs):
        logits = np.asarray(logits)
        probs = np.asarray(probs)
        data = {
            'seed': seed,
            'split': split_name,
            'row_index': np.asarray(idx_orig),
            'y_true': np.asarray(y_true),
        }
        for c in range(n_classes):
            data[f'logit_{c}'] = logits[:, c]
        for c in range(n_classes):
            data[f'prob_{c}'] = probs[:, c]
        out_df = pd.DataFrame(data)
        out_path = os.path.join(out_dir, f'phobert_{split_name}_seed_{seed}.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        logger.info(f"💾 [EXPORT-LOGITS] Đã lưu logits/probs {split_name} (seed={seed}) "
                    f"→ {out_path}  ({len(out_df)} dòng, {n_classes} lớp)")
        return out_path

    path_cal = _save('calib', idx_cal, y_cal, logits_cal, probs_cal)
    path_test = _save('test', idx_te, y_te, logits_te, probs_te)
    return path_cal, path_test


# ═══════════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE — nay nhận tham số trực tiếp (giống train_v40()) thay vì argparse.Namespace, để vòng lặp đa seed ở
#    __main__ có thể gọi train_phobert(seed=s, ...) cho từng seed.
# ═══════════════════════════════════════════════════════════════════
def train_phobert(input_dir='input_data',
                   model_name='vinai/phobert-base-v2',
                   max_length=192,
                   batch_size=32,
                   grad_accum=1,
                   epochs=4,
                   lr=2e-5,
                   seed=42,
                   fp16=True,
                   gradient_checkpointing=False,
                   freeze_layers=0,
                   cleanup_checkpoints=True,
                   group_col=None):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_batch = batch_size * max(n_gpu, 1) * grad_accum
    logger.info("=" * 60)
    logger.info(f"🚀 BẮT ĐẦU FINE-TUNE {model_name} (seed={seed})")
    logger.info(f"   max_length={max_length}  batch_size/GPU={batch_size}  n_gpu={n_gpu}  "
                f"grad_accum={grad_accum} (effective batch={effective_batch})  fp16={fp16}  "
                f"gradient_checkpointing={gradient_checkpointing}  freeze_layers={freeze_layers}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("⚠️ KHÔNG phát hiện GPU — script sẽ chạy trên CPU, RẤT CHẬM với transformer. "
                        "Kiểm tra lại Kaggle Notebook đã bật Accelerator = GPU T4 x2 chưa.")
    else:
        for i in range(n_gpu):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}  VRAM: {vram_gb:.1f} GB")
        if n_gpu > 1:
            logger.info(f"🚀 Phát hiện {n_gpu} GPU — Trainer sẽ tự dùng DataParallel, "
                        f"per_device_batch_size={batch_size} áp dụng CHO MỖI GPU "
                        f"(tổng batch trước grad_accum = {batch_size * n_gpu}).")
        elif n_gpu == 1 and batch_size >= 16:
            logger.warning(f"⚠️ Chỉ phát hiện 1 GPU nhưng batch_size={batch_size} (mặc định canh cho T4 x2). "
                            f"Nếu đây không phải phiên T4 x2, cân nhắc giảm --batch_size hoặc thêm "
                            f"--gradient_checkpointing để tránh OOM.")

    # ── 1. Nạp dữ liệu — CÙNG FILE với V40 để so sánh công bằng ──
    df = load_data(input_dir, group_col=group_col)
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    raw_texts = df['comment'].astype(str).tolist()
    logger.info(f"Đã nạp {len(df)} mẫu. Nhãn: {list(le.classes_)}")

    if group_col is not None and group_col not in df.columns:
        logger.warning(f"Cột group_col='{group_col}' không tồn tại trong dữ liệu — bỏ qua group-aware split.")
        group_col = None

    # ── 2. Chia Train/Cal/Test 60/20/20 — group-aware nếu có group_col
    #      (ĐỒNG BỘ với V40 — Trục 1, mục 2.2, bắt buộc để 2 hệ thống
    #      so sánh trên đúng cùng một split sạch) ──────────────────────
    idx_tr, idx_cal, idx_te = split_data_phobert(df, y, group_col=group_col, seed=seed)
    # [ALIGN-2026-08] KIỂM TRA DẤU VÂN TAY SPLIT — xem split_fingerprint.py
    # Dừng NGAY nếu seed này đã được một hệ thống khác chạy với cách chia khác.
    try:
        from split_fingerprint import check_split_fingerprint
        check_split_fingerprint('phobert', seed, idx_tr, idx_cal, idx_te, len(df))
    except ImportError:
        logger.warning("[ALIGN-2026-08] Không tìm thấy split_fingerprint.py cạnh file train — "
                       "BỎ QUA kiểm tra lệch split. Copy file đó vào cùng thư mục để bật lại.")
    logger.info(f"Train: {len(idx_tr)} | Calibration: {len(idx_cal)} | Test: {len(idx_te)}")

    raw_tr = [raw_texts[i] for i in idx_tr]
    raw_cal = [raw_texts[i] for i in idx_cal]
    raw_te = [raw_texts[i] for i in idx_te]
    y_tr, y_cal, y_te = y[idx_tr], y[idx_cal], y[idx_te]

    # ── 3. Word segmentation cho PhoBERT ─────────────────────────────
    logger.info("Tách từ (word segmentation) cho PhoBERT...")
    seg_tr = word_segment_batch(raw_tr)
    seg_cal = word_segment_batch(raw_cal)
    seg_te = word_segment_batch(raw_te)

    # ── 4. Tokenizer & Model ──────────────────────────────────────────
    logger.info(f"Nạp tokenizer/model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(le.classes_)
    )

    if freeze_layers > 0:
        logger.info(f"Đóng băng embeddings + {freeze_layers} layer encoder đầu (tiết kiệm VRAM)...")
        base = getattr(model, model.base_model_prefix)
        for p in base.embeddings.parameters():
            p.requires_grad = False
        encoder_layers = base.encoder.layer
        for i in range(min(freeze_layers, len(encoder_layers))):
            for p in encoder_layers[i].parameters():
                p.requires_grad = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = SentimentDataset(seg_tr, y_tr, tokenizer, max_length)
    cal_ds = SentimentDataset(seg_cal, y_cal, tokenizer, max_length)
    test_ds = SentimentDataset(seg_te, y_te, tokenizer, max_length)

    # ── 5. Training arguments — tối ưu VRAM thấp (GTX 1650 Ti 4GB) ───
    # output_dir riêng theo seed để tránh Trainer nhầm checkpoint của seed
    # trước là điểm resume cho seed sau (mỗi seed phải train from scratch).
    checkpoint_dir = os.path.join('models', VERSION, 'checkpoints', f'seed_{seed}')
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size * 2, 8),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16 and device == "cuda",
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=cal_ds,          # model selection dùng Calibration, KHÔNG dùng Test — cùng nguyên tắc với V40
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Bắt đầu training...")
    trainer.train()

    # ── 6. Đánh giá DUY NHẤT MỘT LẦN trên Test (đúng nguyên tắc của V40) ──
    logger.info("Đánh giá trên tập Test (chạy đúng 1 lần)...")
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_te = np.argmax(probs, axis=1)

    f1_test = f1_score(y_te, pred_te, average='macro')
    acc_test = accuracy_score(y_te, pred_te)
    logger.info(f"✅ F1-Macro TEST (chạy đúng 1 lần) = {f1_test:.4f}")
    logger.info(f"✅ Accuracy TEST = {acc_test:.4f}")
    logger.info("\n" + classification_report(y_te, pred_te, target_names=[str(c) for c in le.classes_]))

    # ── 7. Phân tích mở rộng — cùng bộ hàm với V40 ────────────────────
    subgroup_results = error_analysis_by_subgroup(raw_te, y_te, pred_te, le)
    prediction_distribution_check(pred_te, y_tr, le)
    conf_stats = confidence_entropy_stats(probs)
    calib_stats = compute_ece_brier(probs, y_te)
    save_diagnostic_plots(y_te, pred_te, le)

    # ── [EXPORT-LOGITS PATCH — Bước 2] Lấy logits trên tập Calibration ────
    # (tập Test đã có `logits`/`probs` ở bước 6 phía trên; ở đây chỉ cần
    # chạy thêm 1 lượt predict trên cal_ds — KHÔNG ảnh hưởng tới nguyên tắc
    # "đánh giá Test đúng 1 lần", vì đây là dự đoán trên tập Calibration,
    # phục vụ fit Temperature Scaling ở Bước 5, không phải Test) ──────────
    logger.info("Trích xuất logits trên tập Calibration (phục vụ Temperature Scaling, Bước 5)...")
    cal_pred_output = trainer.predict(cal_ds)
    cal_logits = cal_pred_output.predictions
    cal_probs = torch.softmax(torch.tensor(cal_logits), dim=1).numpy()

    export_phobert_logits(
        seed=seed,
        idx_cal=idx_cal, y_cal=y_cal, logits_cal=cal_logits, probs_cal=cal_probs,
        idx_te=idx_te, y_te=y_te, logits_te=logits, probs_te=probs,
    )

    # ── 8. Lưu model + baseline giám sát (cùng schema với V40) ────────
    # LƯU Ý: prod_dir cố định -> bị GHI ĐÈ mỗi seed, giống hệt train_v40.py/
    # train_v40.py. Vòng lặp đa seed ở __main__ giữ lại toàn bộ chỉ số qua
    # giá trị trả về của hàm này, không phụ thuộc vào file model còn lại.
    prod_dir = os.path.join('models', 'production', VERSION)
    os.makedirs(prod_dir, exist_ok=True)
    trainer.save_model(prod_dir)
    tokenizer.save_pretrained(prod_dir)
    with open(os.path.join(prod_dir, 'label_encoder_classes.json'), 'w', encoding='utf-8') as f:
        json.dump(list(le.classes_), f, ensure_ascii=False)

    model_size_mb = sum(
        os.path.getsize(os.path.join(prod_dir, fn))
        for fn in os.listdir(prod_dir)
        if os.path.isfile(os.path.join(prod_dir, fn))
    ) / (1024 * 1024)

    total_time = time.time() - t_start
    monitoring_baseline = {
        'version': VERSION,
        'model_name': model_name,
        'seed': seed,
        'f1_test': float(f1_test),
        'accuracy_test': float(acc_test),
        'mean_entropy': conf_stats['mean_entropy'],
        'mean_confidence': conf_stats['mean_confidence'],
        'ece': calib_stats['ece'],
        'brier': calib_stats['brier'],
        'subgroup_results': subgroup_results,
        'train_label_distribution': {str(c): float(np.mean(y_tr == i)) for i, c in enumerate(le.classes_)},
        'hyperparams': {
            'max_length': max_length, 'batch_size': batch_size,
            'grad_accum': grad_accum, 'lr': lr, 'epochs': epochs,
            'freeze_layers': freeze_layers,
        },
        'total_time_seconds': total_time,
        'model_size_mb': float(model_size_mb),
    }
    with open(os.path.join(prod_dir, 'monitoring_baseline.json'), 'w', encoding='utf-8') as f:
        json.dump(monitoring_baseline, f, ensure_ascii=False, indent=2)

    logger.info(f"📁 Đã lưu mô hình + baseline giám sát vào {prod_dir}  ({model_size_mb:.2f} MB)")
    logger.info(f"⏱️  Tổng thời gian: {total_time:.1f}s")

    # ── [FULL-MODEL-10-SEED PATCH] Lưu checkpoint đầy đủ (tokenizer + model +
    #    label_encoder_classes.json + monitoring_baseline.json) RIÊNG cho MỌI
    #    seed, không chỉ seed=42 như bản trước — prod_dir vẫn bị GHI ĐÈ mỗi
    #    seed (để không phá schema monitoring_baseline.json ở trên), nên nếu
    #    không sao lưu riêng, checkpoint của seed vừa chạy sẽ bị mất khi seed
    #    kế tiếp ghi đè lên prod_dir.
    #    Đánh đổi dung lượng: mỗi bản sao ~model_size_mb (phobert-base(-v2)
    #    fp32 thường ~500-540MB) x 10 seed ≈ 5-5.5GB tổng. Đây là lựa chọn có
    #    chủ đích của người dùng (cần đủ 10 checkpoint để dùng lại sau này —
    #    vd. ensemble, phân tích lỗi theo seed), không phải mặc định tiết
    #    kiệm đĩa của bản trước nữa. Nếu chạy trên máy đĩa giới hạn (Kaggle),
    #    xem --session/--seeds ở argparser để chia 10 seed thành nhiều phiên
    #    nhỏ hơn, tải model ra ngoài giữa các phiên.
    seed_dir = os.path.join('models', 'production', f'{VERSION}_seed{seed}')
    try:
        shutil.rmtree(seed_dir, ignore_errors=True)
        shutil.copytree(prod_dir, seed_dir)
        logger.info(f"🔒 [FULL-MODEL-10-SEED] Đã sao lưu riêng checkpoint đầy đủ seed={seed} vào "
                    f"{seed_dir} ({model_size_mb:.2f} MB, không bị các seed sau ghi đè).")
    except Exception as e:
        logger.warning(f"Không sao lưu được checkpoint riêng cho seed={seed}: {e}")

    # ── 9. Dọn checkpoint tạm của seed này để không cộng dồn dung lượng ─
    if cleanup_checkpoints:
        try:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            logger.info(f"🧹 Đã xoá checkpoint tạm của seed={seed} ({checkpoint_dir}) để tiết kiệm đĩa.")
        except Exception as e:
            logger.warning(f"Không xoá được thư mục checkpoint tạm: {e}")

    # Trả về dict đầy đủ — Y HỆT convention của train_v40() — để
    # vòng lặp đa seed ở __main__ tính mean/std cho bảng so sánh 2 hệ thống.
    return {
        # [ALIGN-2026-08] [F] Xem giải thích ở train_v40.py: 'f1_comparable' là khoá dùng cho
        # bảng so sánh 2 hệ thống. PhoBERT đánh giá bằng argmax thuần nên bằng chính 'f1'.
        'f1_comparable': float(f1_test),
        'f1': float(f1_test),
        'accuracy': float(acc_test),
        'ece': calib_stats['ece'],
        'brier': calib_stats['brier'],
        'training_time_sec': float(total_time),
        'model_size_mb': float(model_size_mb),
        'mean_confidence': conf_stats['mean_confidence'],
        'mean_entropy': conf_stats['mean_entropy'],
        'subgroup_results': subgroup_results,
        'hyperparams': monitoring_baseline['hyperparams'],
    }


# ═══════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════
def build_argparser():
    p = argparse.ArgumentParser(description="Fine-tune PhoBERT baseline cho AERS-Pro (so sánh với V40)")
    p.add_argument('--input_dir', type=str, default='input_data',
                   help="Thư mục chứa CÙNG file dữ liệu đã dùng cho train_v40.py")
    p.add_argument('--model_name', type=str, default='vinai/phobert-base-v2',
                   help="vinai/phobert-base-v2 (mặc định, để so sánh công bằng số layer/kích thước với "
                        "báo cáo hiện tại). Với T4 x2 (16GB/GPU) vinai/phobert-large cũng khả thi nếu muốn thử.")
    p.add_argument('--max_length', type=int, default=192,
                   help="192 mặc định — ĐO THỰC TẾ trên dataset_cleaned.csv (25289 dòng): p50=17 từ, "
                        "p95=65, p97=76, p99=127 từ (word-level, TRƯỚC khi tách từ/BPE làm đổi số token). "
                        "max_length=128 sẽ cắt cụt một phần đáng kể ngay ở p99, nên nâng lên 192 để có biên an "
                        "toàn qua cả bước word-segment + BPE của PhoBERT, mà vẫn rẻ hơn nhiều so với VRAM dư "
                        "của T4 x2 16GB/GPU. Giảm nếu OOM (xem log 'VRAM' in ra khi chạy để canh).")
    p.add_argument('--batch_size', type=int, default=32,
                   help="Batch size TRÊN MỖI GPU (mặc định 32, canh cho T4 x2 16GB). Trainer tự nhân đôi cho "
                        "2 GPU qua DataParallel. Giảm nếu OOM hoặc chỉ có 1 GPU.")
    p.add_argument('--grad_accum', type=int, default=1,
                   help="Mặc định 1 — với batch_size=32 x 2 GPU đã có effective batch=64, không cần cộng dồn thêm.")
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--fp16', action='store_true', default=True,
                   help="Bật mixed precision — T4 CÓ Tensor Core nên vừa tiết kiệm VRAM vừa tăng tốc thật")
    p.add_argument('--no-fp16', dest='fp16', action='store_false')
    p.add_argument('--gradient_checkpointing', action='store_true', default=False,
                   help="Mặc định TẮT (đủ VRAM trên T4 x2 16GB, tắt để tối đa tốc độ). Bật lại nếu OOM.")
    p.add_argument('--no-gradient_checkpointing', dest='gradient_checkpointing', action='store_false')
    p.add_argument('--freeze_layers', type=int, default=0,
                   help="Số layer encoder đầu bị đóng băng. Thường không cần trên T4 x2, chỉ dùng nếu OOM.")
    p.add_argument('--keep_checkpoints', dest='cleanup_checkpoints', action='store_false', default=True,
                   help="Giữ lại checkpoint tạm của từng seed thay vì xoá sau khi xong (để debug)")
    p.add_argument('--no_resume', action='store_true',
                   help="Bỏ qua results_phobert_multiseed.json cũ, chạy lại từ seed đầu tiên")
    p.add_argument('--resume_from', type=str, default=None,
                   help="Đường dẫn tới results_phobert_multiseed.json của MỘT PHIÊN KAGGLE TRƯỚC (vd. upload "
                        "làm Kaggle Input Dataset rồi trỏ /kaggle/input/.../results_phobert_multiseed.json) — "
                        "dùng khi phiên cũ đã hết giờ và /kaggle/working bị reset, để tiếp tục các seed còn thiếu.")
    p.add_argument('--group_col', type=str, default='near_dup_group_id',
                   help="Tên cột group-aware split (Trục 1, mục 2.2) — PHẢI khớp với GROUP_COL dùng trong "
                        "train_v40.py để 2 hệ thống so sánh trên cùng một split sạch. "
                        "Cột này do dedup_clustering.py sinh ra — chạy script đó trước. "
                        "Truyền --group_col \"\" để tắt (quay về stratified split cũ, KHÔNG khuyến khích).")
    p.add_argument('--quick_test', action='store_true',
                   help="Chạy kiểm tra nhanh (1 epoch, batch nhỏ, seed=42 duy nhất) đúng README mục 5, "
                        "KHÔNG chạy vòng lặp 10 seed — dùng để xác nhận không OOM/lỗi config trước khi chạy full")
    p.add_argument('--session', type=int, default=None, choices=[1, 2, 3, 4, 5],
                   help="[FULL-MODEL-10-SEED] Chỉ chạy 2 seed của PHIÊN này trong tổng số 5 phiên "
                        "(10 seed / 2 seed mỗi phiên): phiên 1 -> seed[0:2], phiên 2 -> seed[2:4], ..., "
                        "phiên 5 -> seed[8:10] (thứ tự lấy từ SEEDS trong seeds_config.py, mặc định "
                        "42-51 -> phiên 1={42,43}, 2={44,45}, 3={46,47}, 4={48,49}, 5={50,51}). "
                        "Dùng để chia 10 seed thành nhiều lần chạy ngắn hơn (vd. giới hạn thời gian/đĩa "
                        "của Kaggle) — mỗi phiên chỉ train+lưu full model cho đúng 2 seed. Bỏ qua cờ này "
                        "để giữ hành vi cũ: chạy hết mọi seed còn thiếu trong SEEDS. Không dùng cùng lúc "
                        "với --seeds.")
    p.add_argument('--seeds', type=str, default=None,
                   help="[FULL-MODEL-10-SEED] Danh sách seed tường minh, phân cách bởi dấu phẩy (vd. "
                        "\"42,43\") — thay thế cho --session khi muốn tự chọn seed cho một phiên thay vì "
                        "chia đôi mặc định. Mọi seed truyền vào PHẢI nằm trong SEEDS của seeds_config.py. "
                        "Ưu tiên cao hơn --session nếu truyền cả hai.")
    return p


def resolve_session_seeds(args, SEEDS):
    """[FULL-MODEL-10-SEED] Trả về danh sách seed CẦN CHẠY trong lần gọi này.

    - --seeds "42,43" (ưu tiên cao nhất): dùng đúng danh sách truyền vào, sau
      khi kiểm tra mọi giá trị đều nằm trong SEEDS (seeds_config.py) — không
      cho phép seed lạ lọt vào để tránh phá vỡ điều kiện "cùng seed" giữa ba
      hệ thống (xem seeds_config.py).
    - --session N (N=1..5): chia SEEDS thành 5 phiên liên tiếp, mỗi phiên
      đúng 2 seed, theo THỨ TỰ trong SEEDS (không phụ thuộc SEEDS có đúng
      bằng range(42,52) hay không — nếu N_SEEDS đổi, số phiên hợp lệ cũng
      phải đổi theo, assert bên dưới sẽ nhắc).
    - Không truyền gì: giữ nguyên hành vi cũ — trả về toàn bộ SEEDS.
    """
    if args.seeds:
        try:
            requested = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
        except ValueError:
            raise SystemExit(f"--seeds không hợp lệ: '{args.seeds}' — cần dạng \"42,43\"")
        unknown = [s for s in requested if s not in SEEDS]
        if unknown:
            raise SystemExit(
                f"--seeds chứa giá trị không nằm trong SEEDS của seeds_config.py: {unknown}. "
                f"SEEDS hợp lệ hiện tại: {SEEDS}. Không tự thêm seed lạ — sẽ phá điều kiện "
                f"'cùng seed' bắt buộc giữa V40/PhoBERT.")
        return requested

    if args.session is not None:
        n_sessions = 5
        per_session = 2
        assert len(SEEDS) == n_sessions * per_session, (
            f"--session giả định {n_sessions} phiên x {per_session} seed = {n_sessions * per_session} "
            f"seed, nhưng SEEDS hiện có {len(SEEDS)} phần tử. Dùng --seeds \"...\" để tự chia thủ công "
            f"thay vì --session.")
        start = (args.session - 1) * per_session
        return SEEDS[start:start + per_session]

    return list(SEEDS)


def run_multiseed(args):
    from seeds_config import SEEDS

    target_seeds = resolve_session_seeds(args, SEEDS)
    if args.session is not None or args.seeds:
        logger.info(f"🗂️  [FULL-MODEL-10-SEED] Chạy theo phiên: chỉ seed {target_seeds} trong lần gọi này "
                    f"(toàn bộ giao thức vẫn là {len(SEEDS)} seed: {SEEDS}). Chạy lại script với "
                    f"--session/--seeds khác (và --resume_from trỏ vào results_phobert_multiseed.json "
                    f"của phiên này) để hoàn tất các seed còn lại.")

    result_path = 'results_phobert_multiseed.json'
    results = {}
    if not args.no_resume and os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            results = {int(k): v for k, v in prev.get('metrics_by_seed', {}).items()}
            if results:
                logger.info(f"↩️  Tìm thấy {result_path} (cùng phiên) — resume, đã có {len(results)}/{len(SEEDS)} "
                             f"seed hoàn thành: {sorted(results.keys())}. Dùng --no_resume để chạy lại từ đầu.")
        except Exception as e:
            logger.warning(f"Không đọc được {result_path} cũ ({e}) — chạy lại từ đầu.")
            results = {}

    # Resume từ PHIÊN KAGGLE TRƯỚC (session mới, /kaggle/working đã bị reset) —
    # chỉ dùng khi chưa có kết quả nào trong phiên hiện tại, để không vô tình
    # ghi đè tiến độ đã có của phiên đang chạy.
    if not results and not args.no_resume and args.resume_from:
        try:
            with open(args.resume_from, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            results = {int(k): v for k, v in prev.get('metrics_by_seed', {}).items()}
            logger.info(f"↩️  Nạp kết quả từ phiên Kaggle trước ({args.resume_from}) — đã có "
                        f"{len(results)}/{len(SEEDS)} seed hoàn thành: {sorted(results.keys())}.")
        except Exception as e:
            logger.warning(f"Không đọc được --resume_from='{args.resume_from}' ({e}) — chạy lại từ đầu.")
            results = {}

    remaining = [s for s in target_seeds if s not in results]
    if not remaining:
        logger.info(f"✅ Seed của lần gọi này ({target_seeds}) đã hoàn thành từ trước — không còn gì để chạy. "
                     "Xoá results_phobert_multiseed.json, dùng --no_resume, hoặc đổi --session/--seeds "
                     "nếu muốn chạy seed khác.")
    else:
        logger.info("#" * 60)
        logger.info(f"CHẠY PHOBERT — {len(target_seeds)} seed của lần gọi này: {target_seeds}  "
                    f"(còn lại {len(remaining)}: {remaining})  |  toàn bộ giao thức: {len(SEEDS)} seed {SEEDS}")
        logger.info("#" * 60)

    METRIC_KEYS = ['f1_comparable', 'f1', 'accuracy', 'ece', 'brier', 'training_time_sec', 'model_size_mb',
                   'mean_confidence', 'mean_entropy']
    summary = {}
    seed_durations = []
    for i, s in enumerate(remaining, 1):
        logger.info(f"\n########## [PHOBERT] SEED {i}/{len(remaining)} còn lại (seed={s}) ##########")
        if seed_durations:
            eta_sec = np.mean(seed_durations) * (len(remaining) - i + 1)
            logger.info(f"⏳ ETA còn lại ước tính: {eta_sec / 3600:.1f} giờ "
                        f"(dựa trên trung bình {np.mean(seed_durations):.0f}s/seed đã chạy)")

        metrics = train_phobert(
            input_dir=args.input_dir, model_name=args.model_name, max_length=args.max_length,
            batch_size=args.batch_size, grad_accum=args.grad_accum, epochs=args.epochs, lr=args.lr,
            seed=s, fp16=args.fp16, gradient_checkpointing=args.gradient_checkpointing,
            freeze_layers=args.freeze_layers, cleanup_checkpoints=args.cleanup_checkpoints,
            group_col=(args.group_col or None),
        )
        results[s] = metrics
        seed_durations.append(metrics['training_time_sec'])

        # Lưu ngay sau MỖI seed — chống mất kết quả nếu máy bị tắt/ngủ giữa chừng
        # (quan trọng hơn cả với V40 vì đây là laptop, không phải máy thuê
        # luôn bật — có thể phải chia làm nhiều phiên chạy trong nhiều ngày).
        summary = {}
        for key in METRIC_KEYS:
            vals = [results[s2][key] for s2 in results if key in results[s2]]
            if vals:
                summary[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'phobert',
                'model_name': args.model_name,
                'seeds': SEEDS,
                'group_col': (args.group_col or None),
                'metrics_by_seed': {str(k): v for k, v in results.items()},
                'summary': summary,
                'completed_seeds': len(results),
                'total_seeds': len(SEEDS),
            }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ PHOBERT — qua {len(results)}/{len(SEEDS)} seed:")
    for key in METRIC_KEYS:
        if key in summary:
            logger.info(f"   {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    logger.info(f"   Chi tiết từng seed: {sorted(results.keys())}")
    logger.info(f"📁 Đã lưu {result_path}")
    logger.info("=" * 60)


def run_quick_test(args):
    """Đúng README mục 5: 1 epoch, batch nhỏ, seed=42 duy nhất, chỉ để xác
    nhận pipeline chạy hết mà không OOM/lỗi config trước khi chạy full 10 seed."""
    logger.info("🔎 QUICK TEST — 1 epoch, batch_size=2, grad_accum=4, seed=42 (không lưu vào "
                "results_phobert_multiseed.json). Theo dõi log để chắc chắn không OOM trước khi bỏ --quick_test.")
    train_phobert(
        input_dir=args.input_dir, model_name=args.model_name, max_length=args.max_length,
        batch_size=2, grad_accum=4, epochs=1, lr=args.lr,
        seed=42, fp16=args.fp16, gradient_checkpointing=args.gradient_checkpointing,
        freeze_layers=args.freeze_layers, cleanup_checkpoints=args.cleanup_checkpoints,
        group_col=(args.group_col or None),
    )


# ═══════════════════════════════════════════════════════════════════
# 6. CHẠY CHÍNH — ĐA SEED (10 seed, dùng CHUNG với V40 — seeds_config.py)
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    args = build_argparser().parse_args()
    if args.quick_test:
        run_quick_test(args)
    else:
        run_multiseed(args)
