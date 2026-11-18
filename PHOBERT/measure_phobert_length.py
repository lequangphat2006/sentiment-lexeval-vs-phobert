#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# ĐO ĐỘ DÀI THẬT ĐỂ CHỌN max_length CHO PhoBERT
# ───────────────────────────────────────────────────────────────────────────
# VÌ SAO
#   train_phobert.py đặt max_length=192, biện luận trong file dựa trên phân bố
#   độ dài của bộ dữ liệu 25.289 dòng CŨ (p50=17, p95=65, p99=127 từ). Bộ chính
#   thức hiện tại khác hẳn — đo lại ở mức TỪ:
#
#       p50=28   p90=43   p95=47   p97=50   p99=56   p99.9=98   max=353
#
#   Trung vị dài hơn (17→28) nhưng đuôi NGẮN hơn nhiều (p99: 127→56). Nghĩa là
#   192 giờ dư khá nhiều. Vì Dataset dùng padding='max_length' (padding CỐ ĐỊNH,
#   không phải dynamic padding theo batch), chi phí mỗi batch tỉ lệ TRỰC TIẾP với
#   con số này: giảm 192→128 tiết kiệm khoảng 1/3 thời gian train, nhân cho 10 seed.
#
#   NHƯNG số cần dùng không phải số TỪ mà là số token BPE SAU khi tách từ bằng
#   underthesea — hai bước đó làm số token phồng lên khó đoán. Script này đo
#   đúng con số đó, trên đúng tokenizer sẽ dùng để train.
#
# CHẠY Ở ĐÂU
#   Trên Kaggle/máy có transformers + underthesea (môi trường sẽ dùng để train).
#   Script tự tải tokenizer PhoBERT (chỉ tokenizer, không tải model — nhẹ).
#
#   python measure_phobert_length.py
#   python measure_phobert_length.py --sample 20000     # đo nhanh trên mẫu
#
# CÁCH CHỌN max_length TỪ KẾT QUẢ
#   Lấy phân vị 99.5 của số token BPE (đã gồm 2 token đặc biệt <s> </s>), làm
#   tròn LÊN bội số của 8 (thân thiện với Tensor Core của T4). Phần đuôi trên
#   ngưỡng đó bị cắt — script in ra đúng bao nhiêu câu bị ảnh hưởng để bạn quyết
#   định có chấp nhận hay không, thay vì đoán.
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

_WS = re.compile(r'\s+')


def normalize_whitespace(t: str) -> str:
    """Giống _normalize_whitespace() trong train_phobert.py: gộp mọi whitespace
    (\\n, \\r, \\t, nbsp) thành một dấu cách TRƯỚC khi tách từ, tránh dính chữ ở
    các review nhiều-trường ghép bằng '\\n'."""
    return _WS.sub(' ', str(t).replace('\u00a0', ' ')).strip()


def main():
    ap = argparse.ArgumentParser(description="Đo phân vị token BPE của PhoBERT để chọn max_length")
    ap.add_argument('--input', default=os.path.join('input_data', 'train.csv'))
    ap.add_argument('--model_name', default='vinai/phobert-base-v2')
    ap.add_argument('--sample', type=int, default=0, help="0 = dùng toàn bộ; >0 = lấy mẫu ngẫu nhiên")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    texts = df['comment'].astype(str).tolist()
    if args.sample and args.sample < len(texts):
        rng = np.random.RandomState(args.seed)
        texts = [texts[i] for i in rng.choice(len(texts), args.sample, replace=False)]
    print(f"📄 {args.input}: đo trên {len(texts)} câu")

    # ── 1. Mức TỪ (không cần thư viện ngoài) ───────────────────────────────
    w = np.array([len(normalize_whitespace(t).split()) for t in texts])
    qs = [50, 90, 95, 97, 99, 99.5, 99.9]
    print("\n① Số TỪ (whitespace, sau chuẩn hoá whitespace):")
    print("   " + "  ".join(f"p{q}={int(np.percentile(w, q))}" for q in qs) + f"  max={int(w.max())}")

    # ── 2. Sau tách từ underthesea ─────────────────────────────────────────
    try:
        from underthesea import word_tokenize
        seg = [' '.join(word_tokenize(normalize_whitespace(t), format='text').split())
               for t in texts]
        s = np.array([len(x.split()) for x in seg])
        print("\n② Số TOKEN sau underthesea.word_tokenize (nối cụm bằng '_'):")
        print("   " + "  ".join(f"p{q}={int(np.percentile(s, q))}" for q in qs) + f"  max={int(s.max())}")
    except ImportError:
        seg = [normalize_whitespace(t) for t in texts]
        print("\n② ⚠️  Không có underthesea — BỎ QUA bước tách từ. Số BPE dưới đây sẽ "
              "KHÔNG khớp lúc train (PhoBERT được train trên văn bản đã tách từ). "
              "Cài underthesea rồi đo lại trước khi quyết định max_length.")

    # ── 3. Số token BPE thật ───────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("\n③ ❌ Không có transformers — không đo được số BPE. "
              "Chạy script này trong môi trường sẽ dùng để train.")
        return 1
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    lens = np.array([len(tok(x, add_special_tokens=True)['input_ids']) for x in seg])
    print(f"\n③ Số TOKEN BPE của {args.model_name} (đã gồm <s> </s>):")
    print("   " + "  ".join(f"p{q}={int(np.percentile(lens, q))}" for q in qs) + f"  max={int(lens.max())}")

    # ── 4. Khuyến nghị ─────────────────────────────────────────────────────
    p995 = int(np.percentile(lens, 99.5))
    rec = int(np.ceil(p995 / 8) * 8)
    print(f"\n{'─' * 66}")
    print(f"KHUYẾN NGHỊ: --max_length {rec}   (p99.5 = {p995}, làm tròn lên bội số của 8)")
    print(f"{'─' * 66}")
    print(f"   Với max_length={rec}: {int((lens > rec).sum())} / {len(lens)} câu bị cắt "
          f"({(lens > rec).mean():.3%})")
    for cand in sorted({64, 96, 128, 160, 192, 256} | {rec}):
        n_trunc = int((lens > cand).sum())
        rel = cand / 192.0
        print(f"   max_length={cand:<4} → cắt {n_trunc:>6} câu ({n_trunc / len(lens):>7.3%})  "
              f"| chi phí tương đối so với 192: {rel:.2f}x")
    print("\n   Lưu ý: cắt cụt KHÔNG phải luôn có hại — phần đuôi của review dài thường là "
          "lặp lại thông tin đã có ở đầu. Nhưng nếu chọn ngưỡng cắt >1% số câu thì phải nêu "
          "trong bài, vì đó là một quyết định tiền xử lý ảnh hưởng tới kết quả.")
    print("   Sau khi đổi max_length, PHẢI chạy lại toàn bộ 10 seed của PhoBERT — không "
          "gộp số liệu của hai max_length khác nhau vào cùng một bảng.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
