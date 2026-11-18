#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# GỘP KẾT QUẢ 2 HỆ THỐNG → BẢNG SO SÁNH + KIỂM ĐỊNH THEO CẶP
# ───────────────────────────────────────────────────────────────────────────
# Đọc results_v40_multiseed.json và results_phobert_multiseed.json rồi xuất:
#   • bảng mean ± std (và CI 95% qua t-phân phối) cho từng chỉ số, từng hệ thống;
#   • kiểm định THEO CẶP giữa từng đôi hệ thống trên cùng seed (Wilcoxon signed-
#     rank + paired t-test), có hiệu chỉnh Holm-Bonferroni cho họ so sánh;
#   • bảng Markdown dán thẳng vào bài + comparison_table.json để truy vết.
#
# BA LỚP KIỂM TRA TÍNH HỢP LỆ TRƯỚC KHI TÍNH BẤT CỨ SỐ NÀO — script TỪ CHỐI
# dựng bảng nếu không đạt, vì một bảng đẹp từ dữ liệu không ghép cặp được còn
# tệ hơn không có bảng:
#   1. Hai file phải khai báo CÙNG danh sách seed.
#   2. Hai file phải có đủ kết quả cho MỌI seed trong danh sách đó (không được
#      thiếu seed rồi so sánh trên phần giao — đó là chọn lọc sau khi thấy số).
#   3. Nếu có split_fingerprints.json: mọi seed phải cho cùng dấu vân tay split
#      ở cả hai hệ thống. Đây là điều kiện làm cho chữ "theo cặp" có nghĩa.
#
# CHỈ SỐ SO SÁNH CHÍNH: 'f1_comparable'
#   KHÔNG dùng 'f1'. Với V40, 'f1' là F1 SAU MacroF1DecisionLayer — một lớp tinh
#   chỉnh quyết định mà PhoBERT không có; đặt nó cạnh hai số argmax là so
#   sánh lệch. 'f1_comparable' = argmax ở cả hai hệ. Lợi ích riêng của lớp quyết
#   định vẫn được báo cáo, nhưng ở dòng riêng ('delta_decision_layer'), không
#   trộn vào cột so sánh.
#
# CÁCH DÙNG
#   python compare_systems.py
#   python compare_systems.py --dir /kaggle/working --out bang_ket_qua.md
#   python compare_systems.py --allow-partial     # CHỈ để xem tạm khi đang chạy
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

FILES = {
    'V40':     'results_v40_multiseed.json',
    'PhoBERT': 'results_phobert_multiseed.json',
}

PRIMARY = 'f1_comparable'

# Chỉ số nào "cao là tốt" — quyết định dấu của phần chênh và cách đọc kết luận.
HIGHER_BETTER = {
    'f1_comparable': True, 'f1': True, 'accuracy': True, 'f1_argmax': True,
    'ece': False, 'brier': False,
    'training_time_sec': False, 'model_size_mb': False,
    # [LEX-47] v40.4 — độ phủ lexicon. Cao là tốt, NHƯNG đây là chỉ số CHẨN
    # ĐOÁN, không phải chỉ số hiệu năng: đừng dùng nó để kết luận hệ nào hơn.
    # Đo được trên subset 7k: độ phủ 0,548 → 0,985 chỉ đổi F1 +0,0042, vì
    # TF‑IDF đã bắt phần lớn tín hiệu từ vựng. Giá trị của nó là (a) trả lời
    # được câu hỏi "khối thủ công có phải ma trận 0 không" và (b) phát hiện
    # lệch Train/Test của từ điển học theo fold.
    'lexicon_coverage_train': True, 'lexicon_coverage_test': True,
    'lexicon_coverage_before_expand': True, 'lexicon_size': True,
}
REPORT_METRICS = ['f1_comparable', 'accuracy', 'ece', 'brier',
                  'training_time_sec', 'model_size_mb']
# Chỉ số chẩn đoán: in ra để tham khảo, KHÔNG đưa vào paired test.
DIAGNOSTIC_METRICS = ['lexicon_coverage_train', 'lexicon_coverage_test',
                      'lexicon_size']


def load_all(d, allow_partial):
    data, seeds_decl, problems = {}, {}, []
    for name, fname in FILES.items():
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            problems.append(f"Thiếu {fname} — chưa chạy xong hệ thống {name}?")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        by_seed = {int(k): v for k, v in raw.get('metrics_by_seed', {}).items()}
        data[name] = by_seed
        seeds_decl[name] = [int(s) for s in raw.get('seeds', [])]
        print(f"✓ {fname}: {len(by_seed)}/{len(seeds_decl[name])} seed hoàn thành "
              f"(khai báo: {seeds_decl[name]})")

    if len(data) < 2:
        problems.append("Cần ít nhất 2 hệ thống để so sánh.")
        return data, [], problems

    # (1) cùng danh sách seed
    decls = {name: tuple(v) for name, v in seeds_decl.items() if v}
    if len(set(decls.values())) > 1:
        problems.append(
            "Hai file khai báo DANH SÁCH SEED KHÁC NHAU:\n" +
            "\n".join(f"      {n}: {list(v)}" for n, v in decls.items()) +
            "\n      → không ghép cặp được. Đảm bảo cả hai import cùng seeds_config.py "
            "rồi chạy lại TRỌN BỘ (không gộp kết quả từ hai danh sách seed khác nhau).")
        return data, [], problems

    seeds = sorted(next(iter(decls.values()))) if decls else []

    # (2) đủ mọi seed ở mọi hệ thống
    missing = {n: [s for s in seeds if s not in by_seed] for n, by_seed in data.items()}
    missing = {n: m for n, m in missing.items() if m}
    if missing:
        msg = ("Chưa đủ seed:\n" +
               "\n".join(f"      {n}: thiếu {m}" for n, m in missing.items()))
        if allow_partial:
            print(f"⚠️  {msg}\n   --allow-partial: so sánh trên phần giao. "
                  f"KHÔNG dùng số này cho bài báo — báo cáo trên tập seed không đầy đủ "
                  f"là một dạng chọn lọc sau khi đã thấy kết quả.")
            seeds = [s for s in seeds if all(s in by_seed for by_seed in data.values())]
        else:
            problems.append(msg + "\n      → dùng --allow-partial nếu chỉ muốn xem tạm.")
    return data, seeds, problems


def check_fingerprints(d, seeds, systems):
    path = os.path.join(d, 'split_fingerprints.json')
    if not os.path.exists(path):
        print("⚠️  Không có split_fingerprints.json — KHÔNG thể xác nhận hai hệ thống "
              "chia dữ liệu giống nhau. Chữ 'theo cặp' trong kết quả dưới đây là GIẢ ĐỊNH, "
              "chưa được kiểm chứng. Copy split_fingerprint.py vào cạnh 2 file train rồi "
              "chạy lại để có kiểm chứng thật.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        store = json.load(f)
    alias = {'V40': 'v40', 'PhoBERT': 'phobert'}
    bad = []
    for s in seeds:
        entry = store.get(str(s), {})
        hs = {alias[n]: entry.get(alias[n], {}).get('hash_test') for n in systems}
        present = {k: v for k, v in hs.items() if v}
        if len(present) < len(systems):
            bad.append(f"seed {s}: thiếu dấu vân tay của {sorted(set(hs) - set(present))}")
        elif len(set(present.values())) > 1:
            bad.append(f"seed {s}: hash_test KHÁC NHAU giữa các hệ — {present}")
    if bad:
        print("❌ DẤU VÂN TAY SPLIT KHÔNG KHỚP:")
        for b in bad:
            print(f"      {b}")
        return False
    print(f"✓ Dấu vân tay split khớp ở cả {len(seeds)}/{len(seeds)} seed cho "
          f"{len(systems)} hệ thống — paired test hợp lệ.")
    return True


def series(data, system, seeds, metric):
    out = []
    for s in seeds:
        v = data[system].get(s, {}).get(metric)
        out.append(np.nan if v is None else float(v))
    return np.array(out, dtype=float)


def holm(pvals):
    """Hiệu chỉnh Holm-Bonferroni. Trả p đã hiệu chỉnh, cùng thứ tự đầu vào."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # đảm bảo đơn điệu không giảm
        adj[i] = min(running, 1.0)
    return adj


def fmt(v, metric):
    if np.isnan(v):
        return '—'
    if metric in ('training_time_sec',):
        return f"{v:,.0f}s"
    if metric in ('model_size_mb',):
        return f"{v:,.1f}MB"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description="Gộp 2 results_*.json thành bảng so sánh + paired test")
    ap.add_argument('--dir', default='.', help="Thư mục chứa các file results_*.json")
    ap.add_argument('--out', default='comparison_table.md', help="File Markdown xuất ra")
    ap.add_argument('--allow-partial', action='store_true',
                    help="Cho phép so sánh khi chưa đủ seed (CHỈ để xem tạm, không dùng cho bài)")
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    print("═" * 75)
    print("GỘP KẾT QUẢ 2 HỆ THỐNG — AERS")
    print("═" * 75)
    data, seeds, problems = load_all(args.dir, args.allow_partial)
    if problems:
        print("\n❌ TỪ CHỐI DỰNG BẢNG:")
        for p in problems:
            print(f"   • {p}")
        return 1
    systems = list(data.keys())
    print(f"\nSố seed dùng để so sánh: {len(seeds)} → {seeds}")
    fp_ok = check_fingerprints(args.dir, seeds, systems)

    if not HAS_SCIPY:
        print("\n⚠️  Không có scipy — sẽ bỏ phần kiểm định, chỉ in mean ± std.")

    lines = []
    lines.append("# Bảng so sánh 2 hệ thống — V40 / PhoBERT\n")
    lines.append(f"- Số seed: **{len(seeds)}** (`{seeds[0]}`–`{seeds[-1]}`), dùng chung cho cả hai hệ thống.")
    lines.append(f"- Chỉ số so sánh chính: **`{PRIMARY}`** — F1-macro theo argmax ở cả hai hệ. "
                 f"Với V40, đây là `f1_argmax`, KHÔNG phải `f1` (vốn đã qua MacroF1DecisionLayer).")
    if fp_ok is True:
        lines.append("- Dấu vân tay split: **khớp ở mọi seed** — mỗi cặp điểm đến từ đúng cùng một "
                     "lần chia Train/Calibration/Test, nên kiểm định theo cặp hợp lệ.")
    elif fp_ok is False:
        lines.append("- ⚠️ Dấu vân tay split **KHÔNG khớp** — số dưới đây KHÔNG dùng được. Xem log.")
    else:
        lines.append("- ⚠️ Chưa kiểm chứng được dấu vân tay split (thiếu `split_fingerprints.json`).")
    lines.append("")

    # ── Bảng 1: mean ± std ────────────────────────────────────────────────
    lines.append("## Bảng 1 — Hiệu năng qua các seed (mean ± std, CI 95%)\n")
    header = "| Chỉ số | " + " | ".join(systems) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(systems) + 1))
    print(f"\n{'─' * 75}\nBẢNG 1 — mean ± std\n{'─' * 75}")
    for metric in REPORT_METRICS:
        cells = []
        printed = []
        for sysname in systems:
            v = series(data, sysname, seeds, metric)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                cells.append('—'); printed.append('—'); continue
            m, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            if len(v) > 1 and HAS_SCIPY:
                half = stats.t.ppf(1 - 0.025, len(v) - 1) * sd / np.sqrt(len(v))
                cells.append(f"{fmt(m, metric)} ± {fmt(sd, metric)}<br><sub>CI95 [{fmt(m - half, metric)}, {fmt(m + half, metric)}]</sub>")
            else:
                cells.append(f"{fmt(m, metric)} ± {fmt(sd, metric)}")
            printed.append(f"{fmt(m, metric)} ± {fmt(sd, metric)}")
        arrow = '↑' if HIGHER_BETTER.get(metric, True) else '↓'
        lines.append(f"| `{metric}` {arrow} | " + " | ".join(cells) + " |")
        print(f"  {metric:<20} {arrow}  " + "   ".join(f"{s}: {p}" for s, p in zip(systems, printed)))

    # V40: lợi ích của lớp quyết định, báo cáo RIÊNG
    if 'V40' in data:
        d40 = series(data, 'V40', seeds, 'delta_decision_layer')
        d40 = d40[~np.isnan(d40)]
        if len(d40):
            lines.append("")
            lines.append(f"> **Lớp quyết định của V40 (báo cáo riêng, KHÔNG trộn vào bảng trên):** "
                         f"`delta_decision_layer` = {np.mean(d40):+.4f} ± {np.std(d40, ddof=1):.4f} "
                         f"— phần F1 mà `MacroF1DecisionLayer` cộng thêm cho V40 so với argmax. "
                         f"PhoBERT không có lớp này.")
            print(f"\n  V40 delta_decision_layer: {np.mean(d40):+.4f} ± {np.std(d40, ddof=1):.4f} "
                  f"(báo cáo riêng, không trộn vào cột so sánh)")

    # ── Bảng 2: kiểm định theo cặp ────────────────────────────────────────
    if HAS_SCIPY and len(seeds) >= 5:
        lines.append("\n## Bảng 2 — Kiểm định theo cặp trên cùng seed\n")
        lines.append(f"Họ so sánh: mọi đôi hệ thống × chỉ số `{PRIMARY}`, hiệu chỉnh "
                     f"**Holm-Bonferroni** ở mức α = {args.alpha}. "
                     f"Wilcoxon signed-rank là kiểm định chính (không giả định phân phối chuẩn, "
                     f"phù hợp với n = {len(seeds)}); paired t-test in kèm để đối chiếu.\n")
        lines.append("| So sánh | Δ trung bình | Wilcoxon p | t-test p | p (Holm) | Kết luận |")
        lines.append("|---|---|---|---|---|---|")

        pairs, rows = [], []
        for a, b in combinations(systems, 2):
            va, vb = series(data, a, seeds, PRIMARY), series(data, b, seeds, PRIMARY)
            ok = ~(np.isnan(va) | np.isnan(vb))
            va, vb = va[ok], vb[ok]
            diff = va - vb
            if len(diff) < 5 or np.allclose(diff, 0):
                rows.append((f"{a} vs {b}", float(np.mean(diff)) if len(diff) else np.nan,
                             np.nan, np.nan))
                pairs.append(1.0)
                continue
            try:
                w_p = float(stats.wilcoxon(va, vb).pvalue)
            except Exception:
                w_p = np.nan
            t_p = float(stats.ttest_rel(va, vb).pvalue)
            rows.append((f"{a} vs {b}", float(np.mean(diff)), w_p, t_p))
            pairs.append(w_p if not np.isnan(w_p) else t_p)

        adj = holm(np.array(pairs, dtype=float))
        print(f"\n{'─' * 75}\nBẢNG 2 — paired test trên '{PRIMARY}' (Holm-Bonferroni)\n{'─' * 75}")
        for (label, dmean, w_p, t_p), p_adj in zip(rows, adj):
            sig = p_adj < args.alpha
            if np.isnan(dmean):
                verdict = "không đủ dữ liệu"
            elif not sig:
                verdict = f"**không có khác biệt có ý nghĩa** (p_Holm = {p_adj:.4f})"
            else:
                better = label.split(' vs ')[0 if dmean > 0 else 1]
                verdict = f"**{better} cao hơn**, có ý nghĩa (p_Holm = {p_adj:.4f})"
            lines.append(f"| {label} | {dmean:+.4f} | "
                         f"{'—' if np.isnan(w_p) else f'{w_p:.4f}'} | "
                         f"{'—' if np.isnan(t_p) else f'{t_p:.4f}'} | {p_adj:.4f} | {verdict} |")
            print(f"  {label:<20} Δ={dmean:+.4f}  wilcoxon p={w_p:.4f}  t p={t_p:.4f}  "
                  f"Holm p={p_adj:.4f}  {'SIG' if sig else 'ns'}")

        lines.append("")
        lines.append("> **Cách đọc:** \"không có khác biệt có ý nghĩa\" KHÔNG đồng nghĩa với "
                     "\"hai hệ tương đương\" — với n = %d seed, kiểm định chỉ phát hiện được "
                     "khác biệt tương đối lớn. Báo cáo cả Δ và CI ở Bảng 1 để người đọc tự "
                     "đánh giá độ lớn hiệu ứng, đừng chỉ dựa vào p." % len(seeds))
    else:
        lines.append("\n> Bỏ qua Bảng 2: cần scipy và ≥ 5 seed.")

    md = "\n".join(lines) + "\n"
    out_md = os.path.join(args.dir, args.out)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)

    dump = {
        'seeds': seeds,
        'primary_metric': PRIMARY,
        'fingerprints_verified': fp_ok,
        'per_seed': {sysname: {str(s): data[sysname].get(s, {}).get(PRIMARY) for s in seeds}
                     for sysname in systems},
        'summary': {sysname: {m: {'mean': float(np.nanmean(series(data, sysname, seeds, m))),
                                  'std': float(np.nanstd(series(data, sysname, seeds, m), ddof=1))}
                              for m in REPORT_METRICS
                              if not np.all(np.isnan(series(data, sysname, seeds, m)))}
                    for sysname in systems},
    }
    out_json = os.path.join(args.dir, 'comparison_table.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Đã ghi {out_md} (dán thẳng vào bài) và {out_json} (truy vết số liệu)")
    if fp_ok is False:
        print("❌ NHƯNG dấu vân tay split không khớp — KHÔNG dùng bảng này cho bài báo.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
