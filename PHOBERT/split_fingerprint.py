# ═══════════════════════════════════════════════════════════════════════════
# DẤU VÂN TAY CỦA CÁCH CHIA DỮ LIỆU — CHỐNG LỆCH SPLIT GIỮA V40 / PHOBERT
# ───────────────────────────────────────────────────────────────────────────
# VÌ SAO CẦN FILE NÀY
#   Hai hệ thống được so sánh theo cặp (paired test) trên cùng 10 seed. Phép so
#   sánh đó CHỈ có nghĩa nếu với cùng một seed, cả hai sinh ra ĐÚNG cùng một bộ
#   Train/Calibration/Test. Trên thực tế đã từng lệch hai lần mà không ai phát
#   hiện ngay:
#     • lần 1: PhoBERT còn dùng GroupShuffleSplit sau khi V40 đã chuyển sang
#       StratifiedGroupKFold  → 0/10 seed khớp nhau;
#     • lần 2: train_v40.py để GROUP_SPLIT_STRATEGY='shuffle' trong khi V40 và
#       PhoBERT dùng 'stratified' → cũng 0/10 seed khớp nhau, giao nhau giữa hai
#       tập Test chỉ đúng 20% (bằng mức ngẫu nhiên thuần, vì |Test| = 20% dữ liệu).
#   Cả hai lần, mỗi lần chạy riêng lẻ đều "thành công": log không có lỗi, F1 in
#   ra bình thường, results_*.json ghi đủ. Lệch chỉ lộ ra khi đã ngồi viết bảng
#   so sánh — tức sau hàng chục giờ GPU.
#
#   Module này biến lỗi ÂM THẦM đó thành lỗi ỒN ÀO: hệ thống nào chạy trước sẽ
#   ghi lại dấu vân tay (hash của chỉ số từng tập) cho mỗi seed; hệ thống chạy
#   sau đối chiếu — khác là DỪNG ngay ở giây thứ 0 của seed đó, không đốt thêm
#   compute cho một con số về sau không dùng được.
#
# CÁCH DÙNG (đã được patch_align_baselines.py chèn tự động vào cả 2 file train)
#   from split_fingerprint import check_split_fingerprint
#   check_split_fingerprint('v40', seed, idx_tr, idx_cal, idx_te, len(df))
#
# NẾU CỐ Ý MUỐN SPLIT KHÁC NHAU (vd. chạy thử một chiến lược chia mới)
#   AERS_FINGERPRINT_MODE=warn   → chỉ cảnh báo, không raise
#   AERS_FINGERPRINT_MODE=off    → tắt hoàn toàn
#   hoặc xoá file split_fingerprints.json để bắt đầu lại từ đầu.
# ═══════════════════════════════════════════════════════════════════════════

import hashlib
import json
import logging
import os

FINGERPRINT_PATH = os.environ.get('AERS_FINGERPRINT_PATH', 'split_fingerprints.json')
MODE = os.environ.get('AERS_FINGERPRINT_MODE', 'strict').strip().lower()

logger = logging.getLogger(__name__)


def _hash_indices(idx) -> str:
    """Hash ổn định của MỘT TẬP chỉ số, không phụ thuộc thứ tự.

    Sắp xếp trước khi hash: StratifiedGroupKFold và GroupShuffleSplit trả chỉ
    số theo thứ tự khác nhau ngay cả khi tập hợp giống nhau, mà điều ta cần so
    là TẬP HỢP dòng nào thuộc tập nào — không phải thứ tự liệt kê.
    """
    payload = ','.join(str(int(i)) for i in sorted(int(x) for x in idx))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def compute_fingerprint(idx_tr, idx_cal, idx_te, n_rows):
    return {
        'n_rows': int(n_rows),
        'n_train': int(len(idx_tr)),
        'n_cal': int(len(idx_cal)),
        'n_test': int(len(idx_te)),
        'hash_train': _hash_indices(idx_tr),
        'hash_cal': _hash_indices(idx_cal),
        'hash_test': _hash_indices(idx_te),
    }


def _describe_mismatch(new, old, other_system):
    lines = [f"    • hệ thống đã ghi trước: '{other_system}'"]
    for key, label in (('n_rows', 'số dòng dữ liệu'), ('n_train', '|Train|'),
                       ('n_cal', '|Cal|'), ('n_test', '|Test|'),
                       ('hash_train', 'hash Train'), ('hash_cal', 'hash Cal'),
                       ('hash_test', 'hash Test')):
        a, b = new.get(key), old.get(key)
        flag = '  ← LỆCH' if a != b else ''
        lines.append(f"    • {label:<16}: lần này = {a}   |   đã ghi = {b}{flag}")
    return '\n'.join(lines)


def check_split_fingerprint(system, seed, idx_tr, idx_cal, idx_te, n_rows,
                            path=None, mode=None):
    """Ghi (lần đầu) hoặc đối chiếu (các lần sau) dấu vân tay split của 1 seed.

    system : 'v40' | 'phobert' — tên hệ thống đang chạy.
    seed   : seed của lần chạy này.
    Trả về dict dấu vân tay vừa tính, để hàm gọi có thể ghi kèm vào
    results_*.json nếu muốn (không bắt buộc).

    Raise RuntimeError khi split lệch so với hệ thống đã chạy trước ở CÙNG seed
    (trừ khi MODE='warn'/'off').
    """
    path = path or FINGERPRINT_PATH
    mode = (mode or MODE).strip().lower()
    fp = compute_fingerprint(idx_tr, idx_cal, idx_te, n_rows)

    if mode == 'off':
        return fp

    store = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                store = json.load(f)
        except Exception as e:
            logger.warning(f"[SPLIT-FP] Không đọc được {path} ({e}) — tạo lại từ đầu.")
            store = {}

    key = str(seed)
    entry = store.setdefault(key, {})

    # Đối chiếu với MỌI hệ thống đã ghi cho seed này.
    for other_system, old in list(entry.items()):
        if other_system == system:
            continue
        if old != fp:
            msg = (
                f"[SPLIT-FP] ❌ SPLIT LỆCH ở seed={seed}: hệ thống '{system}' sinh ra cách chia "
                f"KHÁC với '{other_system}' đã chạy trước.\n"
                f"{_describe_mismatch(fp, old, other_system)}\n"
                f"  Hệ quả: cặp (\"{system}\" seed {seed}, \"{other_system}\" seed {seed}) KHÔNG phải "
                f"một cặp — mọi paired t-test / Wilcoxon giữa hai hệ thống sẽ vô nghĩa, và câu "
                f"\"exact same train/calibration/test partition\" trong bài báo sẽ KHÔNG đúng.\n"
                f"  Cách khắc phục: đảm bảo cả hai file train dùng CÙNG một hàm chia — "
                f"StratifiedGroupKFold(n_splits=5) rồi (n_splits=4), cùng random_state=seed, cùng "
                f"group_col='near_dup_group_id', và cùng bộ dòng dữ liệu sau khi loại dòng rỗng/NaN "
                f"(xem GROUP_SPLIT_STRATEGY trong train_v40.py và load_data() ở cả hai file).\n"
                f"  Nếu ĐANG CỐ Ý thử một chiến lược chia khác: chạy với AERS_FINGERPRINT_MODE=warn, "
                f"hoặc xoá {path} — nhưng KHÔNG dùng số liệu lần chạy đó cho bảng so sánh chính."
            )
            if mode == 'warn':
                logger.warning(msg)
            else:
                raise RuntimeError(msg)

    prev_self = entry.get(system)
    if prev_self is not None and prev_self != fp:
        logger.warning(
            f"[SPLIT-FP] ⚠️  seed={seed}: chính hệ thống '{system}' cho ra split KHÁC lần chạy trước "
            f"(cùng seed đáng lẽ phải tái lập được).\n{_describe_mismatch(fp, prev_self, system)}\n"
            f"  Nguyên nhân thường gặp: dữ liệu trong input_data/ đã thay đổi, hoặc số dòng sau khi "
            f"loại dòng rỗng/NaN đã khác. Kiểm tra lại trước khi dùng số liệu."
        )

    entry[system] = fp
    store[key] = entry
    try:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(store, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"[SPLIT-FP] Không ghi được {path} ({e}) — bỏ qua, không chặn lần chạy.")

    logger.info(
        f"[SPLIT-FP] ✓ seed={seed} '{system}': Train/Cal/Test = "
        f"{fp['n_train']}/{fp['n_cal']}/{fp['n_test']}  hash_test={fp['hash_test']}  "
        f"(đã đối chiếu {max(len(entry) - 1, 0)} hệ thống khác cùng seed)"
    )
    return fp


def summarize(path=None):
    """In bảng dấu vân tay đã ghi — dùng để kiểm tra nhanh trước khi viết bảng kết quả."""
    path = path or FINGERPRINT_PATH
    if not os.path.exists(path):
        print(f"Chưa có {path} — chưa có lần chạy nào ghi dấu vân tay.")
        return
    with open(path, 'r', encoding='utf-8') as f:
        store = json.load(f)
    print(f"{'seed':>7} | {'hệ thống':<10} | {'Train/Cal/Test':>22} | {'hash_test':<16} | khớp?")
    print('-' * 78)
    for seed in sorted(store, key=lambda s: int(s)):
        entry = store[seed]
        hashes = {v['hash_test'] for v in entry.values()}
        ok = 'OK' if len(hashes) == 1 else 'LỆCH ❌'
        for system, fp in sorted(entry.items()):
            print(f"{seed:>7} | {system:<10} | "
                  f"{fp['n_train']:>6}/{fp['n_cal']:>5}/{fp['n_test']:>5}   | "
                  f"{fp['hash_test']:<16} | {ok if len(entry) > 1 else '(1 hệ)'}")


if __name__ == '__main__':
    summarize()
