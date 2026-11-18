# Bảng so sánh 2 hệ thống — V40 / PhoBERT

- Số seed: **10** (`42`–`51`), dùng chung cho cả hai hệ thống.
- Chỉ số so sánh chính: **`f1_comparable`** — F1-macro theo argmax ở cả hai hệ. Với V40, đây là `f1_argmax`, KHÔNG phải `f1` (vốn đã qua MacroF1DecisionLayer).
- Dấu vân tay split: **khớp ở mọi seed** — mỗi cặp điểm đến từ đúng cùng một lần chia Train/Calibration/Test, nên kiểm định theo cặp hợp lệ.

## Bảng 1 — Hiệu năng qua các seed (mean ± std, CI 95%)

| Chỉ số | V40 | PhoBERT |
|---|---|---|
| `f1_comparable` ↑ | 0.9586 ± 0.0018<br><sub>CI95 [0.9574, 0.9599]</sub> | 0.9482 ± 0.0028<br><sub>CI95 [0.9462, 0.9502]</sub> |
| `accuracy` ↑ | 0.9585 ± 0.0019<br><sub>CI95 [0.9571, 0.9598]</sub> | 0.9480 ± 0.0028<br><sub>CI95 [0.9460, 0.9501]</sub> |
| `ece` ↓ | 0.0071 ± 0.0015<br><sub>CI95 [0.0060, 0.0082]</sub> | 0.0271 ± 0.0044<br><sub>CI95 [0.0240, 0.0302]</sub> |
| `brier` ↓ | 0.0647 ± 0.0024<br><sub>CI95 [0.0630, 0.0664]</sub> | 0.0850 ± 0.0047<br><sub>CI95 [0.0816, 0.0884]</sub> |
| `training_time_sec` ↓ | 1,309s ± 35s<br><sub>CI95 [1,284s, 1,334s]</sub> | 1,997s ± 2s<br><sub>CI95 [1,996s, 1,999s]</sub> |
| `model_size_mb` ↓ | 159.4MB ± 3.4MB<br><sub>CI95 [156.9MB, 161.8MB]</sub> | 517.0MB ± 0.0MB<br><sub>CI95 [517.0MB, 517.0MB]</sub> |

> **Lớp quyết định của V40 (báo cáo riêng, KHÔNG trộn vào bảng trên):** `delta_decision_layer` = -0.0001 ± 0.0002 — phần F1 mà `MacroF1DecisionLayer` cộng thêm cho V40 so với argmax. PhoBERT không có lớp này.

## Bảng 2 — Kiểm định theo cặp trên cùng seed

Họ so sánh: mọi đôi hệ thống × chỉ số `f1_comparable`, hiệu chỉnh **Holm-Bonferroni** ở mức α = 0.05. Wilcoxon signed-rank là kiểm định chính (không giả định phân phối chuẩn, phù hợp với n = 10); paired t-test in kèm để đối chiếu.

| So sánh | Δ trung bình | Wilcoxon p | t-test p | p (Holm) | Kết luận |
|---|---|---|---|---|---|
| V40 vs PhoBERT | +0.0105 | 0.0020 | 0.0000 | 0.0020 | **V40 cao hơn**, có ý nghĩa (p_Holm = 0.0020) |

> **Cách đọc:** "không có khác biệt có ý nghĩa" KHÔNG đồng nghĩa với "hai hệ tương đương" — với n = 10 seed, kiểm định chỉ phát hiện được khác biệt tương đối lớn. Báo cáo cả Δ và CI ở Bảng 1 để người đọc tự đánh giá độ lớn hiệu ứng, đừng chỉ dựa vào p.
