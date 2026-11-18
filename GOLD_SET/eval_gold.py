#!/usr/bin/env python
"""Tính lại chỉ số trên tập vàng (gold set) từ xác suất từng mẫu đã lưu.

Cách dùng:  python eval_gold.py
Đầu vào :  probs_gold/{lexeval,phobert}_gold_seed_{42..51}.csv
Đầu ra  :  metrics_gold.json  (mean ± std qua 10 seed, ddof=1)

Chỉ số: macro-F1, accuracy, ECE (10 khoang đều), Brier (đa lớp, tổng bình phương
sai khác với one-hot), recall từng lớp (0 = tiêu cực, 1 = trung lập, 2 = tích cực).
"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(42, 52))
SYSTEMS = {'LexEval': 'lexeval', 'PhoBERT': 'phobert'}

def ece10(P, y):
    conf, pred = P.max(1), P.argmax(1)
    ok = (pred == y)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e

def brier(P, y):
    return float(((P - np.eye(3)[y]) ** 2).sum(1).mean())

out = {'n_per_seed': None, 'class_dist_eval': None, 'systems': {}}
for name, pref in SYSTEMS.items():
    rows = {k: [] for k in ['f1_macro', 'accuracy', 'ece', 'brier', 'recall_neg', 'recall_neu', 'recall_pos']}
    for s in SEEDS:
        d = pd.read_csv(os.path.join(HERE, 'probs_gold', f'{pref}_gold_seed_{s}.csv'))
        P = d[['prob_0', 'prob_1', 'prob_2']].values
        y = d['y_true'].values
        pred = P.argmax(1)
        rows['f1_macro'].append(f1_score(y, pred, average='macro'))
        rows['accuracy'].append(float((pred == y).mean()))
        rows['ece'].append(ece10(P, y))
        rows['brier'].append(brier(P, y))
        r = recall_score(y, pred, average=None)
        rows['recall_neg'].append(r[0]); rows['recall_neu'].append(r[1]); rows['recall_pos'].append(r[2])
    out['n_per_seed'] = int(len(d))
    out['class_dist_eval'] = [round(float(x), 4) for x in np.bincount(y) / len(y)]
    out['systems'][name] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1))} for k, v in rows.items()}
    print(f"{name:8s} F1={np.mean(rows['f1_macro']):.4f}±{np.std(rows['f1_macro'], ddof=1):.4f}  "
          f"acc={np.mean(rows['accuracy']):.4f}  ECE={np.mean(rows['ece']):.4f}  Brier={np.mean(rows['brier']):.4f}  "
          f"recall(neu)={np.mean(rows['recall_neu']):.3f}")
with open(os.path.join(HERE, 'metrics_gold.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('n mỗi seed =', out['n_per_seed'], '| phân bố lớp (neg/neu/pos) =', out['class_dist_eval'])
