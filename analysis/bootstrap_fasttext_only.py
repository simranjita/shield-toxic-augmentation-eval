#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import f1_score

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
N_BOOT = 10000
RNG = np.random.RandomState(42)

da = np.load('fasttext_naive_duplicate_test_predictions.npz')
db = np.load('fasttext_augmented_test_predictions.npz')
assert np.array_equal(da['y_test'], db['y_test'])
y_true = da['y_test']
n_test = len(y_true)

def macro_f1_at(prob, thr, idx):
    y_pred = np.stack([(prob[idx, i] >= thr[i]).astype(int) for i in range(len(thr))], axis=1)
    return f1_score(y_true[idx], y_pred, average='macro', zero_division=0)

def per_class_f1_at(prob, thr, idx, ci):
    y_pred = (prob[idx, ci] >= thr[ci]).astype(int)
    return f1_score(y_true[idx, ci], y_pred, zero_division=0)

def compare(label, class_i=None):
    if class_i is None:
        obs_a = macro_f1_at(da['mlp_prob'], da['mlp_thr'], np.arange(n_test))
        obs_b = macro_f1_at(db['mlp_prob'], db['mlp_thr'], np.arange(n_test))
    else:
        obs_a = per_class_f1_at(da['mlp_prob'], da['mlp_thr'], np.arange(n_test), class_i)
        obs_b = per_class_f1_at(db['mlp_prob'], db['mlp_thr'], np.arange(n_test), class_i)
    obs_diff = obs_b - obs_a
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.randint(0, n_test, n_test)
        if class_i is None:
            fa = macro_f1_at(da['mlp_prob'], da['mlp_thr'], idx)
            fb = macro_f1_at(db['mlp_prob'], db['mlp_thr'], idx)
        else:
            fa = per_class_f1_at(da['mlp_prob'], da['mlp_thr'], idx, class_i)
            fb = per_class_f1_at(db['mlp_prob'], db['mlp_thr'], idx, class_i)
        diffs[i] = fb - fa
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * (diffs < 0).mean() if obs_diff >= 0 else 2 * (diffs > 0).mean()
    p = min(p, 1.0)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
    print(f"[{label}] naive_dup={obs_a:.4f}  augmented={obs_b:.4f}  diff={obs_diff:+.4f}  "
          f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p~={p:.4f}  {sig}")

print("FastText+MLP, POST length-fix: naive_duplicate vs augmented")
compare("macro_f1")
compare("severe_toxic", class_i=LABELS.index('severe_toxic'))
