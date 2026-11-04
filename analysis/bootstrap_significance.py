#!/usr/bin/env python3
"""
Paired bootstrap significance testing. Valid because every condition per embedding now
shares the IDENTICAL fixed real test set (verified: y_test byte-identical across all 5
conditions per embedding, and across BERT's 3 conditions) -- so for each bootstrap
resample of test-set indices, we can score every condition's predictions on the SAME
resampled examples and compare directly (paired design).

10,000 resamples. Reports: macro-F1 for each condition on the full (non-resampled) test
set, the bootstrap 95% CI of the DIFFERENCE (condition B - condition A), and a two-sided
p-value (fraction of resamples where the sign of the difference flips relative to the
observed direction).
"""
import numpy as np
from sklearn.metrics import f1_score

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
N_BOOT = 10000
RNG = np.random.RandomState(42)


def load(path):
    return np.load(path)


def macro_f1_at(y_true, prob, thr, idx):
    y_pred = np.stack([(prob[idx, i] >= thr[i]).astype(int) for i in range(len(thr))], axis=1)
    return f1_score(y_true[idx], y_pred, average='macro', zero_division=0)


def per_class_f1_at(y_true, prob, thr, idx, class_i):
    y_pred = (prob[idx, class_i] >= thr[class_i]).astype(int)
    return f1_score(y_true[idx, class_i], y_pred, zero_division=0)


def bootstrap_compare(name_a, y_a, prob_a, thr_a, name_b, y_b, prob_b, thr_b, n=None, class_i=None):
    assert np.array_equal(y_a, y_b), "y_test must be identical for a paired test"
    y_true = y_a
    n_test = len(y_true)
    if class_i is None:
        obs_a = macro_f1_at(y_true, prob_a, thr_a, np.arange(n_test))
        obs_b = macro_f1_at(y_true, prob_b, thr_b, np.arange(n_test))
    else:
        obs_a = per_class_f1_at(y_true, prob_a, thr_a, np.arange(n_test), class_i)
        obs_b = per_class_f1_at(y_true, prob_b, thr_b, np.arange(n_test), class_i)
    obs_diff = obs_b - obs_a

    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.randint(0, n_test, n_test)
        if class_i is None:
            fa = macro_f1_at(y_true, prob_a, thr_a, idx)
            fb = macro_f1_at(y_true, prob_b, thr_b, idx)
        else:
            fa = per_class_f1_at(y_true, prob_a, thr_a, idx, class_i)
            fb = per_class_f1_at(y_true, prob_b, thr_b, idx, class_i)
        diffs[i] = fb - fa

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: fraction of bootstrap diffs on the opposite side of 0 from the
    # observed diff, doubled (standard percentile-bootstrap p-value approximation)
    if obs_diff >= 0:
        p = 2 * (diffs < 0).mean()
    else:
        p = 2 * (diffs > 0).mean()
    p = min(p, 1.0)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
    label = f"class={LABELS[class_i]}" if class_i is not None else "macro_f1"
    print(f"  [{label}] {name_a}={obs_a:.4f}  {name_b}={obs_b:.4f}  "
          f"diff={obs_diff:+.4f}  95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p~={p:.4f}  {sig}")


def compare_condition(embedding, model_key, cond_a, cond_b, class_i=None):
    da = load(f"{embedding}_{cond_a}_test_predictions.npz")
    db = load(f"{embedding}_{cond_b}_test_predictions.npz")
    bootstrap_compare(cond_a, da['y_test'], da[f'{model_key}_prob'], da[f'{model_key}_thr'],
                       cond_b, db['y_test'], db[f'{model_key}_prob'], db[f'{model_key}_thr'],
                       class_i=class_i)


def compare_bert(cond_a, cond_b, class_i=None):
    da = load(f"bert_{cond_a}_test_predictions.npz")
    db = load(f"bert_{cond_b}_test_predictions.npz")
    bootstrap_compare(cond_a, da['y_test'], da['bert_prob'], da['bert_thr'],
                       cond_b, db['y_test'], db['bert_prob'], db['bert_thr'],
                       class_i=class_i)


print("=" * 70)
print("BERT: augmented vs original (macro F1 + the 2 classes that reversed)")
print("=" * 70)
compare_bert('original', 'augmented')
compare_bert('original', 'augmented', class_i=LABELS.index('threat'))
compare_bert('original', 'augmented', class_i=LABELS.index('severe_toxic'))
compare_bert('original', 'augmented', class_i=LABELS.index('identity_hate'))

print()
print("=" * 70)
print("BERT: augmented vs naive_duplicate (is LLM text better than just duplicating real data?)")
print("=" * 70)
compare_bert('naive_duplicate', 'augmented')
compare_bert('naive_duplicate', 'augmented', class_i=LABELS.index('threat'))
compare_bert('naive_duplicate', 'augmented', class_i=LABELS.index('severe_toxic'))

print()
print("=" * 70)
print("QWEN+MLP: augmented vs original, vs SMOTE, vs naive_duplicate (macro F1)")
print("=" * 70)
compare_condition('qwen', 'mlp', 'original', 'augmented')
compare_condition('qwen', 'mlp', 'smote', 'augmented')
compare_condition('qwen', 'mlp', 'naive_duplicate', 'augmented')

print()
print("=" * 70)
print("QWEN+MLP: augmented vs original/smote/naive_dup on threat and severe_toxic specifically")
print("=" * 70)
for cls in ['threat', 'severe_toxic']:
    print(f"-- {cls} --")
    compare_condition('qwen', 'mlp', 'original', 'augmented', class_i=LABELS.index(cls))
    compare_condition('qwen', 'mlp', 'smote', 'augmented', class_i=LABELS.index(cls))
    compare_condition('qwen', 'mlp', 'naive_duplicate', 'augmented', class_i=LABELS.index(cls))

print()
print("=" * 70)
print("DeBERTa+MLP and FastText+MLP: augmented vs original (macro F1) -- both showed augmented < original")
print("=" * 70)
compare_condition('deberta', 'mlp', 'original', 'augmented')
compare_condition('fasttext', 'mlp', 'original', 'augmented')
compare_condition('fasttext', 'mlp', 'naive_duplicate', 'augmented')
