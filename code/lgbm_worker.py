#!/usr/bin/env python3
"""
Standalone LightGBM training step, run as a SEPARATE PROCESS from shield_pipeline.py.

Why: on this HPC, LightGBM trains fine in isolation (verified: 3000-estimator fit with
early stopping completes with ~190MB RSS in a clean process) but gets SIGKILLed every time
it runs in the same process as the earlier torch/CUDA embedding+MLP+LSTM steps -- reproduced
consistently across ncpus=1..4, mem=16..32gb, and with OMP/MKL/OPENBLAS thread caps in place.
Root cause not fully isolated (most likely a CUDA-context/host-memory interaction with
LightGBM's own allocator) -- running it in a throwaway subprocess with NO torch import
sidesteps the interaction entirely rather than chasing the exact mechanism. Deliberately
does not import torch/transformers/sentence_transformers: numpy + sklearn + lightgbm only.
"""
import argparse, json, sys
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from lightgbm import LGBMClassifier
import lightgbm as lgbm


def optimize_thresholds(y_true, y_prob, search_range, steps):
    thresholds = np.linspace(*search_range, steps)
    best_t = []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_prob[:, c]
        scores = [f1_score(yt, (yp >= t).astype(int), average='binary', zero_division=0) for t in thresholds]
        best_t.append(float(thresholds[int(np.argmax(scores))]))
    return best_t


def apply_thresholds(y_prob, thresholds):
    return np.stack([(y_prob[:, i] >= t).astype(int) for i, t in enumerate(thresholds)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-npz', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-test-probs', default=None)
    ap.add_argument('--n-jobs', type=int, default=1)
    args = ap.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_tr, y_tr = d['X_tr'], d['y_tr']
    X_val, y_val = d['X_val'], d['y_val']
    X_test, y_test = d['X_test'], d['y_test']
    labels = list(d['labels'])
    focal_alpha = list(d['focal_alpha'])
    search_range = tuple(d['threshold_search_range'])
    steps = int(d['threshold_search_steps'])

    print(f"[lgbm_worker] train={X_tr.shape} val={X_val.shape} test={X_test.shape}", flush=True)

    models = []
    for i, label in enumerate(labels):
        m = LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=64,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            scale_pos_weight=focal_alpha[i], random_state=42,
                            n_jobs=args.n_jobs, verbosity=-1)
        m.fit(X_tr, y_tr[:, i], eval_set=[(X_val, y_val[:, i])],
              callbacks=[lgbm.early_stopping(150, verbose=False)])
        models.append(m)
        print(f"[lgbm_worker] {label} done", flush=True)

    val_p = np.column_stack([m.predict_proba(X_val)[:, 1] for m in models])
    test_p = np.column_stack([m.predict_proba(X_test)[:, 1] for m in models])
    thr = optimize_thresholds(y_val, val_p, search_range, steps)
    pred = apply_thresholds(test_p, thr)

    macro = f1_score(y_test, pred, average='macro', zero_division=0)
    micro = f1_score(y_test, pred, average='micro', zero_division=0)
    prec_macro = precision_score(y_test, pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, pred, average='macro', zero_division=0)
    per_class = classification_report(y_test, pred, target_names=labels, digits=4,
                                       zero_division=0, output_dict=True)
    print(f"[lgbm_worker] macro F1={macro:.5f} micro F1={micro:.5f}", flush=True)

    result = {'macro_f1': macro, 'micro_f1': micro, 'macro_precision': prec_macro,
              'macro_recall': rec_macro, 'per_class': per_class, 'thresholds': thr}
    with open(args.out_json, 'w') as f:
        json.dump(result, f, default=float)
    if args.out_test_probs:
        np.save(args.out_test_probs, test_p)
    print("[lgbm_worker] DONE", flush=True)


if __name__ == '__main__':
    main()
