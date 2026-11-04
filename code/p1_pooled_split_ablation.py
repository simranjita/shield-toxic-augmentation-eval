#!/usr/bin/env python3
"""
P1: designed leakage ablation (pooled-split ablation).

Question: if the 1,923 synthetic rows were pooled with the real rows BEFORE splitting
(instead of being added to the train split only), how much would the augmented condition
look better than it is?

Design, using only the final files and the final cached embeddings (no new embedding
extraction, no earlier run is reused):
  1. Rebuild the pool from the cached augmented-condition arrays: real train rows
     (X_tr minus its last n_synth rows), real val, real test, plus the n_synth synthetic rows.
  2. Split the POOL at random with the same recipe as the final protocol (85/7.5/7.5,
     stratified on any-label).  Nothing else changes.
  3. Train the same MLP (multiseed_mlp.train_and_eval_mlp) in two ways on the pooled train:
        pooled_augmented : train rows as split (real + synthetic)
        pooled_original  : the same train rows with synthetic rows removed
     Both use the same pooled val and the same pooled test.
  4. Score each on (a) the full pooled test (contains synthetic rows: the leaky protocol)
     and (b) the real-only subset of that same test.  The split is identical, only the test
     composition differs, so the (a)-(b) difference isolates the contamination.
  5. Report the synthetic share of test positives per class directly.
Seeds: the seed drives both the pool split and the model initialisation (3 seeds).
"""
import os, sys, json, argparse
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiseed_mlp as M

LABELS = M.LABELS


def per_class_f1(y_true, y_pred):
    return {l: float(v) for l, v in zip(LABELS, f1_score(y_true, y_pred, average=None, zero_division=0))}


def score(y_true, test_p, thr):
    y_pred = M.apply_thresholds(test_p, thr)
    return {'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'per_class': per_class_f1(y_true, y_pred), 'n': int(len(y_true)),
            'positives': {l: int(y_true[:, i].sum()) for i, l in enumerate(LABELS)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-file', required=True, help='<embedding>_augmented_cached.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-synth', type=int, default=1923)
    ap.add_argument('--n-real', type=int, default=159571)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--expect-threat', type=int, default=500, help='synthetic threat rows expected in the tail block')
    ap.add_argument('--max-rows', type=int, default=0, help='debug only: subsample the real pool')
    args = ap.parse_args()
    device = 'cuda' if M.torch.cuda.is_available() else 'cpu'

    d = np.load(args.cache_file)
    X_tr, y_tr, X_val, y_val, X_te, y_te = (d[k] for k in ('X_tr', 'y_tr', 'X_val', 'y_val', 'X_test', 'y_test'))
    ns = args.n_synth
    X_real = np.vstack([X_tr[:-ns], X_val, X_te]); y_real = np.vstack([y_tr[:-ns], y_val, y_te])
    X_syn, y_syn = X_tr[-ns:], y_tr[-ns:]
    assert len(X_real) == args.n_real, (len(X_real), args.n_real)
    assert int(y_syn[:, LABELS.index('threat')].sum()) == args.expect_threat, 'tail is not the synthetic block'
    X = np.vstack([X_real, X_syn]); y = np.vstack([y_real, y_syn])
    is_syn = np.r_[np.zeros(len(X_real), bool), np.ones(ns, bool)]
    if args.max_rows:
        rng = np.random.RandomState(0)
        keep = np.r_[rng.choice(len(X_real), args.max_rows, replace=False), np.arange(len(X_real), len(X))]
        X, y, is_syn = X[keep], y[keep], is_syn[keep]
    M.log(f"pool: {len(X):,} rows ({int(is_syn.sum()):,} synthetic), dim={X.shape[1]}, device={device}")

    any_label = (y.sum(1) > 0).astype(int)
    out = {'cache_file': os.path.basename(args.cache_file), 'pool_rows': int(len(X)),
           'synthetic_rows': int(is_syn.sum()), 'seeds': {}}
    for seed in args.seeds:
        idx = np.arange(len(X))
        i_tr, i_tmp = train_test_split(idx, test_size=0.15, random_state=seed, stratify=any_label)
        i_val, i_te = train_test_split(i_tmp, test_size=0.5, random_state=seed, stratify=any_label[i_tmp])
        te_syn = is_syn[i_te]
        share = {l: {'synthetic': int(y[i_te][te_syn][:, i].sum()), 'total': int(y[i_te][:, i].sum())}
                 for i, l in enumerate(LABELS)}
        rec = {'test_synthetic_share_of_positives': share, 'test_rows': int(len(i_te)),
               'test_synthetic_rows': int(te_syn.sum())}
        M.log(f"seed {seed}: test {len(i_te):,} rows, {int(te_syn.sum())} synthetic; threat positives "
              f"{share['threat']['synthetic']}/{share['threat']['total']} synthetic")
        for name, tr_idx in (('pooled_augmented', i_tr), ('pooled_original', i_tr[~is_syn[i_tr]])):
            _, _, test_p, thr = M.train_and_eval_mlp(X[tr_idx], y[tr_idx], X[i_val], y[i_val], X[i_te], y[i_te], seed, device)
            rec[name] = {'train_rows': int(len(tr_idx)),
                         'full_pooled_test': score(y[i_te], test_p, thr),
                         'real_only_test': score(y[i_te][~te_syn], test_p[~te_syn], thr)}
            M.log(f"  {name}: full-test macro {rec[name]['full_pooled_test']['macro_f1']:.4f} | "
                  f"real-only macro {rec[name]['real_only_test']['macro_f1']:.4f} | threat full "
                  f"{rec[name]['full_pooled_test']['per_class']['threat']:.3f} real-only "
                  f"{rec[name]['real_only_test']['per_class']['threat']:.3f}")
        out['seeds'][str(seed)] = rec
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    M.log(f"saved {args.out}")


if __name__ == '__main__':
    main()
