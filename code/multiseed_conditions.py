#!/usr/bin/env python3
"""
Five-seed MLP runs for the ORIGINAL, UNDERSAMPLED and SMOTE conditions (review point 2), on QWEN and FastText.

No new embedding extraction: the cached augmented-condition arrays hold the real training rows first
(X_tr[:-n_synth], in the order of the real training split), so the three conditions are rebuilt from them
with the same rules, seeds and row order as shield_pipeline.py:
  original     - the real training rows
  undersampled - all toxic rows, then non_toxic_keep_ratio (0.5) of the non-toxic rows (pandas sample, random_state=42)
  smote        - toxic rows + non-toxic undersampled to 3x the toxic count (random_state=42), then ml_smote(seed=42)
The seed-42 macro F1 must match the main grid (<grid-dir>/<emb>_<condition>.json); the log reports the difference.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiseed_mlp as M
import shield_pipeline as SP
LABELS = M.LABELS


def build(condition, X_real, y_real):
    df = pd.DataFrame(y_real, columns=LABELS)
    is_toxic = df[LABELS].sum(axis=1) > 0
    if condition == 'original':
        return X_real, y_real
    toxic_idx = df.index[is_toxic]; non_idx = df.index[~is_toxic]
    if condition == 'undersampled':
        n = int(len(non_idx) * SP.CONFIG['non_toxic_keep_ratio'])
    else:
        n = min(len(non_idx), len(toxic_idx) * SP.CONFIG['smote_neg_multiplier'])
    sampled = df.loc[non_idx].sample(n=n, random_state=SP.CONFIG['seed']).index
    sel = np.r_[toxic_idx.values, sampled.values]
    X, y = X_real[sel], y_real[sel]
    if condition == 'smote':
        X, y = SP.ml_smote(X, y, seed=SP.CONFIG['seed'])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding', required=True, choices=['qwen', 'fasttext'])
    ap.add_argument('--conditions', nargs='+', default=['original', 'undersampled', 'smote'])
    ap.add_argument('--cache-file', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--grid-dir', default='results', help='folder with <emb>_<condition>.json from the main grid')
    ap.add_argument('--n-synth', type=int, default=1923)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1011])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if M.torch.cuda.is_available() else 'cpu'
    d = np.load(args.cache_file)
    ns = args.n_synth
    X_real, y_real = d['X_tr'][:-ns], d['y_tr'][:-ns]
    X_val, y_val, X_test, y_test = d['X_val'], d['y_val'], d['X_test'], d['y_test']
    assert len(X_real) == 135635 and len(X_val) == 11968 and len(X_test) == 11968, (len(X_real), len(X_val), len(X_test))
    for cond in args.conditions:
        X_tr, y_tr = build(cond, X_real, y_real)
        M.log(f"=== {args.embedding} {cond}: train rows {len(X_tr):,} ===")
        results, preds = {}, {'y_test': y_test, 'labels': np.array(LABELS)}
        for seed in args.seeds:
            macro, report, test_p, thr = M.train_and_eval_mlp(X_tr, y_tr, X_val, y_val, X_test, y_test, seed, device)
            results[str(seed)] = {'macro_f1': macro, 'per_class': report}
            preds[f'seed{seed}_prob'] = test_p; preds[f'seed{seed}_thr'] = np.array(thr)
            msg = f"  seed {seed}: macro_f1={macro:.4f}"
            if seed == 42:
                g = os.path.join(args.grid_dir, f'{args.embedding}_{cond}.json')
                if os.path.exists(g):
                    ref = json.load(open(g))['mlp']['macro_f1']
                    msg += f" | main-grid {ref:.4f} (diff {macro - ref:+.5f})"
            M.log(msg)
        json.dump(results, open(os.path.join(args.out_dir, f'multiseed_{args.embedding}_{cond}.json'), 'w'), indent=2, default=float)
        np.savez_compressed(os.path.join(args.out_dir, f'multiseed_{args.embedding}_{cond}_predictions.npz'), **preds)
        f = [results[str(s)]['macro_f1'] for s in args.seeds]
        M.log(f"SUMMARY {args.embedding} {cond}: mean={np.mean(f):.4f} std={np.std(f):.4f} {['%.4f' % x for x in f]}")


if __name__ == '__main__':
    main()
