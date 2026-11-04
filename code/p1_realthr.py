#!/usr/bin/env python3
"""
P1b: threshold-tuning control for the pooled-split ablation (review point 3).

Same design as p1_pooled_split_ablation.py (same pool, same random split recipe, same MLP, same seeds),
but each trained model is scored under TWO threshold rules:
  pooled_thr : per-label thresholds tuned on the full pooled validation set (rows include synthetic), as in P1
  real_thr   : per-label thresholds tuned on the REAL rows of the pooled validation set only
and on two test views: the full pooled test and its real-only subset.
The 'pooled_thr' scores must reproduce p1_*.json (determinism check, reported in the log).
If the QWEN real-only loss survives 'real_thr', it is not an artifact of thresholds fit to a mixed validation set.
"""
import os, sys, json, argparse
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiseed_mlp as M
LABELS = M.LABELS
torch, nn, optim, DataLoader = M.torch, M.nn, M.optim, M.DataLoader


def train_probs(X_tr, y_tr, X_val, y_val, X_test, y_test, model_seed, device):
    """Identical to M.train_and_eval_mlp, but returns validation and test probabilities."""
    np.random.seed(model_seed); torch.manual_seed(model_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(model_seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_val_s = scaler.transform(X_val); X_test_s = scaler.transform(X_test)
    focal_alpha = torch.FloatTensor(M.CONFIG['focal_alpha']).to(device)
    train_dl = DataLoader(M.VecDataset(X_tr_s, y_tr), batch_size=M.CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(M.VecDataset(X_val_s, y_val), batch_size=M.CONFIG['batch_size'], shuffle=False)
    test_dl = DataLoader(M.VecDataset(X_test_s, y_test), batch_size=M.CONFIG['batch_size'], shuffle=False)
    mlp = M.MLP(X_tr.shape[1]).to(device)
    opt = optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = M.FocalLoss(alpha=focal_alpha, gamma=M.CONFIG['focal_gamma'])
    best_f1, best_state, patience = 0, None, 0
    for epoch in range(M.CONFIG['mlp_epochs']):
        mlp.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(mlp(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            vp = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in val_dl])
        f1 = f1_score(y_val, (vp > 0.5).astype(int), average='macro')
        if f1 > best_f1: best_f1, best_state, patience = f1, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else: patience += 1
        if patience >= 5: break
    mlp.load_state_dict(best_state); mlp.eval()
    with torch.no_grad():
        val_p = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in val_dl])
        test_p = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in test_dl])
    return val_p, test_p


def score(y_true, p, thr):
    y_pred = M.apply_thresholds(p, thr)
    return {'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'per_class': {l: float(v) for l, v in zip(LABELS, f1_score(y_true, y_pred, average=None, zero_division=0))},
            'n': int(len(y_true))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-file', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n-synth', type=int, default=1923); ap.add_argument('--n-real', type=int, default=159571)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1011])
    ap.add_argument('--expect-threat', type=int, default=500)
    ap.add_argument('--max-rows', type=int, default=0, help='debug only')
    ap.add_argument('--reference', default='', help='p1 json(s) to check pooled_thr against, comma separated')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(args.cache_file)
    X_tr, y_tr, X_val, y_val, X_te, y_te = (d[k] for k in ('X_tr', 'y_tr', 'X_val', 'y_val', 'X_test', 'y_test'))
    ns = args.n_synth
    X_real = np.vstack([X_tr[:-ns], X_val, X_te]); y_real = np.vstack([y_tr[:-ns], y_val, y_te])
    X_syn, y_syn = X_tr[-ns:], y_tr[-ns:]
    assert len(X_real) == args.n_real and int(y_syn[:, LABELS.index('threat')].sum()) == args.expect_threat
    X = np.vstack([X_real, X_syn]); y = np.vstack([y_real, y_syn])
    is_syn = np.r_[np.zeros(len(X_real), bool), np.ones(ns, bool)]
    if args.max_rows:
        rng = np.random.RandomState(0)
        keep = np.r_[rng.choice(len(X_real), args.max_rows, replace=False), np.arange(len(X_real), len(X))]
        X, y, is_syn = X[keep], y[keep], is_syn[keep]
    ref = {}
    for p in [p for p in args.reference.split(',') if p]:
        ref.update(json.load(open(p))['seeds'])
    any_label = (y.sum(1) > 0).astype(int)
    out = {'cache_file': os.path.basename(args.cache_file), 'seeds': {}}
    for seed in args.seeds:
        idx = np.arange(len(X))
        i_tr, i_tmp = train_test_split(idx, test_size=0.15, random_state=seed, stratify=any_label)
        i_val, i_te = train_test_split(i_tmp, test_size=0.5, random_state=seed, stratify=any_label[i_tmp])
        te_syn, val_syn = is_syn[i_te], is_syn[i_val]
        rec = {'val_rows': int(len(i_val)), 'val_synthetic_rows': int(val_syn.sum())}
        for name, tr_idx in (('pooled_augmented', i_tr), ('pooled_original', i_tr[~is_syn[i_tr]])):
            val_p, test_p = train_probs(X[tr_idx], y[tr_idx], X[i_val], y[i_val], X[i_te], y[i_te], seed, device)
            thr_pooled = M.optimize_thresholds(y[i_val], val_p, M.CONFIG['threshold_search_range'], M.CONFIG['threshold_search_steps'])
            thr_real = M.optimize_thresholds(y[i_val][~val_syn], val_p[~val_syn], M.CONFIG['threshold_search_range'], M.CONFIG['threshold_search_steps'])
            rec[name] = {'thr_pooled': thr_pooled, 'thr_real': thr_real}
            for tn, thr in (('pooled_thr', thr_pooled), ('real_thr', thr_real)):
                rec[name][tn] = {'full_pooled_test': score(y[i_te], test_p, thr),
                                 'real_only_test': score(y[i_te][~te_syn], test_p[~te_syn], thr)}
            msg = f"seed {seed} {name}: real-only macro pooled_thr {rec[name]['pooled_thr']['real_only_test']['macro_f1']:.4f} real_thr {rec[name]['real_thr']['real_only_test']['macro_f1']:.4f}"
            if str(seed) in ref:
                r0 = ref[str(seed)][name]['real_only_test']['macro_f1']
                msg += f" | reference {r0:.4f} (diff {rec[name]['pooled_thr']['real_only_test']['macro_f1'] - r0:+.5f})"
            M.log(msg)
        out['seeds'][str(seed)] = rec
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2); M.log(f"saved {args.out}")


if __name__ == '__main__':
    main()
