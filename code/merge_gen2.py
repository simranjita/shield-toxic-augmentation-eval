#!/usr/bin/env python3
"""Builds Augmented_gen2.csv: generator-2 rows for threat and severe_toxic, generator-1 rows for insult and identity_hate,
with the SAME per-class counts as the final generator-1 file (severe_toxic 477, threat 500, insult 501, identity_hate 499; 1,923 rows)."""
import re, sys, argparse, random
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--gen1', required=True); ap.add_argument('--threat', required=True)
ap.add_argument('--severe', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
L = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
def ws(t): return set(re.findall(r"[a-z']+", str(t).lower()))
def near(t, sets, th=0.5):
    w = ws(t)
    return bool(w) and any(len(w & o) / max(len(w | o), 1) > th for o in sets)
g1 = pd.read_csv(a.gen1); keep = g1[(g1.severe_toxic == 0) & (g1.threat == 0)][['comment_text'] + L].copy()
assert len(keep) == 946 and int(keep.insult.sum()) == 501 and int(keep.identity_hate.sum()) == 499, 'unexpected generator-1 composition'
sets = [ws(t) for t in keep.comment_text]; random.seed(2026)
def prep(path, label, n):
    d = pd.read_csv(path); d = d.sample(frac=1, random_state=2026).reset_index(drop=True); out = []
    for t in d.comment_text:
        if not near(t, sets): out.append(t); sets.append(ws(t))
    print(f"{label}: {len(d)} accepted by the generator run, {len(out)} after cross-class near-duplicate removal, need {n}")
    if len(out) < n: sys.exit(f"NOT ENOUGH {label} rows: {len(out)} < {n}")
    rows = pd.DataFrame({'comment_text': out[:n]})
    for c in L: rows[c] = 0
    rows['toxic'] = 1; rows[label] = 1; return rows
thr = prep(a.threat, 'threat', 500); sev = prep(a.severe, 'severe_toxic', 477)
res = pd.concat([keep, thr, sev], ignore_index=True)[['comment_text'] + L]
assert len(res) == 1923 and res.comment_text.duplicated().sum() == 0
print('final per-class counts:', res[['severe_toxic', 'threat', 'insult', 'identity_hate']].sum().to_dict(), '| rows', len(res))
res.to_csv(a.out, index=False); print('saved', a.out)
