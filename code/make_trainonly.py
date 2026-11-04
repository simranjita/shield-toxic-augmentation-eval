#!/usr/bin/env python3
"""Write the real TRAIN split with the same recipe as shield_pipeline.py (seed 42, 85/15 stratified on any label).
The gen2 generators draw their few-shot examples from this file.
    python3 make_trainonly.py --in data/Original.csv --out data/Original_trainonly.csv
"""
import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
df = pd.read_csv(a.inp)
any_label = (df[LABELS].sum(axis=1) > 0).astype(int).values
tr, _ = train_test_split(np.arange(len(df)), test_size=0.15, random_state=42, stratify=any_label)
df.iloc[tr].to_csv(a.out, index=False)
print(f'{len(tr):,} training rows written to {a.out}')
