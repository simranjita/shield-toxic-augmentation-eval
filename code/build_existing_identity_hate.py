#!/usr/bin/env python3
"""
Build existing_identity_hate.csv: filter the current 689 identity_hate rows in
Augmented.csv down to the ones that actually pass a real-data-calibrated Detoxify
identity_attack threshold, then near-dedup (Jaccard word-overlap > 0.5) the survivors.
This becomes the --existing-csv seed for regenerate_class.py's identity_hate topup.
"""
import pandas as pd
import re
from detoxify import Detoxify
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

THRESHOLD = 0.42  # ~60% of real-data mean identity_attack=0.696
NEAR_DUP_THRESHOLD = 0.5

aug = pd.read_csv("data/Augmented.csv")
ih = aug[aug['identity_hate'] == 1].copy()
print(f"Loaded {len(ih)} identity_hate rows from Augmented.csv", flush=True)

detox = Detoxify('original', device=device)
scores = detox.predict(ih['comment_text'].tolist())
ih['detoxify_identity_attack'] = scores['identity_attack']
ih['detoxify_toxicity'] = scores['toxicity']

passed = ih[ih['detoxify_identity_attack'] > THRESHOLD].copy()
print(f"{len(passed)}/{len(ih)} pass threshold {THRESHOLD} "
      f"({len(passed)/len(ih)*100:.1f}%)", flush=True)

def word_set(text):
    return set(re.findall(r"[a-z']+", str(text).lower()))

kept_rows = []
kept_word_sets = []
for _, row in passed.iterrows():
    ws = word_set(row['comment_text'])
    if not ws:
        continue
    is_dup = False
    for other in kept_word_sets:
        inter = len(ws & other)
        union = len(ws | other)
        if union and inter / union > NEAR_DUP_THRESHOLD:
            is_dup = True
            break
    if not is_dup:
        kept_rows.append(row)
        kept_word_sets.append(ws)

result = pd.DataFrame(kept_rows)
print(f"{len(result)}/{len(passed)} survive near-dup filter "
      f"({len(passed)-len(result)} near-dup rejected)", flush=True)

result.to_csv("data/existing_identity_hate.csv", index=False)
print(f"Saved {len(result)} rows to data/existing_identity_hate.csv", flush=True)
print(f"\nmean identity_attack (kept): {result['detoxify_identity_attack'].mean():.4f}")
print(f"mean identity_attack (real calibration target): 0.696")
