#!/usr/bin/env python3
"""
Detoxify audit of Augmented.csv (and a real-data calibration sample from Original.csv).

Detoxify (unitary/toxic-bert lineage) is a purpose-built toxicity classifier trained on
Jigsaw-family data -- it gives an independent, non-refusing, directly-comparable score per
category, unlike an LLM judge which has to be prompted/framed carefully. This is Stage 1 of
the quality audit: hard numbers before any LLM-as-judge pass.

Categories: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
(identity_attack ~= identity_hate in Jigsaw's original naming)
"""
import pandas as pd
import numpy as np
from detoxify import Detoxify
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

print("Loading Detoxify model...", flush=True)
model = Detoxify('original', device=device)

def score_batch(texts, batch_size=32):
    all_scores = {k: [] for k in ['toxicity','severe_toxicity','obscene','threat','insult','identity_attack']}
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size]]
        res = model.predict(batch)
        for k in all_scores:
            all_scores[k].extend(res[k])
        if i % (batch_size*10) == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    return all_scores

# ---- Augmented.csv ----
print("\n=== Scoring Augmented.csv (3000 rows) ===", flush=True)
aug = pd.read_csv("data/Dataset_Augmented.csv")
t0 = time.time()
scores = score_batch(aug['comment_text'].tolist())
for k, v in scores.items():
    aug[f'detoxify_{k}'] = v
aug.to_csv("results/augmented_detoxify_scores.csv", index=False)
print(f"Done in {time.time()-t0:.1f}s", flush=True)

# ---- Original.csv calibration sample: 300 real severe_toxic rows ----
print("\n=== Scoring 300 real severe_toxic rows from Original.csv (calibration) ===", flush=True)
orig = pd.read_csv("data/Dataset_Original.csv")
real_st = orig[orig['severe_toxic']==1].sample(min(300, (orig['severe_toxic']==1).sum()), random_state=42)
scores2 = score_batch(real_st['comment_text'].tolist())
for k, v in scores2.items():
    real_st[f'detoxify_{k}'] = v
real_st.to_csv("results/real_severe_toxic_detoxify_scores.csv", index=False)

# ---- Summary ----
print("\n" + "="*70)
print("SUMMARY: mean detoxify_severe_toxicity by assigned label, Augmented.csv")
print("="*70)
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
for lab in labels:
    sub = aug[aug[lab]==1]
    if len(sub) == 0:
        print(f"{lab:15s}: n=0")
        continue
    print(f"{lab:15s}: n={len(sub):4d}  mean_toxicity={sub['detoxify_toxicity'].mean():.3f}  "
          f"mean_severe_toxicity={sub['detoxify_severe_toxicity'].mean():.3f}")

print("\nReal severe_toxic (Original.csv) calibration:")
print(f"  n={len(real_st)}  mean_toxicity={real_st['detoxify_toxicity'].mean():.3f}  "
      f"mean_severe_toxicity={real_st['detoxify_severe_toxicity'].mean():.3f}")

print("\nAugmented severe_toxic vs real severe_toxic -- direct comparison:")
aug_st = aug[aug['severe_toxic']==1]
print(f"  Augmented severe_toxic  mean_severe_toxicity = {aug_st['detoxify_severe_toxicity'].mean():.4f}")
print(f"  Real severe_toxic       mean_severe_toxicity = {real_st['detoxify_severe_toxicity'].mean():.4f}")
print(f"  Augmented severe_toxic rows scoring detoxify_severe_toxicity < 0.1: "
      f"{(aug_st['detoxify_severe_toxicity'] < 0.1).sum()} / {len(aug_st)} "
      f"({(aug_st['detoxify_severe_toxicity'] < 0.1).mean()*100:.1f}%)")
print("\nDONE")
