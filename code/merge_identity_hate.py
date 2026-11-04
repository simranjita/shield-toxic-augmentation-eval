#!/usr/bin/env python3
"""
Replace the old (63% word-salad) identity_hate block in Augmented.csv with the
newly filtered+regenerated one (160 kept originals + 345 newly generated =
505 rows), then re-run a global exact + Jaccard near-dup dedup pass across the
WHOLE file (not just identity_hate) since new rows can collide with existing
severe_toxic/threat/insult rows too.
"""
import pandas as pd
import re

NEAR_DUP_THRESHOLD = 0.5

aug = pd.read_csv("data/Augmented.csv")
print(f"Loaded Augmented.csv: {len(aug)} rows", flush=True)

# backup
aug.to_csv("data/Augmented_PRE_identity_hate_regen_backup.csv", index=False)

non_ih = aug[aug['identity_hate'] == 0].copy()
print(f"Non-identity_hate rows kept as-is: {len(non_ih)}", flush=True)

existing_ih = pd.read_csv("data/existing_identity_hate.csv")
new_ih = pd.read_csv("results/identity_hate_NEW.csv")
print(f"existing_identity_hate.csv: {len(existing_ih)} rows", flush=True)
print(f"identity_hate_NEW.csv: {len(new_ih)} rows", flush=True)

cols = ['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
existing_ih_slim = existing_ih[cols].copy()
new_ih_slim = new_ih[cols].copy()
ih_block = pd.concat([existing_ih_slim, new_ih_slim], ignore_index=True)
print(f"Combined identity_hate block: {len(ih_block)} rows", flush=True)

merged = pd.concat([non_ih[cols], ih_block], ignore_index=True)
print(f"Merged (pre-dedup): {len(merged)} rows", flush=True)

# exact dup
before = len(merged)
merged = merged.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
print(f"After exact-dup drop: {len(merged)} ({before - len(merged)} removed)", flush=True)

# global Jaccard near-dup pass
def word_set(text):
    return set(re.findall(r"[a-z']+", str(text).lower()))

kept_idx = []
kept_word_sets = []
removed = 0
for i, row in merged.iterrows():
    ws = word_set(row['comment_text'])
    if not ws:
        kept_idx.append(i)
        continue
    is_dup = False
    for other in kept_word_sets:
        inter = len(ws & other)
        union = len(ws | other)
        if union and inter / union > NEAR_DUP_THRESHOLD:
            is_dup = True
            break
    if is_dup:
        removed += 1
    else:
        kept_idx.append(i)
        kept_word_sets.append(ws)

final = merged.loc[kept_idx].reset_index(drop=True)
print(f"After global Jaccard near-dup pass: {len(final)} ({removed} removed)", flush=True)

print("\nFinal label counts:", flush=True)
for lab in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(f"  {lab}: {int(final[lab].sum())}", flush=True)

final.to_csv("data/Augmented.csv", index=False)
print(f"\nSaved {len(final)} rows to data/Augmented.csv", flush=True)
