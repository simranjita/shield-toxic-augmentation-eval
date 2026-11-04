#!/usr/bin/env python3
"""
Fine-tuned BERT baseline (end-to-end, not frozen embeddings).

The real data are split into a fixed train/val/test FIRST; augmentation/duplication is applied to the train
portion only. Includes a 'naive_duplicate' condition (clone real minority train rows, no LLM) and dumps
per-sample test predictions for significance testing.

Usage:
    python3 bert_baseline.py --condition original --data-dir ... --out-dir ...
    python3 bert_baseline.py --condition augmented --data-dir ... --out-dir ...
    python3 bert_baseline.py --condition naive_duplicate --data-dir ... --out-dir ...
"""
import os, json, gc, argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MINORITY_LABELS = ['severe_toxic', 'threat', 'insult', 'identity_hate']

CONFIG = {
    'seed': 42, 'test_size': 0.15, 'val_size': 0.15,
    'max_len': 128, 'batch_size': 32, 'epochs': 4, 'lr': 2e-5,
    'focal_gamma': 2.0,
    'threshold_search_range': (0.1, 0.9), 'threshold_search_steps': 81,
    'non_toxic_keep_ratio': 0.5,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.pos_weight, self.gamma = pos_weight, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        w = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
        return (w * (1 - pt) ** self.gamma * bce).mean()


class BertToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), padding=True, truncation=True,
                              max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {'input_ids': self.enc['input_ids'][i],
                'attention_mask': self.enc['attention_mask'][i],
                'token_type_ids': self.enc['token_type_ids'][i],
                'labels': self.labels[i]}


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


def predict(model, dl, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in dl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tt = batch['token_type_ids'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=tt).logits
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs)


def load_labeled_csv(path):
    d = pd.read_csv(path)
    d['comment_text'] = d['comment_text'].astype(str).fillna("")
    for c in LABELS:
        if c not in d.columns: d[c] = 0
        d[c] = d[c].fillna(0).astype(int)
    return d


def split_real_data(df):
    """THE fixed, real-only split, shared across every condition -- identical logic/seed
    to shield_pipeline.py's split_real_data so both scripts' 'original'/'augmented' test
    sets are drawn the same way."""
    y = df[LABELS].values.astype(np.float32)
    toxic = (y.sum(axis=1) > 0).astype(int)
    idx = np.arange(len(df))
    idx_tmp, idx_test, toxic_tmp, _ = train_test_split(
        idx, toxic, test_size=CONFIG['test_size'], random_state=CONFIG['seed'], stratify=toxic)
    idx_tr, idx_val = train_test_split(
        idx_tmp, test_size=CONFIG['val_size'] / (1 - CONFIG['test_size']),
        random_state=CONFIG['seed'], stratify=toxic_tmp)
    train_df = df.iloc[idx_tr].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    log(f"Fixed real-only split: train={len(train_df):,}  val={len(val_df):,}  "
        f"test={len(test_df):,} (identical across every condition)")
    return train_df, val_df, test_df


def apply_condition_to_train(train_df, condition, data_dir):
    if condition == 'original':
        return train_df
    if condition == 'augmented':
        df_aug = load_labeled_csv(os.path.join(data_dir, 'Augmented.csv'))[['comment_text'] + LABELS]
        out = pd.concat([train_df, df_aug], ignore_index=True)
        log(f"Augmented (train only): {len(train_df):,} real + {len(df_aug):,} synthetic "
            f"= {len(out):,} total")
        return out
    if condition == 'naive_duplicate':
        df_aug = load_labeled_csv(os.path.join(data_dir, 'Augmented.csv'))
        dup_parts = []
        for lab in MINORITY_LABELS:
            n_needed = int(df_aug[lab].sum())
            pool = train_df[train_df[lab] == 1]
            if n_needed == 0 or len(pool) == 0:
                continue
            sampled = pool.sample(n=n_needed, replace=True, random_state=CONFIG['seed']).copy()
            for other in MINORITY_LABELS:
                if other != lab:
                    sampled[other] = 0
            dup_parts.append(sampled)
            log(f"  naive_duplicate[{lab}]: cloned {n_needed:,} real train rows "
                f"(pool size {len(pool):,}, other minority labels masked)")
        dup_df = pd.concat(dup_parts, ignore_index=True)[['comment_text'] + LABELS]
        out = pd.concat([train_df, dup_df], ignore_index=True)
        log(f"Naive-duplicate (train only): {len(train_df):,} real + {len(dup_df):,} "
            f"duplicated-real = {len(out):,} total")
        return out
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', required=True, choices=['original', 'augmented', 'naive_duplicate'])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--assets-dir', required=True)
    ap.add_argument('--model-seed', type=int, required=True)
    args = ap.parse_args()
    seed_everything(args.model_seed)   # varies init, dropout and batch order; split and data seeds stay at CONFIG['seed']
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = f"bert_{args.condition}_seed{args.model_seed}"
    log(f"=== BERT baseline v2: {run_id} | device={device} ===")

    df = load_labeled_csv(os.path.join(args.data_dir, 'Original.csv'))
    real_train_df, val_df, test_df = split_real_data(df)
    train_df = apply_condition_to_train(real_train_df, args.condition, args.data_dir)

    texts_tr = train_df['comment_text'].tolist()
    texts_val = val_df['comment_text'].tolist()
    texts_test = test_df['comment_text'].tolist()
    y_tr = train_df[LABELS].values.astype(np.float32)
    y_val = val_df[LABELS].values.astype(np.float32)
    y_test = test_df[LABELS].values.astype(np.float32)
    log(f"Final: train={len(texts_tr):,}  val={len(texts_val):,}  test={len(texts_test):,}")

    cache_dir = os.path.join(args.assets_dir, 'hf_cache')
    tok = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(LABELS), problem_type='multi_label_classification',
        cache_dir=cache_dir).to(device)

    train_dl = DataLoader(BertToxicDataset(texts_tr, y_tr, tok, CONFIG['max_len']),
                           batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(BertToxicDataset(texts_val, y_val, tok, CONFIG['max_len']),
                         batch_size=CONFIG['batch_size'], shuffle=False)
    test_dl = DataLoader(BertToxicDataset(texts_test, y_test, tok, CONFIG['max_len']),
                          batch_size=CONFIG['batch_size'], shuffle=False)

    pos_weight = torch.tensor([(len(y_tr) - y_tr[:, i].sum()) / max(y_tr[:, i].sum(), 1)
                                for i in range(len(LABELS))], dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, max=50.0)
    log(f"Per-label pos_weight (clamped<=50): {pos_weight.cpu().numpy().round(2).tolist()}")
    criterion = FocalLoss(pos_weight, gamma=CONFIG['focal_gamma'])
    opt = AdamW(model.parameters(), lr=CONFIG['lr'])

    best_f1, best_state = 0, None
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for batch in train_dl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tt = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=tt).logits
            loss = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        val_probs = predict(model, val_dl, device)
        f1 = f1_score(y_val, (val_probs > 0.5).astype(int), average='macro')
        log(f"Epoch {epoch+1}/{CONFIG['epochs']}  loss={total_loss/len(train_dl):.4f}  val macro F1={f1:.5f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    log(f"Best val macro F1: {best_f1:.5f}")

    val_probs = predict(model, val_dl, device)
    thresholds = optimize_thresholds(y_val, val_probs, CONFIG['threshold_search_range'],
                                      CONFIG['threshold_search_steps'])
    test_probs = predict(model, test_dl, device)
    y_pred = apply_thresholds(test_probs, thresholds)
    macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4,
                                    zero_division=0, output_dict=True)
    log(f"TEST macro F1={macro:.5f}  micro F1={micro:.5f}")

    results = {'run_id': run_id, 'condition': args.condition, 'model': 'bert-base-uncased',
               'n_train': len(texts_tr), 'n_val': len(texts_val), 'n_test': len(texts_test),
               'macro_f1': macro, 'micro_f1': micro, 'thresholds': thresholds,
               'per_class': report, 'config': CONFIG, 'pipeline_version': 'v2_fixed_split'}
    out_path = os.path.join(args.out_dir, f"{run_id}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    pred_path = os.path.join(args.out_dir, f"{run_id}_test_predictions.npz")
    np.savez_compressed(pred_path, y_test=y_test, bert_prob=test_probs,
                         bert_thr=np.array(thresholds), labels=np.array(LABELS))
    log(f"=== DONE. Results written to {out_path} ===")
    log(f"=== Test predictions written to {pred_path} ===")


if __name__ == '__main__':
    main()
