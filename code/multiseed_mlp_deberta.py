#!/usr/bin/env python3
"""
Multi-seed check for two specific findings, MLP only (both claims in question are
"+MLP" results -- RF/LightGBM/LSTM are out of scope here):
  1. QWEN+MLP: does severe_toxic's augmented-beats-naive_duplicate win (p=0.028,
     single seed=42) hold up across training seeds?
  2. FastText+MLP: does augmented-worse-than-naive_duplicate (p=0.018, single seed=42)
     hold up across training seeds?

Design: split_real_data and apply_condition_to_train use FIXED seeds (42) regardless of
--model-seed, so the train/val/test partition and (for naive_duplicate) which real rows
get duplicated are IDENTICAL across all seed runs -- only downstream model-training
randomness (weight init, dropout masks, minibatch shuffling) varies. This isolates
training-run variance from data-composition variance, and keeps the test set exactly
the one used in the original single-seed significance tests (so those results and these
are directly comparable). Embeddings are computed ONCE per (embedding, condition) and
cached to disk, then reused across all 3 model seeds -- the expensive part (QWEN
embedding extraction) doesn't need repeating.
"""
import os, sys, json, gc, argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MINORITY_LABELS = ['severe_toxic', 'threat', 'insult', 'identity_hate']
SPLIT_SEED = 42       # fixed: defines the train/val/test partition, never varies
DATA_SEED = 42        # fixed: defines undersampling/naive_duplicate/smote row selection

CONFIG = {
    'test_size': 0.15, 'val_frac_of_temp': 0.5, 'batch_size': 128,
    'embedding_batch_size': 8, 'mlp_epochs': 16,
    'focal_alpha': [5, 10, 5, 10, 5, 10], 'focal_gamma': 2.0,
    'threshold_search_range': (0.1, 0.9), 'threshold_search_steps': 81,
    'fasttext_top_n': 500000,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base_df(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'Original.csv'))
    df['comment_text'] = df['comment_text'].astype(str).fillna("")
    for col in LABELS:
        if col not in df.columns: df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    return df


def load_augmented_df(data_dir):
    df_aug = pd.read_csv(os.path.join(data_dir, 'Augmented.csv'))
    df_aug['comment_text'] = df_aug['comment_text'].astype(str).fillna("")
    for col in LABELS:
        if col not in df_aug.columns: df_aug[col] = 0
        df_aug[col] = df_aug[col].fillna(0).astype(int)
    return df_aug[['comment_text'] + LABELS]


def split_real_data(df):
    y = df[LABELS].values.astype(np.float32)
    toxic = (y.sum(axis=1) > 0).astype(int)
    idx = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx, test_size=CONFIG['test_size'],
                                        random_state=SPLIT_SEED, stratify=toxic)
    toxic_tmp = toxic[idx_tmp]
    idx_val, idx_test = train_test_split(idx_tmp, test_size=CONFIG['val_frac_of_temp'],
                                          random_state=SPLIT_SEED, stratify=toxic_tmp)
    return (df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_val].reset_index(drop=True),
            df.iloc[idx_test].reset_index(drop=True))


def apply_condition_to_train(train_df, condition, data_dir):
    if condition == 'augmented':
        df_aug = load_augmented_df(data_dir)
        out = pd.concat([train_df, df_aug], ignore_index=True)
        log(f"Augmented (train only): {len(train_df):,} real + {len(df_aug):,} synthetic = {len(out):,}")
        return out
    if condition == 'naive_duplicate':
        df_aug = load_augmented_df(data_dir)
        dup_parts = []
        for lab in MINORITY_LABELS:
            n_needed = int(df_aug[lab].sum())
            pool = train_df[train_df[lab] == 1]
            if n_needed == 0 or len(pool) == 0:
                continue
            sampled = pool.sample(n=n_needed, replace=True, random_state=DATA_SEED).copy()
            for other in MINORITY_LABELS:
                if other != lab:
                    sampled[other] = 0
            dup_parts.append(sampled)
        dup_df = pd.concat(dup_parts, ignore_index=True)[['comment_text'] + LABELS]
        out = pd.concat([train_df, dup_df], ignore_index=True)
        log(f"Naive-duplicate (train only): {len(train_df):,} real + {len(dup_df):,} duplicated = {len(out):,}")
        return out
    raise ValueError(condition)


def encode_with_oom_retry(encode_fn, batch, device, max_gpu_retries=5, wait_s=45):
    for attempt in range(max_gpu_retries):
        try:
            return encode_fn(batch, device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            log(f"  GPU OOM (attempt {attempt+1}/{max_gpu_retries}), waiting {wait_s}s ...")
            torch.cuda.empty_cache(); gc.collect(); time.sleep(wait_s)
    log("  GPU still OOM -- falling back to CPU for this batch.")
    return encode_fn(batch, 'cpu')


def embed_qwen(texts, device, assets_dir):
    from sentence_transformers import SentenceTransformer
    log("Loading Qwen/Qwen3-Embedding-0.6B ...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device,
                                 cache_folder=os.path.join(assets_dir, 'hf_cache'))
    dim = model.get_sentence_embedding_dimension()

    def _encode(batch, dev):
        with torch.no_grad():
            return model.encode(batch, batch_size=len(batch), show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True, device=dev)

    embs = []
    bs = CONFIG['embedding_batch_size']
    for i in tqdm(range(0, len(texts), bs), desc="Qwen embed"):
        batch = [t[:2000] for t in texts[i:i + bs]]
        embs.append(encode_with_oom_retry(_encode, batch, device))
    X = np.vstack(embs)
    del model; gc.collect(); torch.cuda.empty_cache()
    return X, dim


def embed_fasttext(texts, device, assets_dir):
    from gensim.models import KeyedVectors
    vec_path = os.path.join(assets_dir, 'crawl-300d-2M.vec')
    log(f"Loading top {CONFIG['fasttext_top_n']:,} FastText vectors ...")
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False, limit=CONFIG['fasttext_top_n'])
    dim = kv.vector_size
    embs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(tqdm(texts, desc="FastText mean-pool")):
        words = str(t).lower().split()
        vecs = [kv[w] for w in words if w in kv]
        if vecs:
            embs[i] = np.mean(vecs, axis=0)
    return embs, dim


def embed_deberta(texts, device, assets_dir):
    from transformers import AutoTokenizer, AutoModel
    log("Loading microsoft/deberta-v3-base ...")
    cache_dir = os.path.join(assets_dir, 'hf_cache')
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=cache_dir,
                                         use_fast=False)
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base", cache_dir=cache_dir).to(device)
    model.eval()
    dim = model.config.hidden_size
    log(f"DeBERTa embedding dim: {dim}")
    cpu_model_holder = {}

    def _encode(batch, dev):
        m = model
        if dev == 'cpu':
            if 'model' not in cpu_model_holder:
                cpu_model_holder['model'] = AutoModel.from_pretrained(
                    "microsoft/deberta-v3-base", cache_dir=cache_dir).to('cpu').eval()
            m = cpu_model_holder['model']
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = m(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.cpu().numpy()

    embs = []
    bs = CONFIG['embedding_batch_size']
    for i in tqdm(range(0, len(texts), bs), desc="DeBERTa embed"):
        batch = [t[:2000] for t in texts[i:i + bs]]
        embs.append(encode_with_oom_retry(_encode, batch, device))
    X = np.vstack(embs)
    del model, tok
    cpu_model_holder.clear()
    gc.collect(); torch.cuda.empty_cache()
    return X, dim


EMBEDDERS = {'qwen': embed_qwen, 'fasttext': embed_fasttext, 'deberta': embed_deberta}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma
        if self.alpha is not None:
            focal = self.alpha.unsqueeze(0).expand_as(targets) * focal
        return (focal * bce).mean()


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, len(LABELS)))

    def forward(self, x):
        return self.net(x)


class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


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


def get_or_build_embeddings(embedding, condition, data_dir, assets_dir, cache_dir, device):
    cache_path = os.path.join(cache_dir, f"{embedding}_{condition}_cached.npz")
    if os.path.exists(cache_path):
        log(f"Loading cached embeddings from {cache_path}")
        d = np.load(cache_path)
        return d['X_tr'], d['y_tr'], d['X_val'], d['y_val'], d['X_test'], d['y_test']

    df = load_base_df(data_dir)
    real_train_df, val_df, test_df = split_real_data(df)
    train_df = apply_condition_to_train(real_train_df, condition, data_dir)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    texts = full_df['comment_text'].tolist()
    y = full_df[LABELS].values.astype(np.float32)
    n_tr, n_val = len(train_df), len(val_df)

    X, _ = EMBEDDERS[embedding](texts, device, assets_dir)
    X_tr, X_val, X_test = X[:n_tr], X[n_tr:n_tr + n_val], X[n_tr + n_val:]
    y_tr, y_val, y_test = y[:n_tr], y[n_tr:n_tr + n_val], y[n_tr + n_val:]

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
                         X_test=X_test, y_test=y_test)
    log(f"Cached embeddings to {cache_path}")
    return X_tr, y_tr, X_val, y_val, X_test, y_test


def train_and_eval_mlp(X_tr, y_tr, X_val, y_val, X_test, y_test, model_seed, device):
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    focal_alpha = torch.FloatTensor(CONFIG['focal_alpha']).to(device)
    train_dl = DataLoader(VecDataset(X_tr_s, y_tr), batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(VecDataset(X_val_s, y_val), batch_size=CONFIG['batch_size'], shuffle=False)
    test_dl = DataLoader(VecDataset(X_test_s, y_test), batch_size=CONFIG['batch_size'], shuffle=False)

    mlp = MLP(X_tr.shape[1]).to(device)
    opt = optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = FocalLoss(alpha=focal_alpha, gamma=CONFIG['focal_gamma'])
    best_f1, best_state, patience = 0, None, 0
    for epoch in range(CONFIG['mlp_epochs']):
        mlp.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(mlp(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            vp = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in val_dl])
        f1 = f1_score(y_val, (vp > 0.5).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_state, patience = f1, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 5:
            break
    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        val_p = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in val_dl])
        test_p = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in test_dl])
    thr = optimize_thresholds(y_val, val_p, CONFIG['threshold_search_range'], CONFIG['threshold_search_steps'])
    y_pred = apply_thresholds(test_p, thr)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4,
                                    zero_division=0, output_dict=True)
    return macro_f1, report, test_p, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding', required=True, choices=['qwen', 'fasttext', 'deberta'])
    ap.add_argument('--condition', required=True, choices=['augmented', 'naive_duplicate'])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--assets-dir', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log(f"=== Multi-seed MLP: {args.embedding}_{args.condition} | seeds={args.seeds} | device={device} ===")
    X_tr, y_tr, X_val, y_val, X_test, y_test = get_or_build_embeddings(
        args.embedding, args.condition, args.data_dir, args.assets_dir, args.cache_dir, device)
    log(f"Data: train={X_tr.shape}  val={X_val.shape}  test={X_test.shape}")

    results = {}
    all_preds = {'y_test': y_test, 'labels': np.array(LABELS)}
    for seed in args.seeds:
        log(f"--- training MLP, model_seed={seed} ---")
        macro_f1, report, test_p, thr = train_and_eval_mlp(
            X_tr, y_tr, X_val, y_val, X_test, y_test, seed, device)
        st_f1 = report['severe_toxic']['f1-score']
        log(f"  seed={seed}: macro_f1={macro_f1:.4f}  severe_toxic_f1={st_f1:.4f}")
        results[str(seed)] = {'macro_f1': macro_f1, 'per_class': report}
        all_preds[f'seed{seed}_prob'] = test_p
        all_preds[f'seed{seed}_thr'] = np.array(thr)

    out_json = os.path.join(args.out_dir, f"multiseed_{args.embedding}_{args.condition}.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    out_npz = os.path.join(args.out_dir, f"multiseed_{args.embedding}_{args.condition}_predictions.npz")
    np.savez_compressed(out_npz, **all_preds)
    log(f"=== DONE. Saved {out_json} and {out_npz} ===")
    macro_fs = [results[str(s)]['macro_f1'] for s in args.seeds]
    st_fs = [results[str(s)]['per_class']['severe_toxic']['f1-score'] for s in args.seeds]
    log(f"SUMMARY macro_f1 across seeds: mean={np.mean(macro_fs):.4f} std={np.std(macro_fs):.4f} {macro_fs}")
    log(f"SUMMARY severe_toxic_f1 across seeds: mean={np.mean(st_fs):.4f} std={np.std(st_fs):.4f} {st_fs}")


if __name__ == '__main__':
    main()
