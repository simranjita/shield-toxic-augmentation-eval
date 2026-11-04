#!/usr/bin/env python3
"""
SHIELD experiment grid — one embedding x one data condition -> RF/LightGBM/MLP/LSTM
with weighted focal loss + per-class threshold optimization.

The real data are split ONCE into a fixed train/val/test (identical across every condition); all
condition-specific rebalancing or augmentation is then applied to the TRAIN portion alone. The val/test
sets are always the same 100%-real held-out rows, so F1 numbers are comparable across conditions and a
paired significance test across conditions is valid (same test examples every time).

Also provides:
  - a 'naive_duplicate' condition: clones real minority-class TRAIN rows (no LLM, no
    interpolation) to match the augmented condition's per-class synthetic counts --
    isolates "more positive examples" from "LLM-authored diverse content."
  - per-sample test-set prediction dumping (probabilities + thresholds) alongside the
    JSON, needed for bootstrap significance testing without retraining.
  - LSTM vocabulary built from TRAIN text only.

Usage:
    python3 shield_pipeline.py --embedding qwen --condition augmented \
        --data-dir /path/to/shield/data --out-dir /path/to/shield/results \
        --assets-dir /path/to/shield/assets

Conditions:
    original         - raw Jigsaw data, no rebalancing
    undersampled     - keep all toxic train rows, keep non_toxic_keep_ratio of non-toxic train rows
    smote            - undersample train negatives to 1:3, then MLSMOTE the train embedding matrix to 1:1
    augmented        - real train data + the Detoxify-verified LLM-augmented set (train only)
    naive_duplicate  - real train data + duplicated real minority train rows (count-matched to augmented)

Embeddings:
    qwen      - Qwen/Qwen3-Embedding-0.6B via sentence-transformers (pooling as set in the model config: last token), 1024-dim
    deberta   - microsoft/deberta-v3-base via transformers AutoModel, mean-pooled, 768-dim
    fasttext  - crawl-300d-2M.vec (gensim KeyedVectors), mean-pooled per comment, 300-dim
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMClassifier
import lightgbm as lgbm
from tqdm import tqdm

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MINORITY_LABELS = ['severe_toxic', 'threat', 'insult', 'identity_hate']  # the ones ever augmented

CONFIG = {
    'seed': 42,
    'test_size': 0.15,
    'val_frac_of_temp': 0.5,          # matches source notebooks: temp split 50/50 into val/test
    'batch_size': 128,
    'embedding_batch_size': 8,
    'max_seq_length': 200,
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'lstm_dropout': 0.3,
    'lstm_epochs': 10,
    'mlp_epochs': 16,
    'focal_alpha': [5, 10, 5, 10, 5, 10],
    'focal_gamma': 2.0,
    'threshold_search_range': (0.1, 0.9),
    'threshold_search_steps': 81,
    'non_toxic_keep_ratio': 0.5,      # undersampling: ~1:5 toxic:non-toxic, per paper III.A.1.a
    'smote_neg_multiplier': 3,        # undersample negatives to 1:3 before MLSMOTE to 1:1
    'fasttext_top_n': 500000,
}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================
# DATA LOADING + THE ONE FIXED REAL-ONLY SPLIT
# =====================================================================
def load_base_df(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'Original.csv'))
    df['comment_text'] = df['comment_text'].astype(str).fillna("")
    for col in LABELS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    return df


def load_augmented_df(data_dir):
    aug_path = os.path.join(data_dir, 'Augmented.csv')
    df_aug = pd.read_csv(aug_path)
    df_aug['comment_text'] = df_aug['comment_text'].astype(str).fillna("")
    for col in LABELS:
        if col not in df_aug.columns:
            df_aug[col] = 0
        df_aug[col] = df_aug[col].fillna(0).astype(int)
    return df_aug[['comment_text'] + LABELS]


def split_real_data(df):
    """THE fixed, real-only train/val/test split, shared by every condition. Must run
    before any condition-specific rebalancing/augmentation so val/test are always the
    same 100%-real rows regardless of condition -- this is what makes F1 numbers
    comparable across conditions and a paired significance test valid."""
    y = df[LABELS].values.astype(np.float32)
    toxic = (y.sum(axis=1) > 0).astype(int)
    idx = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=CONFIG['test_size'], random_state=CONFIG['seed'], stratify=toxic)
    toxic_tmp = toxic[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=CONFIG['val_frac_of_temp'], random_state=CONFIG['seed'],
        stratify=toxic_tmp)
    train_df = df.iloc[idx_tr].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    log(f"Fixed real-only split: train={len(train_df):,}  val={len(val_df):,}  "
        f"test={len(test_df):,} (identical across every condition)")
    return train_df, val_df, test_df


def apply_condition_to_train(train_df, condition, data_dir):
    """All condition-specific rebalancing/augmentation happens on the TRAIN split ONLY.
    val/test are never touched here -- they're passed separately and stay 100% real."""
    if condition == 'original':
        return train_df

    if condition == 'undersampled':
        is_toxic = (train_df[LABELS].sum(axis=1) > 0)
        toxic_df = train_df[is_toxic]
        non_toxic_df = train_df[~is_toxic]
        keep_n = int(len(non_toxic_df) * CONFIG['non_toxic_keep_ratio'])
        non_toxic_sampled = non_toxic_df.sample(n=keep_n, random_state=CONFIG['seed'])
        out = pd.concat([toxic_df, non_toxic_sampled], ignore_index=True)
        log(f"Undersampled (train only): {len(toxic_df):,} toxic + "
            f"{len(non_toxic_sampled):,}/{len(non_toxic_df):,} non-toxic kept = {len(out):,} total")
        return out

    if condition == 'augmented':
        df_aug = load_augmented_df(data_dir)
        out = pd.concat([train_df, df_aug], ignore_index=True)
        log(f"Augmented (train only): {len(train_df):,} real + {len(df_aug):,} synthetic "
            f"= {len(out):,} total")
        return out

    if condition == 'naive_duplicate':
        # Clone real TRAIN minority rows (no LLM, no interpolation) to match the augmented
        # condition's per-class synthetic counts exactly. Isolates "more positive examples
        # of this class" from "LLM-authored diverse content" -- same total per-class boost
        # as 'augmented', but via verbatim duplication of real text instead of generation.
        df_aug = load_augmented_df(data_dir)
        dup_parts = []
        for lab in MINORITY_LABELS:
            n_needed = int(df_aug[lab].sum())
            pool = train_df[train_df[lab] == 1]
            if n_needed == 0 or len(pool) == 0:
                continue
            sampled = pool.sample(n=n_needed, replace=True, random_state=CONFIG['seed']).copy()
            # Mask other minority labels on the duplicate so it contributes to exactly one
            # class's added count, same as a synthetic row would (real rows are naturally
            # multi-labeled; without this, duplicating a severe_toxic row that's also
            # insult=1 silently inflates insult's count too -- caught by local validation).
            for other in MINORITY_LABELS:
                if other != lab:
                    sampled[other] = 0
            dup_parts.append(sampled)
            log(f"  naive_duplicate[{lab}]: cloned {n_needed:,} real train rows "
                f"(pool size {len(pool):,}, sampled with replacement, other minority labels masked)")
        dup_df = pd.concat(dup_parts, ignore_index=True)[['comment_text'] + LABELS]
        out = pd.concat([train_df, dup_df], ignore_index=True)
        log(f"Naive-duplicate (train only): {len(train_df):,} real + {len(dup_df):,} "
            f"duplicated-real = {len(out):,} total")
        return out

    if condition == 'smote':
        # Negative-undersample pre-step on the TRAIN split only. The MLSMOTE positive-interpolation
        # happens post-embedding, also train-only.
        is_toxic = (train_df[LABELS].sum(axis=1) > 0)
        toxic_df = train_df[is_toxic]
        non_toxic_df = train_df[~is_toxic]
        n_neg_target = min(len(non_toxic_df), len(toxic_df) * CONFIG['smote_neg_multiplier'])
        non_toxic_sampled = non_toxic_df.sample(n=n_neg_target, random_state=CONFIG['seed'])
        out = pd.concat([toxic_df, non_toxic_sampled], ignore_index=True)
        log(f"SMOTE pre-step (train only, 1:{CONFIG['smote_neg_multiplier']} undersample): "
            f"{len(toxic_df):,} toxic + {len(non_toxic_sampled):,} non-toxic = {len(out):,} total")
        return out

    raise ValueError(condition)


def ml_smote(X, y, k_neighbors=5, seed=42):
    """Multi-label SMOTE on an embedding matrix: interpolate between each minority
    (any-label-positive) sample and one of its k nearest minority neighbors, carrying
    the union of both samples' labels to the synthetic point. Oversamples positives to 1:1
    with negatives."""
    rng = np.random.RandomState(seed)
    pos_mask = (y.sum(axis=1) > 0)
    X_pos, y_pos = X[pos_mask], y[pos_mask]
    n_neg = (~pos_mask).sum()
    n_needed = max(0, n_neg - len(X_pos))
    if n_needed == 0 or len(X_pos) < 2:
        return X, y

    k = min(k_neighbors, len(X_pos) - 1)
    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(X_pos)
    _, neighbor_idx = nn_model.kneighbors(X_pos)

    synth_X, synth_y = [], []
    for _ in range(n_needed):
        i = rng.randint(0, len(X_pos))
        j = neighbor_idx[i][rng.randint(1, k + 1)]  # skip self at index 0
        lam = rng.uniform(0, 1)
        synth_X.append(X_pos[i] + lam * (X_pos[j] - X_pos[i]))
        synth_y.append(np.maximum(y_pos[i], y_pos[j]))  # union of labels

    X_out = np.vstack([X, np.array(synth_X)])
    y_out = np.vstack([y, np.array(synth_y)])
    log(f"MLSMOTE: +{n_needed:,} synthetic positives -> {X_out.shape[0]:,} total "
        f"({(y_out.sum(axis=1) > 0).sum():,} positive / {(y_out.sum(axis=1) == 0).sum():,} negative)")
    return X_out, y_out


# =====================================================================
# EMBEDDING EXTRACTORS
# =====================================================================
def encode_with_oom_retry(encode_fn, batch, device, max_gpu_retries=5, wait_s=45):
    for attempt in range(max_gpu_retries):
        try:
            return encode_fn(batch, device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            log(f"  GPU OOM (attempt {attempt+1}/{max_gpu_retries}). "
                f"Clearing cache, waiting {wait_s}s for other jobs to free memory ...")
            torch.cuda.empty_cache(); gc.collect(); time.sleep(wait_s)
    log("  GPU still OOM after retries -- falling back to CPU for this batch (slow).")
    return encode_fn(batch, 'cpu')


def embed_qwen(texts, device, assets_dir):
    from sentence_transformers import SentenceTransformer
    log("Loading Qwen/Qwen3-Embedding-0.6B ...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device,
                                 cache_folder=os.path.join(assets_dir, 'hf_cache'))
    dim = model.get_sentence_embedding_dimension()
    log(f"Qwen embedding dim: {dim}")

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


def embed_fasttext(texts, device, assets_dir):
    from gensim.models import KeyedVectors
    vec_path = os.path.join(assets_dir, 'crawl-300d-2M.vec')
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"{vec_path} not found. Run:\n"
            f"  wget -P {assets_dir} https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip\n"
            f"  unzip {assets_dir}/crawl-300d-2M.vec.zip -d {assets_dir}"
        )
    log(f"Loading top {CONFIG['fasttext_top_n']:,} FastText vectors ...")
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False, limit=CONFIG['fasttext_top_n'])
    dim = kv.vector_size
    embs = np.zeros((len(texts), dim), dtype=np.float32)
    hits = 0
    for i, t in enumerate(tqdm(texts, desc="FastText mean-pool")):
        words = str(t).lower().split()
        vecs = [kv[w] for w in words if w in kv]
        if vecs:
            embs[i] = np.mean(vecs, axis=0)
            hits += 1
    log(f"FastText coverage: {hits}/{len(texts)} comments had >=1 known word")
    return embs, dim


EMBEDDERS = {'qwen': embed_qwen, 'deberta': embed_deberta, 'fasttext': embed_fasttext}


# =====================================================================
# FOCAL LOSS + THRESHOLD OPTIMIZATION
# =====================================================================
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


def optimize_thresholds(y_true, y_prob, search_range, steps):
    thresholds = np.linspace(*search_range, steps)
    best_t, best_f1 = [], []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_prob[:, c]
        scores = [f1_score(yt, (yp >= t).astype(int), average='binary', zero_division=0) for t in thresholds]
        k = int(np.argmax(scores))
        best_t.append(float(thresholds[k])); best_f1.append(float(scores[k]))
    return best_t, best_f1


def apply_thresholds(y_prob, thresholds):
    return np.stack([(y_prob[:, i] >= t).astype(int) for i, t in enumerate(thresholds)], axis=1)


# =====================================================================
# MODELS
# =====================================================================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, len(LABELS)))

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, len(LABELS))

    def forward(self, x):
        e = self.embedding(x)
        _, (h, _) = self.lstm(e)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h))


class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SeqDataset(Dataset):
    def __init__(self, seqs, y):
        self.seqs, self.y = torch.LongTensor(seqs), torch.FloatTensor(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.seqs[i], self.y[i]


def build_lstm_vocab(train_texts):
    """Vocabulary built from TRAIN text only. val/test words not seen in train map to <UNK>."""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for t in train_texts:
        for w in str(t).lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def texts_to_seqs(texts, vocab, max_len):
    seqs = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = str(t).lower().split()[:max_len]
        ids = [vocab.get(w, 1) for w in toks]
        seqs[i, :len(ids)] = ids
    return seqs


def evaluate_report(y_true, y_prob, thresholds, name):
    y_pred = apply_thresholds(y_prob, thresholds)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    per_class = classification_report(y_true, y_pred, target_names=LABELS, digits=4,
                                       zero_division=0, output_dict=True)
    log(f"{name}: macro F1={macro:.5f}  micro F1={micro:.5f}  "
        f"macro P={prec_macro:.5f}  macro R={rec_macro:.5f}")
    return {'macro_f1': macro, 'micro_f1': micro, 'macro_precision': prec_macro,
            'macro_recall': rec_macro, 'per_class': per_class, 'thresholds': thresholds}


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding', required=True, choices=['qwen', 'deberta', 'fasttext'])
    ap.add_argument('--condition', required=True,
                     choices=['original', 'undersampled', 'smote', 'augmented', 'naive_duplicate'])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--assets-dir', required=True)
    ap.add_argument('--n-jobs', type=int, default=int(os.environ.get('NCPUS', 4)))
    args = ap.parse_args()

    seed_everything(CONFIG['seed'])
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.assets_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = f"{args.embedding}_{args.condition}"
    log(f"=== SHIELD run v2: {run_id} | device={device} ===")

    df = load_base_df(args.data_dir)
    real_train_df, val_df, test_df = split_real_data(df)
    train_df = apply_condition_to_train(real_train_df, args.condition, args.data_dir)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    texts = full_df['comment_text'].tolist()
    y = full_df[LABELS].values.astype(np.float32)
    n_tr, n_val, n_test = len(train_df), len(val_df), len(test_df)

    X, emb_dim = EMBEDDERS[args.embedding](texts, device, args.assets_dir)
    X_tr, X_val, X_test = X[:n_tr], X[n_tr:n_tr + n_val], X[n_tr + n_val:]
    y_tr, y_val, y_test = y[:n_tr], y[n_tr:n_tr + n_val], y[n_tr + n_val:]

    vocab = build_lstm_vocab(train_df['comment_text'].tolist())
    seq_tr = texts_to_seqs(train_df['comment_text'].tolist(), vocab, CONFIG['max_seq_length'])
    seq_val = texts_to_seqs(val_df['comment_text'].tolist(), vocab, CONFIG['max_seq_length'])
    seq_test = texts_to_seqs(test_df['comment_text'].tolist(), vocab, CONFIG['max_seq_length'])
    vocab_size = len(vocab)
    log(f"Split: train={n_tr:,}  val={n_val:,}  test={n_test:,}  "
        f"(val/test are the SAME fixed real rows regardless of condition)")

    if args.condition == 'smote':
        n_before = X_tr.shape[0]
        X_tr, y_tr = ml_smote(X_tr, y_tr, seed=CONFIG['seed'])
        pad_rows = np.zeros((X_tr.shape[0] - n_before, CONFIG['max_seq_length']), dtype=np.int64)
        seq_tr = np.vstack([seq_tr, pad_rows])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    focal_alpha = torch.FloatTensor(CONFIG['focal_alpha']).to(device)
    train_dl = DataLoader(VecDataset(X_tr, y_tr), batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(VecDataset(X_val, y_val), batch_size=CONFIG['batch_size'], shuffle=False)
    test_dl = DataLoader(VecDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
    lstm_train_dl = DataLoader(SeqDataset(seq_tr, y_tr), batch_size=CONFIG['batch_size'], shuffle=True)
    lstm_val_dl = DataLoader(SeqDataset(seq_val, y_val), batch_size=CONFIG['batch_size'], shuffle=False)
    lstm_test_dl = DataLoader(SeqDataset(seq_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)

    results = {'run_id': run_id, 'embedding': args.embedding, 'condition': args.condition,
               'embedding_dim': emb_dim, 'n_train': int(n_tr), 'n_val': int(n_val),
               'n_test': int(n_test), 'config': CONFIG, 'pipeline_version': 'v2_fixed_split'}
    test_predictions = {'y_test': y_test, 'labels': np.array(LABELS)}

    # ---------------- MLP ----------------
    log("Training MLP ...")
    mlp = MLP(emb_dim).to(device)
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
        log(f"  MLP epoch {epoch+1:02d}  val macro F1={f1:.5f} (best {best_f1:.5f})")
        if patience >= 5:
            log("  MLP early stop"); break
    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        mlp_val_p = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in val_dl])
        mlp_test_p = np.vstack([torch.sigmoid(mlp(xb.to(device))).cpu().numpy() for xb, _ in test_dl])
    mlp_thr, _ = optimize_thresholds(y_val, mlp_val_p, CONFIG['threshold_search_range'],
                                      CONFIG['threshold_search_steps'])
    results['mlp'] = evaluate_report(y_test, mlp_test_p, mlp_thr, 'MLP')
    test_predictions['mlp_prob'] = mlp_test_p
    test_predictions['mlp_thr'] = np.array(mlp_thr)
    del mlp; gc.collect(); torch.cuda.empty_cache()

    # ---------------- LSTM ----------------
    log("Training LSTM ...")
    lstm = LSTMClassifier(vocab_size, 256, CONFIG['lstm_hidden'], CONFIG['lstm_layers'],
                           CONFIG['lstm_dropout']).to(device)
    opt = optim.AdamW(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = FocalLoss(alpha=focal_alpha, gamma=CONFIG['focal_gamma'])
    best_f1, best_state, patience = 0, None, 0
    for epoch in range(CONFIG['lstm_epochs']):
        lstm.train()
        for sb, yb in lstm_train_dl:
            sb, yb = sb.to(device), yb.to(device)
            loss = crit(lstm(sb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()
        lstm.eval()
        with torch.no_grad():
            vp = np.vstack([torch.sigmoid(lstm(sb.to(device))).cpu().numpy() for sb, _ in lstm_val_dl])
        f1 = f1_score(y_val, (vp > 0.5).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_state, patience = f1, {k: v.clone() for k, v in lstm.state_dict().items()}, 0
        else:
            patience += 1
        log(f"  LSTM epoch {epoch+1:02d}  val macro F1={f1:.5f} (best {best_f1:.5f})")
        if patience >= 3:
            log("  LSTM early stop"); break
    lstm.load_state_dict(best_state)
    lstm.eval()
    with torch.no_grad():
        lstm_val_p = np.vstack([torch.sigmoid(lstm(sb.to(device))).cpu().numpy() for sb, _ in lstm_val_dl])
        lstm_test_p = np.vstack([torch.sigmoid(lstm(sb.to(device))).cpu().numpy() for sb, _ in lstm_test_dl])
    lstm_thr, _ = optimize_thresholds(y_val, lstm_val_p, CONFIG['threshold_search_range'],
                                       CONFIG['threshold_search_steps'])
    results['lstm'] = evaluate_report(y_test, lstm_test_p, lstm_thr, 'LSTM')
    test_predictions['lstm_prob'] = lstm_test_p
    test_predictions['lstm_thr'] = np.array(lstm_thr)
    del lstm; gc.collect(); torch.cuda.empty_cache()

    # ---------------- Random Forest (on the SAME embeddings) ----------------
    log("Training Random Forest on embeddings ...")
    rf = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=500, max_depth=20, n_jobs=1, random_state=CONFIG['seed'],
        class_weight='balanced'), n_jobs=args.n_jobs)
    rf.fit(X_tr, y_tr)
    rf_val_p = np.stack([p[:, 1] for p in rf.predict_proba(X_val)], axis=1)
    rf_test_p = np.stack([p[:, 1] for p in rf.predict_proba(X_test)], axis=1)
    rf_thr, _ = optimize_thresholds(y_val, rf_val_p, CONFIG['threshold_search_range'],
                                     CONFIG['threshold_search_steps'])
    results['random_forest'] = evaluate_report(y_test, rf_test_p, rf_thr, 'RandomForest')
    test_predictions['rf_prob'] = rf_test_p
    test_predictions['rf_thr'] = np.array(rf_thr)
    del rf; gc.collect()

    # ---------------- LightGBM (per-label) — run as a SEPARATE PROCESS ----------------
    log("Training LightGBM (per-label) via subprocess worker ...")
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, 'lgbm_data.npz')
        np.savez(npz_path, X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test, labels=np.array(LABELS),
                 focal_alpha=np.array(CONFIG['focal_alpha']),
                 threshold_search_range=np.array(CONFIG['threshold_search_range']),
                 threshold_search_steps=CONFIG['threshold_search_steps'])
        out_json = os.path.join(tmpdir, 'lgbm_result.json')
        out_probs = os.path.join(tmpdir, 'lgbm_test_probs.npy')
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lgbm_worker.py')
        proc = subprocess.run(
            [sys.executable, '-u', worker, '--data-npz', npz_path, '--out-json', out_json,
             '--out-test-probs', out_probs, '--n-jobs', str(args.n_jobs)],
            capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            log(line)
        if proc.returncode != 0 or not os.path.exists(out_json):
            log(f"LightGBM worker FAILED (exit {proc.returncode}). stderr tail:\n" +
                "\n".join(proc.stderr.splitlines()[-30:]))
            raise RuntimeError("lgbm_worker subprocess failed")
        with open(out_json) as f:
            results['lightgbm'] = json.load(f)
        log(f"LightGBM: macro F1={results['lightgbm']['macro_f1']:.5f} "
            f"micro F1={results['lightgbm']['micro_f1']:.5f}")
        if os.path.exists(out_probs):
            test_predictions['lgbm_prob'] = np.load(out_probs)
            test_predictions['lgbm_thr'] = np.array(results['lightgbm']['thresholds'])

    out_path = os.path.join(args.out_dir, f"{run_id}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    pred_path = os.path.join(args.out_dir, f"{run_id}_test_predictions.npz")
    np.savez_compressed(pred_path, **test_predictions)
    log(f"=== DONE. Results written to {out_path} ===")
    log(f"=== Test predictions (for significance testing) written to {pred_path} ===")
    log("SUMMARY (macro F1): " + ", ".join(
        f"{k}={results[k]['macro_f1']:.4f}" for k in ['random_forest', 'lightgbm', 'mlp', 'lstm']))


if __name__ == '__main__':
    main()
