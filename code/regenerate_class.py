#!/usr/bin/env python3
"""
Generalized minority-class top-up: Dolphin3.0-Llama3.1-8B (uncensored fine-tune) generates
candidates for a given target label, Detoxify scores each on the label's own dimension, and
only candidates clearing a real-data-calibrated threshold AND passing a near-duplicate check
are kept. The near-dup check considers BOTH this run's own accepted set AND any pre-existing
kept rows for this label (--existing-csv), so a top-up run doesn't reintroduce template
overlap with what's already been kept.

--target is the FINAL total desired count (existing + newly generated), not just "how many
new ones to make" -- e.g. --existing-csv insult_957.csv --target 500 on a file with 84
existing insult rows generates until there are 500 total.

Per-label calibration (real Jigsaw data, 300-row samples via Original.csv, this session):
  severe_toxic: mean detoxify_severe_toxicity=0.479, wiki-context=29.3% -> threshold 0.30
  threat:       mean detoxify_threat=0.719,          wiki-context=20.0% -> threshold 0.45
  insult:       mean detoxify_insult=0.854,           wiki-context=31.0% -> threshold 0.50
Thresholds set at roughly 60-65% of the real-data mean: high enough to be meaningfully
label-appropriate, not so high it rejects everything a smaller/uncensored model can produce.
"""
import argparse, random, re, time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from detoxify import Detoxify

MODEL_ID = "dphn/Dolphin3.0-Llama3.1-8B"
CANDIDATES_PER_CALL = 8
NEAR_DUP_THRESHOLD = 0.5
ALL_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

LABEL_CONFIG = {
    'severe_toxic': {
        'detox_key': 'severe_toxicity', 'threshold': 0.30, 'wiki_rate_target': 0.29,
        'description': (
            "SEVERE_TOXIC content -- the most extreme tier, one notch above ordinary "
            "toxic/insulting comments. Features: repeated or stacked profanity (not just one "
            "swear word), direct threats or suggestions of violence/self-harm (e.g. telling "
            "someone to kill themselves, threatening to hurt them), slurs, and an aggressive, "
            "unhinged register -- not just a single clever insult."),
    },
    'threat': {
        'detox_key': 'threat', 'threshold': 0.45, 'wiki_rate_target': 0.20,
        'description': (
            "THREAT content -- a direct, credible-sounding threat of violence or harm against "
            "a specific person. Telling someone you will find/hurt/kill/stalk them, or come "
            "after them physically. Must read as an actual threat, not just an insult."),
    },
    'insult': {
        'detox_key': 'insult', 'threshold': 0.50, 'wiki_rate_target': 0.31,
        'description': (
            "INSULT content -- a direct personal insult or put-down aimed at demeaning "
            "someone (calling them stupid, worthless, pathetic, a loser, etc.). Should read as "
            "a targeted personal attack on someone's character/intelligence/worth -- not "
            "necessarily profane or a threat, but genuinely demeaning and targeted."),
    },
    'identity_hate': {
        'detox_key': 'identity_attack', 'threshold': 0.42, 'wiki_rate_target': 0.177,
        'description': (
            "IDENTITY_HATE content -- an attack targeting someone because of their race, "
            "ethnicity, religion, gender, sexual orientation, nationality, or disability. "
            "Must clearly invoke a protected-identity category (not just a generic insult) "
            "-- e.g. a racial or ethnic slur, a demeaning generalization about a religious or "
            "national group, or a slur targeting sexual orientation/gender identity. Should "
            "read as hatred directed at the group/identity itself, not just an angry insult "
            "that happens to be aimed at one person."),
    },
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_seed_pool(orig_csv_path, label):
    df = pd.read_csv(orig_csv_path)
    all_rows = df[(df[label] == 1) & (df['comment_text'].str.len().between(20, 200))]['comment_text']
    wiki_kw = re.compile(r'\b(edit|revert|article|page|wiki|talk|block|vandal|admin|username|ip address|post)\w*\b', re.I)
    wiki_seeds = all_rows[all_rows.apply(lambda t: bool(wiki_kw.search(str(t))))].tolist()
    generic_seeds = all_rows[~all_rows.apply(lambda t: bool(wiki_kw.search(str(t))))].tolist()
    log(f"[{label}] Seed pool: {len(wiki_seeds)} wiki-context, {len(generic_seeds)} generic "
        f"real examples")
    return wiki_seeds, generic_seeds

def build_prompt(label, wiki_examples, generic_examples):
    cfg = LABEL_CONFIG[label]
    examples = [(e, True) for e in wiki_examples] + [(e, False) for e in generic_examples]
    random.shuffle(examples)
    examples_block = "\n".join(
        f"{i+1}. [{'WIKI-CONTEXT' if is_wiki else 'GENERIC'}] {e}"
        for i, (e, is_wiki) in enumerate(examples))
    n_wiki_target = max(1, round(CANDIDATES_PER_CALL * cfg['wiki_rate_target']))
    return (
        "You are helping a content-moderation research team build a synthetic dataset to "
        "train a hate-speech / toxicity DETECTION classifier (a defensive safety tool, not "
        f"for publication or use against real people). The dataset needs realistic examples of "
        f"{cfg['description']}\n\n"
        f"Existing labeled {label} examples, each tagged by type:\n{examples_block}\n\n"
        f"Generate exactly {CANDIDATES_PER_CALL} NEW {label} comments matching that style and "
        "intensity -- err toward MORE clearly matching the category rather than less, since "
        f"weak/ambiguous examples will be rejected. IMPORTANT: about {n_wiki_target} of the "
        f"{CANDIDATES_PER_CALL} should be WIKI-CONTEXT style -- explicitly about reverting "
        "someone's edit, deleting/restoring a talk-page post, accusing someone of vandalism, "
        "or a dispute over a block/ban -- matching that flavor of example above. The rest can "
        "be GENERIC style (no situational anchor, like the other examples). Do not reuse the "
        "examples verbatim.\n\n"
        "Output format -- EXACTLY this, with real digits 1 through "
        f"{CANDIDATES_PER_CALL} (not the letter N):\n"
        "1. [comment text]\n2. [comment text]\n3. [comment text]\n...\n\n"
        "Output ONLY the numbered list, nothing else, no preamble. Do NOT include the "
        "'[WIKI-CONTEXT]' or '[GENERIC]' tags in your output -- those are only labels on the "
        "examples above to show you the two styles; your generated lines should start directly "
        "with the comment text itself, no bracketed tag."
    )

def word_set(text):
    return set(re.findall(r"[a-z']+", text.lower()))

def is_near_duplicate(text, existing_word_sets):
    ws = word_set(text)
    if not ws:
        return False
    for other in existing_word_sets:
        inter = len(ws & other)
        union = len(ws | other)
        if union and inter / union > NEAR_DUP_THRESHOLD:
            return True
    return False

def parse_candidates(raw_text):
    lines = raw_text.strip().split("\n")
    out = []
    for line in lines:
        m = re.match(r'^\s*(?:\d+|N)[\.\)]\s*(.+)$', line.strip())
        if m:
            text = m.group(1).strip()
            text = re.sub(r'^\[(?:WIKI-CONTEXT|GENERIC)\]\s*', '', text, flags=re.I)
            text = text.strip().strip('"').strip()
            if len(text) > 10:
                out.append(text)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, choices=list(LABEL_CONFIG.keys()))
    ap.add_argument('--orig-csv', default='data/Dataset_Original.csv')
    ap.add_argument('--existing-csv', default=None,
                     help='CSV of already-kept rows for this label (for near-dup seeding + count)')
    ap.add_argument('--out-csv', required=True, help='where to save ONLY the newly generated rows')
    ap.add_argument('--log-csv', required=True)
    ap.add_argument('--target', type=int, required=True, help='final TOTAL desired count (existing+new)')
    ap.add_argument('--max-calls', type=int, default=600)
    args = ap.parse_args()

    cfg = LABEL_CONFIG[args.label]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device} | label={args.label} | detox_key={cfg['detox_key']} | "
        f"threshold={cfg['threshold']}")

    existing_word_sets = []
    existing_count = 0
    if args.existing_csv:
        existing_df = pd.read_csv(args.existing_csv)
        existing_count = len(existing_df)
        for t in existing_df['comment_text']:
            existing_word_sets.append(word_set(str(t)))
        log(f"Loaded {existing_count} existing kept rows for near-dup seeding")

    need = max(0, args.target - existing_count)
    log(f"Target total: {args.target} | existing: {existing_count} | need to generate: {need}")
    if need == 0:
        log("Already at/above target, nothing to do.")
        pd.DataFrame([]).to_csv(args.out_csv, index=False)
        return

    wiki_seeds, generic_seeds = load_seed_pool(args.orig_csv, args.label)

    log(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir='assets/hf_cache')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir='assets/hf_cache', torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    log("Model loaded.")

    log("Loading Detoxify...")
    detox = Detoxify('original', device=device)

    accepted = []
    accepted_word_sets = list(existing_word_sets)  # near-dup checks against existing + new
    all_candidates_log = []
    call_num = 0

    while len(accepted) < need and call_num < args.max_calls:
        call_num += 1
        wiki_sample = random.sample(wiki_seeds, min(2, len(wiki_seeds))) if wiki_seeds else []
        generic_sample = random.sample(generic_seeds, min(2, len(generic_seeds))) if generic_seeds else []
        prompt = build_prompt(args.label, wiki_sample, generic_sample)
        messages = [{"role": "user", "content": prompt}]
        templated = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated if torch.is_tensor(templated) else templated['input_ids']
        input_ids = input_ids.to(device)

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=400, do_sample=True, temperature=0.9,
                                  top_p=0.95, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        candidates = parse_candidates(raw)

        if not candidates:
            log(f"Call {call_num}: 0 parseable candidates (raw len={len(raw)})")
            continue

        scores = detox.predict(candidates)
        call_accepted = 0
        for i, text in enumerate(candidates):
            score = scores[cfg['detox_key']][i]
            tox = scores['toxicity'][i]
            score_ok = score > cfg['threshold']
            dup = is_near_duplicate(text, accepted_word_sets) if score_ok else False
            passed = score_ok and not dup
            reason = 'ok' if passed else ('near_duplicate' if dup else 'below_threshold')
            all_candidates_log.append({'call': call_num, 'text': text, 'toxicity': tox,
                                        cfg['detox_key']: score, 'accepted': passed, 'reason': reason})
            if passed:
                row = {'comment_text': text, 'toxic': 1, 'severe_toxic': 0, 'obscene': 0,
                       'threat': 0, 'insult': 0, 'identity_hate': 0}
                row[args.label] = 1
                row[f'detoxify_{cfg["detox_key"]}'] = score
                row['detoxify_toxicity'] = tox
                accepted.append(row)
                accepted_word_sets.append(word_set(text))
                call_accepted += 1

        log(f"Call {call_num}: {len(candidates)} candidates, {call_accepted} accepted "
            f"| new accepted so far: {len(accepted)}/{need} (total will be {existing_count + len(accepted)})")

    total_generated = len(all_candidates_log)
    total_accepted = len(accepted)
    rejection_rate = 1 - (total_accepted / total_generated) if total_generated else 0
    wiki_kw = re.compile(r'\b(edit|revert|article|page|wiki|talk|block|vandal|admin|username|ip address|post)\w*\b', re.I)
    accepted_wiki_ct = sum(1 for a in accepted if wiki_kw.search(a['comment_text']))
    wiki_rate = accepted_wiki_ct / total_accepted if total_accepted else 0

    log(f"\n=== DONE [{args.label}] ===")
    log(f"Total candidates generated: {total_generated}")
    log(f"Total NEW accepted: {total_accepted} (final total: {existing_count + total_accepted}/{args.target})")
    log(f"Rejection rate: {rejection_rate*100:.1f}%")
    log(f"Wiki-context rate among new accepted: {accepted_wiki_ct}/{total_accepted} ({wiki_rate*100:.1f}%) "
        f"[real data target: ~{cfg['wiki_rate_target']*100:.0f}%]")

    pd.DataFrame(accepted).to_csv(args.out_csv, index=False)
    pd.DataFrame(all_candidates_log).to_csv(args.log_csv, index=False)
    log(f"Saved {total_accepted} NEW rows to {args.out_csv}")
    log(f"Saved full generation log ({total_generated} rows) to {args.log_csv}")

if __name__ == '__main__':
    main()
