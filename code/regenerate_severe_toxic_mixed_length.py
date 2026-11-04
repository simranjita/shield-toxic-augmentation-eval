#!/usr/bin/env python3
"""
PRODUCTION mixed-length regeneration for severe_toxic, replacing the entire class.

Mixed-length generation: 75% natural mode plus 25% long-tail mode (min_new_tokens=200), so the length
distribution matches the real class (median 17 words, long tail p90=139, p95=468, p99=979). Degenerate
repetition in long-tail outputs is caught by a unique-word-ratio filter (reject <0.35).

This FULLY REPLACES severe_toxic in Augmented.csv (not a top-up).
"""
import argparse, random, re, time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from detoxify import Detoxify

MODEL_ID = "dphn/Dolphin3.0-Llama3.1-8B"
CANDIDATES_PER_CALL = 8
NEAR_DUP_THRESHOLD = 0.5
LONG_TAIL_RATE = 0.25
LONG_TAIL_MIN_WORDS = 100
MIN_UNIQUE_WORD_RATIO = 0.35  # rejects degenerate "X X X X..." repetition loops
LABEL_CFG = {'detox_key': 'severe_toxicity', 'threshold': 0.30}
DESCRIPTION = (
    "SEVERE_TOXIC content -- the most extreme tier, one notch above ordinary "
    "toxic/insulting comments. Features: repeated or stacked profanity (not just one "
    "swear word), direct threats or suggestions of violence/self-harm (e.g. telling "
    "someone to kill themselves, threatening to hurt them), slurs, and an aggressive, "
    "unhinged register -- not just a single clever insult.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pools(orig_csv_path, label):
    df = pd.read_csv(orig_csv_path)
    rows = df[df[label] == 1]['comment_text'].astype(str)
    wc = rows.apply(lambda t: len(t.split()))
    general_pool = rows[(rows.str.len().between(20, 600))].tolist()
    long_tail_pool = rows[wc >= LONG_TAIL_MIN_WORDS].tolist()
    log(f"[{label}] general pool: {len(general_pool)}  long-tail pool (>={LONG_TAIL_MIN_WORDS}w): "
        f"{len(long_tail_pool)}")
    return general_pool, long_tail_pool


def word_set(text):
    return set(re.findall(r"[a-z']+", text.lower()))


def unique_word_ratio(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0


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


def parse_list_candidates(raw_text):
    lines = raw_text.strip().split("\n")
    out = []
    for line in lines:
        m = re.match(r'^\s*(?:\d+|N)[\.\)]\s*(.+)$', line.strip())
        if m:
            text = m.group(1).strip().strip('"').strip()
            if len(text) > 10:
                out.append(text)
    return out


def build_natural_prompt(examples):
    examples_block = "\n".join(f"{i+1}. {e}" for i, e in enumerate(examples))
    return (
        "You are helping a content-moderation research team build a synthetic dataset to "
        "train a hate-speech / toxicity DETECTION classifier (a defensive safety tool, not "
        f"for publication or use against real people). The dataset needs realistic examples of "
        f"{DESCRIPTION}\n\n"
        f"Existing labeled examples:\n{examples_block}\n\n"
        f"Generate exactly {CANDIDATES_PER_CALL} NEW comments matching that style and "
        "intensity -- err toward MORE clearly matching the category rather than less. Vary "
        "length naturally -- some short and blunt, some a bit longer -- like the examples "
        "above. Do not reuse the examples verbatim.\n\n"
        "Output format -- EXACTLY this, with real digits 1 through "
        f"{CANDIDATES_PER_CALL} (not the letter N):\n"
        "1. [comment text]\n2. [comment text]\n3. [comment text]\n...\n\n"
        "Output ONLY the numbered list, nothing else, no preamble."
    )


def build_long_tail_prompt(long_examples):
    examples_block = "\n\n".join(f"EXAMPLE {i+1}:\n{e}" for i, e in enumerate(long_examples))
    return (
        "You are helping a content-moderation research team build a synthetic dataset to "
        "train a hate-speech / toxicity DETECTION classifier (a defensive safety tool, not "
        f"for publication or use against real people). {DESCRIPTION}\n\n"
        "Below are REAL examples of long, rambling, repetitive angry rants from this exact "
        "category -- study their structure closely: they repeat the same accusations in "
        "different words, go off on tangents, pile up multiple grievances, and generally "
        f"do NOT stay focused or brief:\n\n{examples_block}\n\n"
        "Write ONE new comment in this exact rambling, repetitive, long-winded style -- "
        "matching the LENGTH and RANGE OF GRIEVANCES of the examples above (at least "
        "150 words, ideally longer), not a short punchy insult. Ramble across MULTIPLE "
        "DIFFERENT angry points, tangents and insults -- do NOT just repeat one sentence "
        "over and over, that is boring and unrealistic; real rants pile up many different "
        "complaints and insults, not one looped phrase. Do not reuse the examples verbatim.\n\n"
        "Output ONLY the comment text, nothing else, no preamble, no quotation marks."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='severe_toxic')
    ap.add_argument('--orig-csv', default='data/Original.csv')
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--log-csv', required=True)
    ap.add_argument('--target', type=int, default=500)
    ap.add_argument('--max-calls', type=int, default=400)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"PRODUCTION mixed-length regen: device={device} | label={args.label} | "
        f"threshold={LABEL_CFG['threshold']} | long_tail_rate={LONG_TAIL_RATE} | "
        f"min_unique_ratio={MIN_UNIQUE_WORD_RATIO}")

    general_pool, long_tail_pool = load_pools(args.orig_csv, args.label)
    target_long = round(args.target * LONG_TAIL_RATE)
    target_natural = args.target - target_long
    log(f"Target: {target_natural} natural + {target_long} long-tail = {args.target} total")

    log(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir='assets/hf_cache')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir='assets/hf_cache', torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    log("Model loaded.")

    log("Loading Detoxify...")
    detox = Detoxify('original', device=device)

    accepted = []
    accepted_word_sets = []
    all_candidates_log = []
    call_num = 0

    # ---- Phase 1: natural-length batch generation ----
    log("\n--- Phase 1: natural-length generation ---")
    while len([a for a in accepted if a['mode'] == 'natural']) < target_natural and call_num < args.max_calls:
        call_num += 1
        examples = random.sample(general_pool, min(4, len(general_pool)))
        prompt = build_natural_prompt(examples)
        messages = [{"role": "user", "content": prompt}]
        templated = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated if torch.is_tensor(templated) else templated['input_ids']
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=400, do_sample=True,
                                  temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        candidates = parse_list_candidates(raw)
        call_accepted = 0
        for text in candidates:
            wc = len(text.split())
            preds = detox.predict([text])
            score = preds[LABEL_CFG['detox_key']][0]
            tox = preds['toxicity'][0]
            score_ok = score > LABEL_CFG['threshold']
            dup = is_near_duplicate(text, accepted_word_sets) if score_ok else False
            passed = score_ok and not dup
            all_candidates_log.append({'mode': 'natural', 'call': call_num, 'text': text,
                                        'word_count': wc, 'score': score, 'accepted': passed})
            if passed:
                accepted.append({'comment_text': text, 'word_count': wc, 'mode': 'natural',
                                  f'detoxify_{LABEL_CFG["detox_key"]}': score,
                                  'detoxify_toxicity': tox})
                accepted_word_sets.append(word_set(text))
                call_accepted += 1
        if call_num % 10 == 0 or call_accepted:
            log(f"Call {call_num} [natural]: {len(candidates)} cand, {call_accepted} accepted | "
                f"natural so far: {len([a for a in accepted if a['mode']=='natural'])}/{target_natural}")

    # ---- Phase 2: long-tail single-comment generation, min_new_tokens forced ----
    log("\n--- Phase 2: long-tail generation ---")
    while len([a for a in accepted if a['mode'] == 'long_tail']) < target_long and call_num < args.max_calls * 4:
        call_num += 1
        long_examples = random.sample(long_tail_pool, min(2, len(long_tail_pool)))
        prompt = build_long_tail_prompt(long_examples)
        messages = [{"role": "user", "content": prompt}]
        templated = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated if torch.is_tensor(templated) else templated['input_ids']
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=500, min_new_tokens=200,
                                  do_sample=True, temperature=1.0, top_p=0.95,
                                  pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().strip('"').strip()
        if len(raw) < 20:
            continue
        wc = len(raw.split())
        uwr = unique_word_ratio(raw)
        preds = detox.predict([raw])
        score = preds[LABEL_CFG['detox_key']][0]
        tox = preds['toxicity'][0]
        score_ok = score > LABEL_CFG['threshold']
        diverse_ok = uwr >= MIN_UNIQUE_WORD_RATIO
        dup = is_near_duplicate(raw, accepted_word_sets) if (score_ok and diverse_ok) else False
        passed = score_ok and diverse_ok and not dup
        reason = 'ok' if passed else ('near_duplicate' if dup else (
            'degenerate_repetition' if not diverse_ok else 'below_threshold'))
        all_candidates_log.append({'mode': 'long_tail', 'call': call_num, 'text': raw,
                                    'word_count': wc, 'unique_word_ratio': uwr,
                                    'score': score, 'accepted': passed, 'reason': reason})
        if passed:
            accepted.append({'comment_text': raw, 'word_count': wc, 'mode': 'long_tail',
                              f'detoxify_{LABEL_CFG["detox_key"]}': score,
                              'detoxify_toxicity': tox})
            accepted_word_sets.append(word_set(raw))
        if call_num % 10 == 0 or passed:
            log(f"Call {call_num} [long_tail]: wc={wc} uwr={uwr:.2f} score={score:.3f} "
                f"reason={reason} | long_tail so far: "
                f"{len([a for a in accepted if a['mode']=='long_tail'])}/{target_long}")

    total_generated = len(all_candidates_log)
    total_accepted = len(accepted)
    log(f"\n=== DONE [{args.label}] mixed-length regeneration ===")
    log(f"Total candidates: {total_generated}  |  Total accepted: {total_accepted}")
    for mode in ['natural', 'long_tail']:
        wc_mode = [a['word_count'] for a in accepted if a['mode'] == mode]
        if wc_mode:
            log(f"  {mode}: n={len(wc_mode)}  mean_wc={sum(wc_mode)/len(wc_mode):.1f}  "
                f"min={min(wc_mode)}  max={max(wc_mode)}")
    long_tail_cands = [c for c in all_candidates_log if c['mode'] == 'long_tail']
    degenerate_ct = sum(1 for c in long_tail_cands if c.get('reason') == 'degenerate_repetition')
    if long_tail_cands:
        log(f"  long_tail degenerate-repetition rejections: {degenerate_ct}/{len(long_tail_cands)} "
            f"({100*degenerate_ct/len(long_tail_cands):.1f}%)")

    out_df = pd.DataFrame(accepted)
    for lab in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        out_df[lab] = 1 if lab == args.label else 0
    out_df['toxic'] = 1
    out_df.to_csv(args.out_csv, index=False)
    pd.DataFrame(all_candidates_log).to_csv(args.log_csv, index=False)
    log(f"Saved {total_accepted} rows to {args.out_csv}")


if __name__ == '__main__':
    main()
