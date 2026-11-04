# Data card: LLM-generated minority-class comments

**Content warning.** `data/Augmented.csv` and `data/Augmented_gen2.csv` contain machine-written
toxic text, including threats, insults, identity-based hate and severe toxicity. The rows are
intended for research on moderation systems only.

## Files
| File | Rows | md5 |
|---|---|---|
| `data/Augmented.csv` (8B generator, used for the main results) | 1,923 | 379079c243b46fd793533e26e855365e |
| `data/Augmented_gen2.csv` (24B generator for the threat and severe toxic rows; other rows identical) | 1,923 | 7ad620130ea7ab8936b45f67cc55ba2f |

Columns: `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` (0/1).
No real Jigsaw comment is included. The real data comes from the Jigsaw Toxic Comment
Classification Challenge (Kaggle) and must be downloaded separately.

## Contents
Per-class counts: 477 severe toxic, 500 threat, 501 insult, 499 identity hate. They add up to 1,977
because 54 rows carry both insult and identity hate. Every row is labeled for exactly the minority class
or classes it was generated for.

## Generation
- Generator 1: Dolphin3.0-Llama3.1-8B. Generator 2 (threat and severe toxic only):
  Dolphin-Mistral-24B-Venice-Edition, 4-bit.
- Each call showed four real labeled comments of the target class and asked for eight new ones.
- Filters: Detoxify score above a per-class threshold, Jaccard word overlap above 0.5 against an accepted
  row, unique-word ratio below 0.35, and a final near-duplicate pass across classes.
- Rejection rates by class: 62% to 88% of candidates.
- Generator 1 few-shot examples came from all real comments of a class. Generator 2 used training rows only
  (`code/make_trainonly.py` writes the file).

## Known properties
- The rows are shorter than real comments. Median words for threat, insult and identity hate are 13, 11 and 12
  against 20 to 23 for real comments (generator 2 threat median: 17).
- The rows are easy to separate from real comments with word and character n-grams (AUC 0.98 to 1.00;
  `analysis/cue_check.py`).
- A blinded single-annotator pilot of 240 comments found 70.6% of synthetic rows matched their intended class.
  This is not an agreement statistic.

## Intended use and limits
Research on the evaluation of synthetic training data. The paper finds no reliable accuracy gain from these
rows under a leakage-safe protocol. Do not use them to build tools that generate or amplify abuse.

## Licenses and terms
The code is MIT. The rows were generated with Llama 3.1-derived and Mistral-derived models; the license terms of
those models (for example the Llama 3.1 Community License) may apply to the outputs. Check the upstream terms
before redistributing. Terms of the Jigsaw data apply to the real comments.
