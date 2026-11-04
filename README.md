# SHIELD: evaluating LLM-generated toxic comments

Code, synthetic rows and result files for "Pooling Before Splitting Inflates Gains From Synthetic Toxic Comments"
(Khadolia and Singh, submitted to IEEE Intelligent Systems). The data is the Jigsaw Toxic Comment
Classification Challenge (Kaggle); the real comments are not redistributed here.

**Content warning:** `data/` contains machine-written toxic text. See `DATA_CARD.md`.

## Layout
- `code/`: pipeline, multi-seed, BERT, pooled-split ablation, generation and filtering scripts.
- `data/`: `Augmented.csv` (8B generator) and `Augmented_gen2.csv` (24B generator). Put the Kaggle training file at `data/Original.csv`
  (columns `comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`).
- `analysis/`: bootstrap tests and the detectability check (`cue_check.py`, run as `python3 analysis/cue_check.py .` from the repo root).
- `results/`: result JSONs and bootstrap logs behind the tables. Prediction arrays and model weights are not included.

## Protocol
The 159,571 real comments are split once (seed 42, 85/7.5/7.5, stratified on any label) into 135,635 / 11,968 /
11,968 comments. Every condition (original, undersampled, SMOTE, augmented, naive duplicate) is built from the
training split only, and all share one real test set. BERT uses its own 70/15/15 split.

## Running
All scripts expect a data directory holding `Original.csv` and `Augmented.csv`; adjust the paths to your system.
```
pip install -r requirements.txt
# main grid, one embedding x one condition (also trains RF, LightGBM and LSTM and writes prediction .npz files)
python3 code/shield_pipeline.py --embedding qwen --condition augmented --data-dir data --out-dir out --assets-dir assets
# five-seed MLP runs (default seeds are 42 123 456; the paper uses five)
python3 code/multiseed_mlp.py --embedding qwen --condition augmented --data-dir data --assets-dir assets --cache-dir cache --out-dir out --seeds 42 123 456 789 1011
python3 code/multiseed_conditions.py --embedding qwen --cache-file cache/qwen_augmented_cached.npz --out-dir out --grid-dir out
# pooled-split ablation and its threshold control
python3 code/p1_pooled_split_ablation.py --cache-file cache/qwen_augmented_cached.npz --out out/p1_qwen.json --seeds 42 123 456 789 1011
python3 code/p1_realthr.py --cache-file cache/qwen_augmented_cached.npz --out out/p1b_qwen.json
# fine-tuned BERT
python3 code/bert_baseline_seed.py --condition augmented --data-dir data --out-dir out --assets-dir assets --model-seed 42
```
BERT uses a class-weighted loss with a computed `pos_weight` (clamped at 50). The near-duplicate pass across classes lives in
`merge_identity_hate.py` and `merge_gen2.py`.

### Regenerating the synthetic rows (needs a GPU and the generator weights)
```
python3 code/make_trainonly.py --in data/Original.csv --out data/Original_trainonly.csv   # few-shot pool for generator 2
python3 code/regenerate_class_gen2.py --label threat --orig-csv data/Original_trainonly.csv --out-csv out/threat.csv --log-csv out/threat_log.csv --target 500
```
Several scripts keep their original default paths (`data/Dataset_Original.csv`, `results/...`); pass explicit paths as above.

## Reproducibility notes
- GPU training is not bit-deterministic: reruns with the same seeds differ by up to 0.025 macro F1 per seed.
- No API keys or credentials are used or stored. Model weights download from Hugging Face on first use.

## License
Code: MIT (`LICENSE`). Data terms: `DATA_CARD.md`.
