# Language-AI

This repo trains two stylometry classifiers (TF-IDF + Logistic Regression and fastText) and evaluates how text obfuscations affect performance. The main entrypoint is `draft_script_clean.py`.

## What the script does

- Load a dataset from `.xlsx` or `.csv` and normalizes columns.
- Split by author to prevent leakage (default 80/10/10 train/dev/test).
- Train two baselines (Logistic Regression and fastText).
- Evaluate on clean test data.
- Optionally evaluate on obfuscated test data and write a single `results.json`.

## Data format

Expected columns (case-insensitive):
- Author: `author_id` (or any column that contains "author")
- Text: `text`, `post`, or `body`
- Label: `female`, `label`, or `y` (binary 0/1)

The zip includes `gender.xlsx` as a sample dataset.

## Obfuscations

These are registered in `draft_script_clean.py`:
- `none`: no changes (identity transform).
- `lexical_remove`: removes explicit gender cues (`obfs/obfuscation_lexical.py`).
- `char_visual`: VIPER-style character perturbations (`obfs/obfuscation_char_visual.py`).
- `high_weight_sub`: substitutes high-weight LR features using synonyms or masked LM candidates (`obfs/obfuscation_high_weight.py`).
- `chained_light`: `lexical_remove` -> `char_visual` -> `high_weight_sub`.

Notes:
- `high_weight_sub` is only available after the LR model is trained (the script handles this).
- If you want to avoid transformers, use `--substitute_mode synonym_only` so only WordNet synonyms are used.

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```



## Usage

0 obfuscations (baseline only). The script always computes a clean baseline; `none` just records the identity transform:
```bash
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations none
```

One obfuscation at a time:
```bash
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations lexical_remove
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations char_visual
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations high_weight_sub
```

All three chained:
```bash
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations chained_light
```


## Output

- Results are written to `results.json` inside `--out_dir`.
- The JSON includes baseline metrics and per-obfuscation metrics for both models.

## Common options

- `--data`: path to dataset
- `--out_dir`: output directory (default `results_si`)
- `--obfuscations`: comma-separated list or `all`
- `--seed`: random seed
- `--test_size`, `--dev_size`: author-level split fractions
- `--viper_p`, `--viper_ces`, `--viper_k`, `--viper_font_path`: VIPER settings
- `--substitute_mode`: use `synonym_only` to disable masked LM candidates; other values currently behave like the default mix
