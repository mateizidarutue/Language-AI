# Language-AI

This repo trains and evaluates two stylometry classifiers (TF-IDF + Logistic Regression and fastText) and measures how much three light obfuscations degrade performance. The main runner is `draft_script_clean.py`, and the sample dataset is `gender.xlsx` in the repo root.

## Pipeline Overview

1) Data loading and normalization
- Reads an `.xlsx` or `.csv` file via `--data`.
- Finds the author column (`author_id` or similar), the text column (`text`/`post`/`body`), and the label column (`female`/`label`/`y`).
- Coerces labels to integers, drops missing labels, and filters to binary labels {0, 1}.
- Casts all text to strings and fills missing text with empty strings.

2) Author-level split
- Splits by author to prevent leakage so an author’s posts never appear in more than one split.
- Default split is 80/10/10 (train/dev/test) with stratification on the author label.

3) Baseline training
- Trains a TF-IDF + Logistic Regression model on the training split.
- Trains a fastText supervised model on the training split.

4) Baseline evaluation
- Evaluates both models on the clean test set.
- Stores accuracy, macro-F1, and per-class reports in the results structure.

5) Obfuscation registry
The script builds a registry of obfuscation functions, each taking a list of texts and returning a list of obfuscated texts. The three light perturbations are:
- `lexical_remove`: removes lexical cue words via `obfs/obfuscation_lexical.py`.
- `char_visual`: character/visual perturbations using VIPER via `obfs/obfuscation_char_visual.py` with per-text deterministic seeding.
- `high_weight_sub`: substitutes high-weight features using the trained LR model and TF-IDF vectorizer via `obfs/obfuscation_high_weight.py`. This is only registered after training because it depends on the trained model.
- `chained_light`: applies `lexical_remove` → `char_visual` → `high_weight_sub` in sequence. This is a stronger, combined version of the three light perturbations.

6) Obfuscation evaluation
- For each requested obfuscation name, the script:
  - Obfuscates the test texts.
  - Evaluates both models on the obfuscated texts.
  - Stores metrics under `results["obfuscations"][name]`.

7) Results output
- Writes a single JSON file to `--out_dir` (default `results_si/results.json`).
- Includes baseline metrics, per-obfuscation metrics, and metadata about the run.

## Data

The repo includes `gender.xlsx`, which is the default dataset you can point to with `--data`. The script accepts any dataset with the required columns as described above.

## Usage

Run all available obfuscations:
```bash
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations all
```

Run only the chained light perturbation:
```bash
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations chained_light
```

Run a custom subset:
```bash
python draft_script_clean.py --data gender.xlsx --out_dir results_si --obfuscations lexical_remove,char_visual,high_weight_sub
```
