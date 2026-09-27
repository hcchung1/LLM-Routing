# LLM-Routing

A Kaggle/homework project on cost-aware LLM routing using the `Reward_{0.85}` metric.

```
Reward_0.85 = 0.85 * mean(performance) - 0.15 * mean(cost) / global_max_cost
```

The final routing decision is cost-aware by design: candidate models are scored
with `0.85 * predicted_performance - 0.15 * normalized_cost`, where
`normalized_cost = cost / global_max_cost`.

## Project report summary

The project evaluates two CARROT routing strategies on top of Qwen3 base models:

- **Independent CARROT router**: trains separate performance and cost heads, then
  combines their predictions with the competition reward formula. Performance
  training uses binary cross-entropy plus a pairwise ranking loss; cost training
  uses regression.
- **Soft-Oracle classifier**: converts each row's candidate rewards into a
  regret-weighted softmax target and trains one classifier with a weighted KL
  divergence loss.

Both approaches use regret-aware row weights. Rows with a large gap between the
best and second-best candidate receive more weight because routing mistakes on
those rows have a larger effect on the final reward.

The strongest reported approach was **CARROT mean-row-max rescore**, which keeps
performance and cost prediction separate and applies the reward formula only at
decision time. In the report's comparison table it achieved a public score of
`0.47480`; the report body also mentions `0.47668`, so the exact leaderboard
submission should be verified before treating either value as canonical.

### Reported experiment comparison

| Rank | Method | Public score |
|---:|---|---:|
| 1 | CARROT mean-row-max rescore | 0.47480 |
| 2 | CARROT soft-oracle classifier | 0.46504 |
| 3 | CARROT soft-oracle classifier | 0.46210 |
| 4 | CARROT pairwise routing variant | 0.46103 |
| 5 | CARROT pairwise routing variant | 0.46052 |
| 6 | CARROT soft-oracle classifier | 0.45625 |
| 7 | CARROT soft-oracle classifier | 0.45489 |

### Training configuration from the report

| Category | Configuration |
|---|---|
| LoRA | `r=16`, `lora_alpha=32`, dropout `0.05` |
| LoRA targets | `q`, `k`, `v`, `o`, `gate`, `up`, `down_proj` |
| Optimization | Learning rate `3e-5`, linear scheduler, `paged_adamw_8bit` |
| Regularization | Warmup ratio `0.1`, weight decay `0.01` |
| Input/training | Batch size `1`, maximum length `512` |
| Ranking loss | Weight `1.0`, temperature `0.05`, minimum gap `1e-4` |

### Post-training calibration

- **Coordinate-search bias calibration** searches per-model logit offsets from
  coarse to fine (`(step, radius, passes) = (0.10, 1.5, 3), (0.05, 0.5, 3),
  (0.02, 0.2, 2)`) and applies a shrink factor of `0.7`.
- **KNN residual correction** embeds queries with `intfloat/e5-base-v2`, finds
  the 50 nearest training queries, and blends their residuals into the predicted
  model scores.

## Project structure

```
main.py                      # CARROT router — train, predict, calibrate (primary entry point)
Test_version/                # Experimental routers (TF-IDF/KNN, GBM, reward-blend, etc.)
dataset/                     # train.csv and test.csv (not committed)
references/                  # Paper summaries and notes
```

## CARROT router (`main.py`)

`main.py` implements the CARROT (Cost-Aware Rate-Optimal Router) method with QLoRA fine-tuning on top of Qwen3 base models.  It has six sub-commands.

### Hardware profiles

| Profile | Model | GPU |
|---|---|---|
| `qwen3-4b-1080ti` | Qwen/Qwen3-4B-Base | NVIDIA 1080 Ti (FP16) |
| `qwen3-8b-l4` | Qwen/Qwen3-8B-Base | NVIDIA L4 (BF16) |

List profiles:

```bash
python main.py list-profiles
```

Override any profile field on the command line with `--model-name`, `--max-length`, `--batch-size`, `--learning-rate`, etc.

---

### 1. `train` — train performance and cost heads

```bash
python main.py train \
  --train dataset/train.csv \
  --profile qwen3-4b-1080ti \
  --output-dir carrot_runs/qwen3-4b-1080ti \
  --epochs 5 \
  --head both          # "both" | "performance" | "cost"
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--head` | `both` | Which head(s) to train |
| `--epochs` | `5.0` | Training epochs |
| `--val-size` | `0.1` | Fraction of train data used for validation |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--ranking-loss-weight` | `1.0` | Weight of the pairwise ranking loss |
| `--ranking-temperature` | `0.05` | Softplus temperature for ranking loss |
| `--row-weight-floor` | `0.05` | Minimum per-row loss weight |
| `--audit-out` | auto | Path to save a validation audit `.npz` |
| `--resume-from-checkpoint` | — | Resume from a saved checkpoint |

Outputs saved to `--output-dir`:
- `metadata.json` — training metadata
- `performance/` — performance-head adapter
- `cost/` — cost-head adapter
- `validation_audit.npz` + `validation_audit.json` — per-row predictions on the held-out validation split

---

### 2. `train-soft-oracle` — train a soft-label classifier

Instead of separate performance/cost heads, this trains a single classifier whose labels are softmax-smoothed routing probabilities weighted by the regret gap.

```bash
python main.py train-soft-oracle \
  --train dataset/train.csv \
  --profile qwen3-4b-1080ti \
  --output-dir carrot_runs/qwen3-4b-soft-oracle \
  --soft-label-temperature 0.05 \
  --cost-denominator mean-row-max   # "mean-row-max" | "global-max"
```

Outputs saved to `--output-dir`:
- `soft_oracle_metadata.json`
- `classifier/` — single QLoRA adapter
- `validation_audit.npz` + `validation_audit.json`

---

### 3. `predict` — route test queries with CARROT

```bash
python main.py predict \
  --test dataset/test.csv \
  --adapter-dir carrot_runs/qwen3-4b-1080ti \
  --out submission_carrot.csv
```

Optional flags:

| Flag | Description |
|---|---|
| `--cost-weight` | Override cost weight (default: value saved in metadata, 0.15) |
| `--calibration` | Path to a `carrot_calibration.json` produced by `calibrate` |
| `--knn-correction` | Path to a KNN residual `.npz` produced by `calibrate --knn-out` |
| `--knn-weight` | Blend weight for KNN correction (0–1) |
| `--score-out` | Save raw scores to an `.npz` file |
| `--prediction-out` | Save per-model predicted performance, cost, and score to CSV |

---

### 4. `predict-soft-oracle` — route with the soft-oracle classifier

```bash
python main.py predict-soft-oracle \
  --test dataset/test.csv \
  --adapter-dir carrot_runs/qwen3-4b-soft-oracle \
  --out submission_carrot_soft_oracle.csv
```

Optional flags: `--score-out`, `--prediction-out` (saves logits and per-model probabilities).

---

### 5. `calibrate` — fit per-model score biases on a validation audit

```bash
python main.py calibrate \
  --audit carrot_runs/qwen3-4b-1080ti/validation_audit.npz \
  --out carrot_calibration.json
```

Optional: also fit a KNN residual correction:

```bash
python main.py calibrate \
  --audit carrot_runs/qwen3-4b-1080ti/validation_audit.npz \
  --out carrot_calibration.json \
  --knn-out carrot_knn.npz \
  --embedding-model intfloat/e5-base-v2 \
  --knn-k 50
```

Then pass `--calibration carrot_calibration.json --knn-correction carrot_knn.npz` to `predict`.

---

### 6. `calibrate-soft-oracle-bias` — calibrate soft-oracle logit biases

Requires `--enable` to run (disabled by default to prevent accidental overfitting).

```bash
python main.py calibrate-soft-oracle-bias \
  --audit  carrot_runs/qwen3-4b-soft-oracle/validation_audit.npz \
  --test-score soft_oracle_scores.npz \
  --sample dataset/sample_submission.csv \
  --out-json soft_oracle_bias.json \
  --out-csv  submission_soft_oracle_bias_cal.csv \
  --enable
```

---

## Dataset format

The training CSV must have:
- A text column (`query`, `prompt`, `question`, `Question`, or `text` — auto-detected, or set with `--text-column`)
- Column pairs `<Model>_performance` and `<Model>_cost` for each candidate model

The test CSV must have an `ID` column. Performance/cost columns are optional; if present, the script prints the routing reward.

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is required for training and inference. Use `--profile qwen3-4b-1080ti` for FP16 (1080 Ti / older GPUs) and `--profile qwen3-8b-l4` for BF16 (A10 / L4 and newer).

Optional: set `HF_TOKEN` (or pass `--hf-token`) if the base model requires Hugging Face authentication.