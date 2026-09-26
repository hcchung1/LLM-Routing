# CARROT Mean-Row-Max Soft-Oracle Runbook

## Purpose

This experiment trains one 11-class Qwen3-8B router using the competition-aligned reward:

```text
reward = 0.85 * performance
       - 0.15 * cost / mean(per-query maximum cost)
```

For each training query:

```text
soft_target = softmax(reward / 0.05)
row_weight = clip(best_reward - second_best_reward, 0, 1)
loss = weighted KLDiv(soft_target, model probabilities)
```

The classifier directly predicts the selected model. It does not use or train a separate cost head.

This run is independent from the saved CARROT artifacts that produced the `0.47480` submission. Do not overwrite:

```text
output_G/carrot_runs/qwen3-8b/
output_G/carrot_scores.npz
output_G/carrot_meanrowmax_rescore/
```

## 1. Environment Check

Use the existing container Python:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

Confirm the required packages and GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
python -c "import transformers, peft, bitsandbytes; print(transformers.__version__, peft.__version__, bitsandbytes.__version__)"
```

The `qwen3-8b-l4` profile requires one CUDA GPU with BF16 support.

## 2. Lightweight Validation

Run before starting the GPU job:

```bash
python -m py_compile carrot_router.py tests/test_carrot_soft_oracle.py
python -m unittest discover -s tests -p 'test_carrot_soft_oracle.py' -v
python carrot_router.py train-soft-oracle --help
python carrot_router.py predict-soft-oracle --help
```

Expected unit-test result:

```text
Ran 8 tests
OK
```

## 3. Train the Soft-Oracle Classifier

This is a long-running QLoRA training command. Run it intentionally on an L4 GPU, preferably in `tmux`:

```bash
tmux new -s carrot-soft-oracle
cd /workspace/Junior/LLM-Routing
```

Start training:

```bash
python carrot_router.py train-soft-oracle \
  --profile qwen3-8b-l4 \
  --train dataset/train.csv \
  --output-dir output_G/carrot_runs/qwen3-8b-soft-oracle-t005 \
  --epochs 5 \
  --soft-label-temperature 0.05 \
  --cost-denominator mean-row-max \
  --val-size 0.1 \
  --seed 42 \
  --audit-out output_G/carrot_runs/qwen3-8b-soft-oracle-t005/validation_audit.npz
```

Default training settings:

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen3-8B-Base` |
| Max length | `512` |
| Train batch size | `1` |
| Evaluation batch size | `2` |
| Gradient accumulation | `16` |
| Learning rate | `3e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Seed | `42` |

Expected output structure:

```text
output_G/carrot_runs/qwen3-8b-soft-oracle-t005/
├── classifier/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── soft_oracle_metadata.json
├── validation_audit.npz
└── validation_audit.json
```

The expected training-set denominator is approximately:

```text
0.077205403389806
```

## 4. Inspect the Validation Audit

```bash
cat output_G/carrot_runs/qwen3-8b-soft-oracle-t005/validation_audit.json
```

Review at least:

```text
routing_reward
oracle_reward
regret
top1_accuracy
cost_denominator
selection_counts
oracle_counts
row_weight_quantiles
reward_gap_quantiles
margin_quantiles
```

Do not rely on accuracy alone. The primary local metric is `routing_reward`.

Treat the run as suspicious if:

- nearly all queries route to one model;
- `Model_H` selection is far above its Oracle count;
- routing reward is driven by a degenerate selection distribution;
- prediction margins are almost all near zero.

Keep the adapter and audit even when the run is suspicious. Do not apply calibration before understanding the selection distribution.

## 5. Generate the Submission

This is a full Qwen3-8B GPU inference command:

```bash
python carrot_router.py predict-soft-oracle \
  --profile qwen3-8b-l4 \
  --test dataset/test.csv \
  --adapter-dir output_G/carrot_runs/qwen3-8b-soft-oracle-t005 \
  --out output_G/carrot_soft_oracle_t005/submission_carrot_soft_oracle_t005.csv \
  --score-out output_G/carrot_soft_oracle_t005/scores_carrot_soft_oracle_t005.npz \
  --prediction-out output_G/carrot_soft_oracle_t005/predictions_carrot_soft_oracle_t005.csv \
  --seed 42
```

The score file contains:

```text
logits
probabilities
ids
model_names
```

The debug CSV contains each model's logit and probability for every test query.

## 6. Validate the Submission

```bash
python - <<'PY'
import pandas as pd

sample = pd.read_csv("dataset/sample_submission.csv")
submission = pd.read_csv(
    "output_G/carrot_soft_oracle_t005/"
    "submission_carrot_soft_oracle_t005.csv"
)

assert submission.shape == (2550, 2)
assert submission.columns.tolist() == ["ID", "pred_model"]
assert submission["ID"].equals(sample["ID"])
assert submission["ID"].is_unique
assert not submission.isna().any().any()
assert submission["pred_model"].str.fullmatch(r"Model_[A-K]").all()

print(submission["pred_model"].value_counts().sort_index())
PY
```

Compare its model counts against:

```text
output_G/carrot_runs/qwen3-8b-soft-oracle-t005/validation_audit.json
```

Large train-validation versus test selection shifts should be recorded before uploading.

## 7. Kaggle Comparison

Use the following established result as the current reference:

```text
output_G/carrot_meanrowmax_rescore/
submission_carrot_meanrowmax_exact_w0758758.csv
Public Score: 0.47480
```

Upload the uncalibrated soft-oracle submission first. Record its public score, model counts, command, and artifact paths before considering temperature changes, calibration, blending, or additional seeds.
