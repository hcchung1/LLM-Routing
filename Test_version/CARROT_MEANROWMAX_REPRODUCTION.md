# CARROT Mean-Row-Max Submission Reproduction Record

## Result

- Submission file: `output_G/carrot_meanrowmax_rescore/submission_carrot_meanrowmax_exact_w0758758.csv`
- Kaggle Public Score: `0.47480`
- Uploaded on: `2026-06-11`
- Rows: `2550`
- Submission columns: `ID,pred_model`

This record documents the path from the original Qwen3-8B CARROT training artifacts to the submitted mean-row-max rescored file.

## Competition Metric

The competition description defines:

```text
Reward_0.85 = 0.85 * mean(performance)
            - 0.15 * mean(cost) / mean(per-query maximum cost)
```

The local source is:

```text
Homework Specifications/Competition Description.md
```

For `dataset/train.csv`:

```text
mean_row_max_cost = mean(max(cost across 11 models for each row))
                  = 0.077205403389806
```

The original CARROT metadata instead stored:

```text
global_max_cost = max(all training costs)
                = 1.3760199546813965
```

These denominators differ by approximately `17.82285x`.

## Original Training Artifacts

The successful submission uses the original adapters under:

```text
output_G/carrot_runs/qwen3-8b/
├── metadata.json
├── performance/
│   └── adapter_model.safetensors
└── cost/
    └── adapter_model.safetensors
```

Model configuration:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-8B-Base` |
| Profile | `qwen3-8b-l4` |
| Text column | `query` |
| Maximum sequence length | `512` |
| Quantization | 4-bit NF4 QLoRA |
| Compute dtype | BF16 |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Seed | `42` |

The saved `training_args.bin` files confirm the following settings for both the performance and cost heads:

| Setting | Performance head | Cost head |
|---|---:|---:|
| Epochs | `5` | `5` |
| Learning rate | `3e-5` | `3e-5` |
| Weight decay | `0.01` | `0.01` |
| Warmup ratio | `0.1` | `0.1` |
| Train batch size | `1` | `1` |
| Evaluation batch size | `2` | `2` |
| Gradient accumulation | `16` | `16` |
| Optimizer | `paged_adamw_8bit` | `paged_adamw_8bit` |
| Scheduler | `linear` | `linear` |
| Best checkpoint metric | `eval_loss` | `eval_loss` |

The historical training command is reconstructed as:

```bash
python carrot_router.py train \
  --profile qwen3-8b-l4 \
  --head both \
  --output-dir carrot_runs/qwen3-8b \
  --epochs 5
```

The artifacts were later stored under `output_G/carrot_runs/qwen3-8b/`.

### Retraining Limitation

The current `carrot_router.py` is not the exact script revision used to train these original adapters. It now uses a routing-aware performance loss, while the saved original run selected both heads by `eval_loss` and its metadata has no routing-loss fields.

Therefore:

- prediction from the saved adapters is reproducible;
- rescoring `output_G/carrot_scores.npz` is exactly reproducible;
- exact retraining requires the historical pre-routing-aware `carrot_router.py` revision;
- running the reconstructed training command with the current script will not reproduce the same adapters.

Do not replace or delete the saved `output_G/carrot_runs/qwen3-8b/` artifacts.

## Original Prediction

The original adapters produced:

```text
output_G/submission_carrot_8b.csv
output_G/carrot_scores.npz
```

The corresponding prediction command was:

```bash
python carrot_router.py predict \
  --profile qwen3-8b-l4 \
  --adapter-dir output_G/carrot_runs/qwen3-8b \
  --out output_G/submission_carrot_8b.csv \
  --score-out output_G/carrot_scores.npz
```

`output_G/carrot_scores.npz` contains:

```text
scores          (2550, 11) float32
performance     (2550, 11) float32
predicted_cost  (2550, 11) float32
ids             (2550,)    int64
model_names     (11,)       string
```

The original prediction used:

```text
score_old = 0.85 * predicted_performance
          - 0.15 * predicted_cost / 1.3760199546813965
```

Its model counts were:

| Model | Count |
|---|---:|
| Model_A | 42 |
| Model_B | 98 |
| Model_C | 9 |
| Model_D | 35 |
| Model_E | 7 |
| Model_F | 256 |
| Model_G | 29 |
| Model_H | 1345 |
| Model_I | 137 |
| Model_J | 10 |
| Model_K | 582 |

## Mean-Row-Max Rescoring

The desired ranking is:

```text
score_target = 0.85 * predicted_performance
             - 0.15 * predicted_cost / 0.077205403389806
```

`carrot_router.py` expresses scores as:

```text
score_router = (1 - w) * predicted_performance
             - w * predicted_cost / old_denominator
```

Multiplying every model score for a query by the same positive constant does not change `argmax`. The equivalent weight is obtained from:

```text
w / (1 - w)
    = (0.15 / 0.85) * (old_denominator / mean_row_max_cost)
    = 3.145207979810809

w = 0.758757581074221
```

The filename rounds this to `w0758758`, while the generation used the full floating-point calculation.

## Exact CPU-Only Reproduction

This recreates the submitted CSV from the saved prediction matrix without loading Qwen, using a GPU, or retraining:

```bash
python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

train = pd.read_csv("dataset/train.csv")
cost_columns = [column for column in train.columns if column.endswith("_cost")]
train_cost = train[cost_columns].to_numpy(dtype=np.float64)

old_denominator = 1.3760199546813965
mean_row_max_cost = float(train_cost.max(axis=1).mean())
relative_cost_weight = (
    (0.15 / 0.85) * (old_denominator / mean_row_max_cost)
)
cost_weight = relative_cost_weight / (1.0 + relative_cost_weight)

data = np.load("output_G/carrot_scores.npz", allow_pickle=True)
performance = np.asarray(data["performance"], dtype=np.float64)
predicted_cost = np.asarray(data["predicted_cost"], dtype=np.float64)
model_names = np.asarray(data["model_names"], dtype=object)
ids = np.asarray(data["ids"])

scores = (
    (1.0 - cost_weight) * performance
    - cost_weight * predicted_cost / old_denominator
)
predictions = model_names[scores.argmax(axis=1)]

output = Path(
    "output_G/carrot_meanrowmax_rescore/"
    "submission_carrot_meanrowmax_exact_w0758758.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "ID": ids,
    "pred_model": predictions,
}).to_csv(output, index=False)

print(f"mean_row_max_cost={mean_row_max_cost:.15f}")
print(f"cost_weight={cost_weight:.15f}")
print(f"wrote={output}")
PY
```

Expected values:

```text
mean_row_max_cost=0.077205403389806
cost_weight=0.758757581074221
```

Expected model counts:

| Model | Count |
|---|---:|
| Model_A | 14 |
| Model_B | 109 |
| Model_C | 34 |
| Model_D | 181 |
| Model_E | 8 |
| Model_F | 133 |
| Model_G | 36 |
| Model_H | 787 |
| Model_I | 308 |
| Model_J | 27 |
| Model_K | 913 |

This changes `830` of the `2550` predictions relative to `output_G/submission_carrot_8b.csv`.

## Direct Prediction Alternative

If the original adapters, required Python packages, Qwen base model, and an L4-class GPU are available, the same decision rule can be applied directly during prediction:

```bash
python carrot_router.py predict \
  --profile qwen3-8b-l4 \
  --adapter-dir output_G/carrot_runs/qwen3-8b \
  --cost-weight 0.758757581074221 \
  --out output_G/carrot_meanrowmax_rescore/submission_carrot_meanrowmax_exact_w0758758.csv \
  --score-out output_G/carrot_meanrowmax_rescore/carrot_meanrowmax_scores.npz
```

The CPU-only reproduction above is preferred when `output_G/carrot_scores.npz` is already available because it avoids model inference variability and cost.

## Validation

Run:

```bash
python - <<'PY'
import hashlib

import pandas as pd

path = (
    "output_G/carrot_meanrowmax_rescore/"
    "submission_carrot_meanrowmax_exact_w0758758.csv"
)
submission = pd.read_csv(path)

assert submission.shape == (2550, 2)
assert submission.columns.tolist() == ["ID", "pred_model"]
assert submission["ID"].is_unique
assert not submission.isna().any().any()
assert submission["pred_model"].str.fullmatch(r"Model_[A-K]").all()

digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(submission["pred_model"].value_counts().sort_index())
print("sha256:", digest)
PY
```

Expected submission SHA-256:

```text
4f998fa27aad15e97ec1a37f775fc534f0154dd5e5364b4815b9fa796751f46d
```

## Artifact Checksums

```text
a118ef4696c90233e94b4fb2ec746362f9009d9fe98bfb5cc5983af6099b5486  output_G/carrot_runs/qwen3-8b/metadata.json
f19e86108aa0adf88d3a1aebff368157e28d4e57be0c73f5f140d04826d74ca1  output_G/carrot_runs/qwen3-8b/performance/adapter_model.safetensors
07e24560129a4937bf2d1e8bf0219bba440c1dc6e90ba30e563e3b4db8623493  output_G/carrot_runs/qwen3-8b/cost/adapter_model.safetensors
1420385f27fa0252b5a65f18e9d8823a2921e90d544161f0b259413585c71ae5  output_G/carrot_scores.npz
7364a64ad620b28fb893ecd48016e1398df1ce2d20bdbd7a05d8ba6ce5edf06e  output_G/submission_carrot_8b.csv
4f998fa27aad15e97ec1a37f775fc534f0154dd5e5364b4815b9fa796751f46d  output_G/carrot_meanrowmax_rescore/submission_carrot_meanrowmax_exact_w0758758.csv
```

## Public Leaderboard Record

| Submission | Public Score |
|---|---:|
| `output_G/submission_carrot_8b.csv` | `0.45705` |
| `output_G/carrot_conservative_pairwise_filtered/submission_carrot_pairwise_KtoFH_top70.csv` | `0.46114` |
| `output_G/carrot_meanrowmax_rescore/submission_carrot_meanrowmax_exact_w0758758.csv` | **`0.47480`** |

The `0.47480` result confirms that aligning inference-time cost normalization with the competition's mean-per-query-maximum denominator was the decisive improvement for this saved CARROT model.
