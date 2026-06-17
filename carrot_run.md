# CARROT Routing-Aware Run Order

Use the existing container Python. Run training on an L4 GPU.

## 1. Train the routing-aware performance head

This is the important new training run. It uses:

- multi-label performance BCE;
- reward-weighted pairwise ranking;
- lower weight for rows with little model-to-model reward information;
- validation `Reward_0.85` for best-checkpoint selection;
- a validation audit saved as `validation_audit.npz` and `validation_audit.json`.

```bash
python3 carrot_router.py train \
  --profile qwen3-8b-l4 \
  --head performance \
  --output-dir carrot_runs/qwen3-8b-routing \
  --epochs 5 \
  --ranking-loss-weight 1.0 \
  --ranking-temperature 0.05 \
  --ranking-min-gap 0.0001 \
  --row-weight-floor 0.05
```

## 2. Train the cost head

The cost head is unchanged. You may either train a new one:

```bash
python3 carrot_router.py train \
  --profile qwen3-8b-l4 \
  --head cost \
  --output-dir carrot_runs/qwen3-8b-routing \
  --epochs 5
```

Or reuse the existing cost adapter:

```bash
cp -r carrot_runs/qwen3-8b/cost carrot_runs/qwen3-8b-routing/
```

When copying from another machine, also verify that this file exists:

```bash
ls carrot_runs/qwen3-8b-routing/cost/adapter_model.safetensors
```

## 3. Inspect the validation audit

```bash
cat carrot_runs/qwen3-8b-routing/validation_audit.json
```

Compare:

- `routing_reward`
- `oracle_reward`
- `selection_counts`
- `oracle_counts`
- `margin_quantiles`

Do not proceed with a run that still selects almost only `Model_H`.

## 4. Fit conservative score-bias calibration

This uses only the held-out validation split. Biases are shrunk by 50% to reduce overfitting.

```bash
python3 carrot_router.py calibrate \
  --audit carrot_runs/qwen3-8b-routing/validation_audit.npz \
  --out carrot_runs/qwen3-8b-routing/calibration.json \
  --max-bias 0.05 \
  --bias-step 0.002 \
  --passes 3 \
  --shrink 0.5
```

## 5. Generate the primary submission

Use routing-aware training plus conservative calibration:

```bash
python3 carrot_router.py predict \
  --profile qwen3-8b-l4 \
  --adapter-dir carrot_runs/qwen3-8b-routing \
  --calibration carrot_runs/qwen3-8b-routing/calibration.json \
  --out submission_carrot_ranking_calibrated.csv \
  --score-out carrot_ranking_calibrated_scores.npz \
  --prediction-out carrot_ranking_calibrated_debug.csv
```

This is the first submission to upload.

## 6. Generate the uncalibrated control

```bash
python3 carrot_router.py predict \
  --profile qwen3-8b-l4 \
  --adapter-dir carrot_runs/qwen3-8b-routing \
  --out submission_carrot_ranking_raw.csv \
  --score-out carrot_ranking_raw_scores.npz
```

Upload this only if the calibrated submission regresses or if quota permits a direct ablation.

## 7. Optional KNN residual correction

Build the correction from validation residuals:

```bash
python3 carrot_router.py calibrate \
  --audit carrot_runs/qwen3-8b-routing/validation_audit.npz \
  --out carrot_runs/qwen3-8b-routing/calibration.json \
  --knn-out carrot_runs/qwen3-8b-routing/knn_residuals.npz \
  --embedding-model intfloat/e5-base-v2 \
  --embedding-batch-size 64 \
  --knn-k 50
```

Generate a low-weight correction candidate:

```bash
python3 carrot_router.py predict \
  --profile qwen3-8b-l4 \
  --adapter-dir carrot_runs/qwen3-8b-routing \
  --calibration carrot_runs/qwen3-8b-routing/calibration.json \
  --knn-correction carrot_runs/qwen3-8b-routing/knn_residuals.npz \
  --knn-weight 0.15 \
  --embedding-batch-size 64 \
  --out submission_carrot_ranking_knn15.csv \
  --score-out carrot_ranking_knn15_scores.npz
```

Treat this as the third candidate, not the default.

## Upload Order

1. `submission_carrot_ranking_calibrated.csv`
2. `submission_carrot_ranking_raw.csv` if calibration regresses
3. `submission_carrot_ranking_knn15.csv` only after the first two results are understood

