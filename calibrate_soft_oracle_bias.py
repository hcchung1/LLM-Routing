import json
import numpy as np
import pandas as pd
from pathlib import Path

AUDIT = Path("output_G/carrot_runs/qwen3-8b-soft-oracle-t005/validation_audit.npz")
TEST_SCORE = Path("output_G/carrot_soft_oracle_t005/scores_carrot_soft_oracle_t005.npz")
SAMPLE = Path("dataset/sample_submission.csv")
OUT_JSON = Path("output_G/carrot_soft_oracle_t005/soft_oracle_bias.json")
OUT_CSV = Path("output_G/carrot_soft_oracle_t005/submission_carrot_soft_oracle_t005_biascal.csv")

audit = np.load(AUDIT, allow_pickle=True)
val_logits = audit["logits"].astype(np.float32)
true_reward = audit["true_reward"].astype(np.float32)
model_names = [str(x) for x in audit["model_names"]]

def score_with_bias(bias):
    pred = (val_logits + bias[None, :]).argmax(axis=1)
    rows = np.arange(len(pred))
    return float(true_reward[rows, pred].mean())

bias = np.zeros(val_logits.shape[1], dtype=np.float32)
best = score_with_bias(bias)

# 粗到細 coordinate search
for step, radius, passes in [(0.10, 1.5, 3), (0.05, 0.5, 3), (0.02, 0.2, 2)]:
    grid = np.arange(-radius, radius + 1e-9, step, dtype=np.float32)
    for _ in range(passes):
        improved = False
        for j in range(len(bias)):
            current = bias[j]
            local_best = best
            local_value = current
            for delta in grid:
                trial = bias.copy()
                trial[j] = current + delta
                s = score_with_bias(trial)
                if s > local_best:
                    local_best = s
                    local_value = trial[j]
            if local_best > best:
                bias[j] = local_value
                best = local_best
                improved = True
        if not improved:
            break

# shrink 避免對 validation 過擬合
shrink = 0.7
bias = bias * shrink
final_val_score = score_with_bias(bias)

print("model_names:", model_names)
print("bias:", dict(zip(model_names, bias.tolist())))
print("validation reward before:", score_with_bias(np.zeros_like(bias)))
print("validation reward after :", final_val_score)

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "model_names": model_names,
    "bias": bias.tolist(),
    "shrink": shrink,
    "validation_reward_after": final_val_score,
}, indent=2))

test = np.load(TEST_SCORE, allow_pickle=True)
test_logits = test["logits"].astype(np.float32)
ids = test["ids"]

pred_idx = (test_logits + bias[None, :]).argmax(axis=1)
pred_model = [model_names[i] for i in pred_idx]

sample = pd.read_csv(SAMPLE)
sub = pd.DataFrame({
    "ID": sample["ID"],
    "pred_model": pred_model,
})
sub.to_csv(OUT_CSV, index=False)

print(sub["pred_model"].value_counts().sort_index())
print("saved:", OUT_CSV)

