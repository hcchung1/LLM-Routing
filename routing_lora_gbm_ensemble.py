import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler


def parse_model_names(columns: List[str]) -> List[str]:
    models = []
    for col in columns:
        if col.startswith("Model_") and col.endswith("_performance"):
            models.append(col.replace("_performance", ""))
    return sorted(models)


def load_reference(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df, parse_model_names(train_df.columns.tolist())


def read_np_scores(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "scores" in loaded:
            scores = loaded["scores"]
        elif "reward" in loaded:
            scores = loaded["reward"]
        else:
            keys = ", ".join(loaded.files)
            raise ValueError(f"{path} must contain a `scores` or `reward` array; found: {keys}")
        model_names = loaded["model_names"].astype(str).tolist() if "model_names" in loaded else None
        return scores.astype(np.float32), model_names
    return np.asarray(loaded, dtype=np.float32), None


def read_csv_scores(path: Path, model_names: List[str]) -> Tuple[np.ndarray, Optional[List[str]]]:
    df = pd.read_csv(path)
    direct_cols = [m for m in model_names if m in df.columns]
    reward_cols = [f"{m}_reward" for m in model_names if f"{m}_reward" in df.columns]
    score_cols = [f"{m}_score" for m in model_names if f"{m}_score" in df.columns]
    if len(direct_cols) == len(model_names):
        return df[model_names].to_numpy(dtype=np.float32), model_names
    if len(reward_cols) == len(model_names):
        return df[[f"{m}_reward" for m in model_names]].to_numpy(dtype=np.float32), model_names
    if len(score_cols) == len(model_names):
        return df[[f"{m}_score" for m in model_names]].to_numpy(dtype=np.float32), model_names
    raise ValueError(
        f"{path} does not look like a score CSV. Expected columns Model_A.., "
        "Model_A_reward.., or Model_A_score..."
    )


def load_scores(path_str: str, reference_model_names: List[str]) -> Tuple[np.ndarray, Optional[List[str]]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".npz", ".npy"}:
        return read_np_scores(path)
    if path.suffix.lower() == ".csv":
        return read_csv_scores(path, reference_model_names)
    raise ValueError(f"Unsupported score file extension: {path.suffix}")


def align_scores(
    scores: np.ndarray,
    source_model_names: Optional[List[str]],
    reference_model_names: List[str],
    label: str,
) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"{label} scores must be a 2D matrix; got shape {scores.shape}")
    if source_model_names is None:
        if scores.shape[1] != len(reference_model_names):
            raise ValueError(
                f"{label} has {scores.shape[1]} columns, expected {len(reference_model_names)}."
            )
        logger.warning("{} has no model_names metadata; assuming train CSV order.", label)
        return scores.astype(np.float32)

    missing = [name for name in reference_model_names if name not in source_model_names]
    if missing:
        raise ValueError(f"{label} is missing model columns: {missing}")
    order = [source_model_names.index(name) for name in reference_model_names]
    return scores[:, order].astype(np.float32)


def scale_scores(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return scores.astype(np.float32)
    if method == "minmax":
        return MinMaxScaler().fit_transform(scores).astype(np.float32)
    raise ValueError(f"Unknown scale method: {method}")


def make_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
    pd.DataFrame({"ID": test_df["ID"], "pred_model": preds}).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheme 3: ensemble LoRA reward scores with E5/GBM scores.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--starcoder-reward", default="starcoder2_qlora_runs/test_reward_scores.npz")
    parser.add_argument("--gbm-reward", default="e5_gbm_reward.npz")
    parser.add_argument("--out", default="submission_lora_gbm_ensemble.csv")
    parser.add_argument("--final-score-out", default="lora_gbm_final_scores.npz")
    parser.add_argument("--starcoder-weight", type=float, default=0.5)
    parser.add_argument("--gbm-weight", type=float, default=0.5)
    parser.add_argument("--scale", default="minmax", choices=["minmax", "none"])
    args = parser.parse_args()

    _, test_df, model_names = load_reference(args.train, args.test)
    starcoder_scores, starcoder_models = load_scores(args.starcoder_reward, model_names)
    gbm_scores, gbm_models = load_scores(args.gbm_reward, model_names)

    starcoder_scores = align_scores(starcoder_scores, starcoder_models, model_names, "starcoder")
    gbm_scores = align_scores(gbm_scores, gbm_models, model_names, "gbm")
    if starcoder_scores.shape != gbm_scores.shape:
        raise ValueError(f"Score shapes differ: {starcoder_scores.shape} vs {gbm_scores.shape}")
    if len(test_df) != starcoder_scores.shape[0]:
        raise ValueError(f"Test rows={len(test_df)} but score rows={starcoder_scores.shape[0]}")

    total_weight = args.starcoder_weight + args.gbm_weight
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive number.")
    starcoder_weight = args.starcoder_weight / total_weight
    gbm_weight = args.gbm_weight / total_weight

    s1 = scale_scores(starcoder_scores, args.scale)
    s2 = scale_scores(gbm_scores, args.scale)
    final_scores = starcoder_weight * s1 + gbm_weight * s2
    preds = [model_names[idx] for idx in final_scores.argmax(axis=1)]
    make_submission(test_df, preds, args.out)
    logger.info("Saved scheme-3 submission to {}", args.out)

    if args.final_score_out:
        path = Path(args.final_score_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scores=final_scores.astype(np.float32),
            ids=test_df["ID"].to_numpy(),
            model_names=np.array(model_names),
        )
        logger.info("Saved final ensemble scores to {}", args.final_score_out)


if __name__ == "__main__":
    main()
