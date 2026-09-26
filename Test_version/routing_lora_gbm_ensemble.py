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


def parse_weights(raw: str, n_scores: int) -> np.ndarray:
    if raw.strip():
        weights = np.array([float(part.strip()) for part in raw.split(",") if part.strip()], dtype=np.float32)
        if len(weights) != n_scores:
            raise ValueError(f"--weights provides {len(weights)} values, expected {n_scores}.")
    else:
        weights = np.ones(n_scores, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Score weights must sum to a positive value.")
    return weights / total


def blend_score_files(
    score_files: List[str],
    weights: np.ndarray,
    model_names: List[str],
    test_rows: int,
    scale: str,
) -> np.ndarray:
    final_scores = None
    for idx, score_file in enumerate(score_files):
        scores, source_models = load_scores(score_file, model_names)
        scores = align_scores(scores, source_models, model_names, f"score_file_{idx}")
        if len(scores) != test_rows:
            raise ValueError(f"{score_file} has {len(scores)} rows, expected {test_rows}.")
        scaled_scores = scale_scores(scores, scale)
        if final_scores is None:
            final_scores = weights[idx] * scaled_scores
        else:
            final_scores = final_scores + weights[idx] * scaled_scores
    if final_scores is None:
        raise ValueError("No score files were provided.")
    return final_scores.astype(np.float32)


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
    parser.add_argument(
        "--score-files",
        nargs="*",
        default=[],
        help="Optional generic score files. When set, these replace --starcoder-reward/--gbm-reward.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Comma-separated weights for --score-files. Defaults to equal weights.",
    )
    parser.add_argument("--scale", default="minmax", choices=["minmax", "none"])
    args = parser.parse_args()

    _, test_df, model_names = load_reference(args.train, args.test)
    if args.score_files:
        weights = parse_weights(args.weights, len(args.score_files))
        final_scores = blend_score_files(
            args.score_files,
            weights,
            model_names,
            len(test_df),
            args.scale,
        )
        logger.info("Blended {} generic score files with weights {}", len(args.score_files), weights.tolist())
    else:
        score_files = [args.starcoder_reward, args.gbm_reward]
        weights = parse_weights(f"{args.starcoder_weight},{args.gbm_weight}", len(score_files))
        final_scores = blend_score_files(score_files, weights, model_names, len(test_df), args.scale)
        logger.info("Blended legacy StarCoder/GBM scores with weights {}", weights.tolist())

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
