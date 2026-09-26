import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tests.routing_tfidf_knn_router import (
    compute_reward,
    find_text_column,
    parse_model_names,
    router_scores,
)


def load_reference(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    models = parse_model_names(train_df.columns.tolist())
    return train_df, test_df, models


def make_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
    if "ID" not in test_df.columns:
        raise ValueError("test.csv must contain an ID column.")
    pd.DataFrame({"ID": test_df["ID"], "pred_model": preds}).to_csv(out_path, index=False)


def save_score_npz(out_path: str, scores: np.ndarray, ids: pd.Series, model_names: List[str]) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        ids=ids.to_numpy(),
        model_names=np.array(model_names),
    )


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
        return np.asarray(scores, dtype=np.float32), model_names
    return np.asarray(loaded, dtype=np.float32), None


def read_csv_scores(path: Path, model_names: List[str]) -> Tuple[np.ndarray, Optional[List[str]]]:
    df = pd.read_csv(path)
    direct_cols = [model for model in model_names if model in df.columns]
    reward_cols = [f"{model}_reward" for model in model_names if f"{model}_reward" in df.columns]
    score_cols = [f"{model}_score" for model in model_names if f"{model}_score" in df.columns]
    if len(direct_cols) == len(model_names):
        return df[model_names].to_numpy(dtype=np.float32), model_names
    if len(reward_cols) == len(model_names):
        return df[[f"{model}_reward" for model in model_names]].to_numpy(dtype=np.float32), model_names
    if len(score_cols) == len(model_names):
        return df[[f"{model}_score" for model in model_names]].to_numpy(dtype=np.float32), model_names
    raise ValueError(f"{path} does not contain model score columns.")


def load_scores(path_str: str, model_names: List[str]) -> Tuple[np.ndarray, Optional[List[str]]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".npz", ".npy"}:
        return read_np_scores(path)
    if path.suffix.lower() == ".csv":
        return read_csv_scores(path, model_names)
    raise ValueError(f"Unsupported score file extension: {path.suffix}")


def align_scores(
    scores: np.ndarray,
    source_model_names: Optional[List[str]],
    reference_model_names: List[str],
    label: str,
) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"{label} scores must be a 2D matrix; got shape {scores.shape}.")
    if source_model_names is None:
        if scores.shape[1] != len(reference_model_names):
            raise ValueError(f"{label} has {scores.shape[1]} columns, expected {len(reference_model_names)}.")
        return scores.astype(np.float32)

    missing = [model for model in reference_model_names if model not in source_model_names]
    if missing:
        raise ValueError(f"{label} is missing model columns: {missing}")
    order = [source_model_names.index(model) for model in reference_model_names]
    return scores[:, order].astype(np.float32)


def scale_scores(scores: np.ndarray, method: str) -> np.ndarray:
    scores = scores.astype(np.float32)
    if method == "none":
        return scores
    if method == "minmax":
        mins = scores.min(axis=0, keepdims=True)
        maxs = scores.max(axis=0, keepdims=True)
        denom = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)
        return ((scores - mins) / denom).astype(np.float32)
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
        raise ValueError("Weights must sum to a positive number.")
    return weights / total


def blend_scores(
    score_files: List[str],
    weights: np.ndarray,
    model_names: List[str],
    test_rows: int,
    scale: str,
) -> np.ndarray:
    final_scores = np.zeros((test_rows, len(model_names)), dtype=np.float32)
    for idx, score_file in enumerate(score_files):
        scores, source_models = load_scores(score_file, model_names)
        scores = align_scores(scores, source_models, model_names, f"score_file_{idx}")
        if scores.shape[0] != test_rows:
            raise ValueError(f"{score_file} has {scores.shape[0]} rows, expected {test_rows}.")
        final_scores += weights[idx] * scale_scores(scores, scale)
    return final_scores.astype(np.float32)


def constant_scores(model_names: List[str], model_name: str, n_rows: int) -> np.ndarray:
    if model_name not in model_names:
        raise ValueError(f"Unknown constant model {model_name!r}; expected one of {model_names}.")
    scores = np.zeros((n_rows, len(model_names)), dtype=np.float32)
    scores[:, model_names.index(model_name)] = 1.0
    return scores


def print_train_baselines(reward_df: pd.DataFrame) -> None:
    means = reward_df.mean(axis=0).sort_values(ascending=False)
    print("Train constant-model rewards:")
    for model_name, score in means.items():
        print(f"  {model_name}: {score:.6f}")


def print_prediction_counts(preds: List[str]) -> None:
    counts = pd.Series(preds).value_counts().sort_index()
    print("Prediction counts:")
    for model_name, count in counts.items():
        print(f"  {model_name}: {int(count)}")


def load_base_submission(path_str: str, test_df: pd.DataFrame) -> List[str]:
    df = pd.read_csv(path_str)
    expected_columns = ["ID", "pred_model"]
    if df.columns.tolist() != expected_columns:
        raise ValueError(f"{path_str} columns must be exactly {expected_columns}; got {df.columns.tolist()}.")
    if len(df) != len(test_df):
        raise ValueError(f"{path_str} has {len(df)} rows, expected {len(test_df)}.")
    if not df["ID"].reset_index(drop=True).equals(test_df["ID"].reset_index(drop=True)):
        raise ValueError(f"{path_str} ID order does not match test.csv.")
    return df["pred_model"].astype(str).tolist()


def apply_base_overrides(
    base_preds: List[str],
    scores: np.ndarray,
    model_names: List[str],
    min_override_margin: float,
    min_top_gap: float,
    max_overrides: int,
) -> Tuple[List[str], pd.DataFrame]:
    model_to_idx = {model_name: idx for idx, model_name in enumerate(model_names)}
    missing = sorted({pred for pred in base_preds if pred not in model_to_idx})
    if missing:
        raise ValueError(f"Base submission contains unknown model labels: {missing}")

    top_idx = scores.argmax(axis=1)
    top_scores = scores[np.arange(len(scores)), top_idx]
    if scores.shape[1] > 1:
        second_scores = np.partition(scores, -2, axis=1)[:, -2]
    else:
        second_scores = np.zeros(len(scores), dtype=np.float32)

    base_idx = np.array([model_to_idx[pred] for pred in base_preds], dtype=np.int64)
    base_scores = scores[np.arange(len(scores)), base_idx]
    margins = top_scores - base_scores
    top_gaps = top_scores - second_scores
    override_mask = (
        (top_idx != base_idx)
        & (margins >= min_override_margin)
        & (top_gaps >= min_top_gap)
    )

    candidate_idx = np.flatnonzero(override_mask)
    selected_mask = np.zeros(len(scores), dtype=bool)
    if max_overrides > 0 and len(candidate_idx) > max_overrides:
        order = np.lexsort((-top_gaps[candidate_idx], -margins[candidate_idx]))
        selected = candidate_idx[order[:max_overrides]]
        selected_mask[selected] = True
    else:
        selected_mask[candidate_idx] = True

    final_preds = list(base_preds)
    score_preds = [model_names[idx] for idx in top_idx]
    for idx in np.flatnonzero(selected_mask):
        final_preds[idx] = score_preds[idx]

    debug_df = pd.DataFrame(
        {
            "base_pred": base_preds,
            "score_pred": score_preds,
            "final_pred": final_preds,
            "override": selected_mask,
            "override_margin": margins.astype(np.float32),
            "top_gap": top_gaps.astype(np.float32),
        }
    )
    return final_preds, debug_df


def print_override_summary(debug_df: pd.DataFrame) -> None:
    selected = debug_df[debug_df["override"]]
    print(f"Overrides selected: {len(selected)}")
    if selected.empty:
        return

    transitions: Dict[Tuple[str, str], int] = {}
    for base_pred, final_pred in zip(selected["base_pred"], selected["final_pred"]):
        transitions[(str(base_pred), str(final_pred))] = transitions.get((str(base_pred), str(final_pred)), 0) + 1
    print("Override transitions:")
    for (base_pred, final_pred), count in sorted(transitions.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {base_pred} -> {final_pred}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create constant, TF-IDF, or blended LLM-routing submissions.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_reward_blend.csv")
    parser.add_argument("--score-out", default="")
    parser.add_argument("--mode", choices=["constant", "tfidf", "blend"], default="blend")
    parser.add_argument("--constant-model", default="Model_F")
    parser.add_argument("--score-files", nargs="*", default=[])
    parser.add_argument("--weights", default="")
    parser.add_argument("--scale", default="minmax", choices=["minmax", "none"])
    parser.add_argument("--base-submission", default="")
    parser.add_argument("--min-override-margin", type=float, default=0.0)
    parser.add_argument("--min-top-gap", type=float, default=0.0)
    parser.add_argument("--max-overrides", type=int, default=0)
    parser.add_argument("--debug-out", default="")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--knn-weight", type=float, default=0.70)
    parser.add_argument("--prior-smoothing", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df, model_names = load_reference(args.train, args.test)
    reward_df = compute_reward(train_df, model_names, args.alpha)
    print_train_baselines(reward_df)

    if args.mode == "constant":
        scores = constant_scores(model_names, args.constant_model, len(test_df))
    elif args.mode == "tfidf":
        text_col = find_text_column(train_df)
        if text_col not in test_df.columns:
            raise ValueError(f"test.csv is missing text column {text_col!r}.")
        scores = router_scores(
            train_df,
            test_df,
            reward_df,
            text_col,
            args.k,
            args.knn_weight,
            args.prior_smoothing,
            args.batch_size,
        )
    else:
        if not args.score_files:
            raise ValueError("--mode blend requires at least one --score-files value.")
        weights = parse_weights(args.weights, len(args.score_files))
        scores = blend_scores(args.score_files, weights, model_names, len(test_df), args.scale)
        print(f"Blend weights: {weights.tolist()}")

    debug_df = None
    if args.base_submission:
        base_preds = load_base_submission(args.base_submission, test_df)
        preds, debug_df = apply_base_overrides(
            base_preds,
            scores,
            model_names,
            args.min_override_margin,
            args.min_top_gap,
            args.max_overrides,
        )
        print_override_summary(debug_df)
    else:
        preds = [model_names[idx] for idx in scores.argmax(axis=1)]

    make_submission(test_df, preds, args.out)
    print_prediction_counts(preds)
    print(f"Wrote {args.out} with {len(preds)} rows.")

    if args.debug_out and debug_df is not None:
        debug_path = Path(args.debug_out)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_with_ids = pd.concat([test_df[["ID"]].reset_index(drop=True), debug_df], axis=1)
        debug_with_ids.to_csv(debug_path, index=False)
        print(f"Wrote {args.debug_out} with {len(debug_with_ids)} rows.")

    if args.score_out:
        save_score_npz(args.score_out, scores, test_df["ID"], model_names)
        print(f"Wrote {args.score_out} with score shape {scores.shape}.")


if __name__ == "__main__":
    main()
