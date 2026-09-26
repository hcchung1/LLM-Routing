import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-score weighted vote over prior Kaggle submissions.")
    parser.add_argument("--history", default="kaggle_submissions_history.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_leaderboard_vote.csv")
    parser.add_argument("--score-out", default="")
    parser.add_argument("--used-out", default="")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-public-score", type=float, default=0.44)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--base-submission", default="")
    parser.add_argument("--base-weight", type=float, default=0.0)
    parser.add_argument(
        "--duplicate-policy",
        default="exclude",
        choices=["exclude", "max", "all"],
        help="How to handle repeated file names in history.",
    )
    return parser.parse_args()


def resolve_submission_path(file_name: str) -> Path:
    candidates = [Path(file_name), Path("output") / file_name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(file_name)


def load_submission(path: Path, test_df: pd.DataFrame) -> pd.Series:
    df = pd.read_csv(path)
    expected_columns = ["ID", "pred_model"]
    if df.columns.tolist() != expected_columns:
        raise ValueError(f"{path} columns must be exactly {expected_columns}; got {df.columns.tolist()}.")
    if len(df) != len(test_df):
        raise ValueError(f"{path} has {len(df)} rows, expected {len(test_df)}.")
    if not df["ID"].reset_index(drop=True).equals(test_df["ID"].reset_index(drop=True)):
        raise ValueError(f"{path} ID order does not match test.csv.")
    return df["pred_model"].astype(str)


def select_history_rows(args: argparse.Namespace) -> pd.DataFrame:
    hist = pd.read_csv(args.history)
    hist = hist[hist["status"].eq("Complete")].copy()
    hist = hist[hist["public_score"].notna()].copy()
    hist = hist[hist["public_score"] >= args.min_public_score].copy()

    if args.duplicate_policy == "exclude":
        duplicated = hist["file_name"].duplicated(keep=False)
        hist = hist[~duplicated].copy()
    elif args.duplicate_policy == "max":
        hist = hist.sort_values("public_score", ascending=False).drop_duplicates("file_name", keep="first")

    hist = hist.sort_values("public_score", ascending=False)
    if args.top_n > 0:
        hist = hist.head(args.top_n).copy()
    if hist.empty:
        raise ValueError("No usable history rows after filtering.")
    return hist


def public_score_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("--temperature must be positive.")
    shifted = (scores - scores.max()) / temperature
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Could not compute positive vote weights.")
    return (weights / total).astype(np.float32)


def model_order_from_predictions(predictions: List[pd.Series]) -> List[str]:
    models = sorted({str(value) for preds in predictions for value in preds.unique().tolist()})
    if not models:
        raise ValueError("No model labels found in submissions.")
    return models


def vote_scores(predictions: List[pd.Series], weights: np.ndarray, model_names: List[str]) -> np.ndarray:
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    n_rows = len(predictions[0])
    scores = np.zeros((n_rows, len(model_names)), dtype=np.float32)
    for pred, weight in zip(predictions, weights):
        for row_idx, model_name in enumerate(pred.tolist()):
            scores[row_idx, model_to_idx[str(model_name)]] += float(weight)
    return scores


def add_base_vote(
    scores: np.ndarray,
    base_preds: pd.Series,
    model_names: List[str],
    base_weight: float,
) -> np.ndarray:
    if base_weight <= 0:
        return scores
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    missing = sorted({str(pred) for pred in base_preds.unique().tolist() if str(pred) not in model_to_idx})
    if missing:
        raise ValueError(f"Base submission has unknown labels: {missing}")

    mixed = scores * (1.0 - base_weight)
    for row_idx, model_name in enumerate(base_preds.tolist()):
        mixed[row_idx, model_to_idx[str(model_name)]] += base_weight
    return mixed.astype(np.float32)


def save_score_npz(out_path: str, scores: np.ndarray, ids: pd.Series, model_names: List[str]) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        ids=ids.to_numpy(),
        model_names=np.array(model_names),
    )


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test)
    selected = select_history_rows(args)

    predictions = []
    used_rows = []
    for _, row in selected.iterrows():
        path = resolve_submission_path(str(row["file_name"]))
        preds = load_submission(path, test_df)
        predictions.append(preds)
        used_rows.append(
            {
                "file_name": row["file_name"],
                "public_score": row["public_score"],
                "path": str(path),
            }
        )

    model_names = model_order_from_predictions(predictions)
    weights = public_score_weights(selected["public_score"].to_numpy(dtype=np.float32), args.temperature)
    scores = vote_scores(predictions, weights, model_names)

    if args.base_submission:
        base_preds = load_submission(Path(args.base_submission), test_df)
        if args.base_weight > 0:
            for model_name in base_preds.unique().tolist():
                if str(model_name) not in model_names:
                    model_names.append(str(model_name))
            model_names = sorted(model_names)
            scores = vote_scores(predictions, weights, model_names)
            scores = add_base_vote(scores, base_preds, model_names, args.base_weight)

    preds = [model_names[idx] for idx in scores.argmax(axis=1)]
    pd.DataFrame({"ID": test_df["ID"], "pred_model": preds}).to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(preds)} rows.")
    print(pd.Series(preds).value_counts().sort_index().to_string())

    used_df = pd.DataFrame(used_rows)
    used_df["weight"] = weights
    print("Used submissions:")
    print(used_df.to_string(index=False))
    if args.used_out:
        path = Path(args.used_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        used_df.to_csv(path, index=False)

    if args.score_out:
        save_score_npz(args.score_out, scores, test_df["ID"], model_names)
        print(f"Wrote {args.score_out} with score shape {scores.shape}.")


if __name__ == "__main__":
    main()
