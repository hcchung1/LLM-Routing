import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise preference router for Kaggle LLM routing.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_pairwise_ranker.csv")
    parser.add_argument("--score-out", default="")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--analyzer", default="char_wb", choices=["word", "char", "char_wb"])
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--margin-power", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--cv-seeds", default="")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_model_names(columns: List[str]) -> List[str]:
    models = []
    for col in columns:
        if col.startswith("Model_") and col.endswith("_performance"):
            model = col.removesuffix("_performance")
            if f"{model}_cost" in columns:
                models.append(model)
    if not models:
        raise ValueError("No model performance/cost column pairs found.")
    return sorted(models)


def find_text_column(df: pd.DataFrame) -> str:
    for col in ["query", "prompt", "Question", "question", "text"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a text column.")


def compute_reward(train_df: pd.DataFrame, models: List[str], alpha: float) -> pd.DataFrame:
    perf = train_df[[f"{model}_performance" for model in models]].copy()
    cost = train_df[[f"{model}_cost" for model in models]].copy()
    perf.columns = models
    cost.columns = models
    denom = float(cost.max().max())
    if denom <= 0:
        denom = 1.0
    return alpha * perf - (1.0 - alpha) * (cost / denom)


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        strip_accents="unicode",
        norm="l2",
    )


def predict_pair_proba(
    x_train: sparse.spmatrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    x_test: sparse.spmatrix,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    unique = np.unique(y)
    if len(unique) == 1:
        return np.full(x_test.shape[0], float(unique[0]), dtype=np.float32)

    class_weight = None if args.class_weight == "none" else "balanced"
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.sgd_alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train, y, sample_weight=sample_weight)
    proba = clf.predict_proba(x_test)
    pos_col = int(np.flatnonzero(clf.classes_ == 1)[0])
    return proba[:, pos_col].astype(np.float32)


def pairwise_vote_scores(
    x_train: sparse.spmatrix,
    reward_train: np.ndarray,
    x_test: sparse.spmatrix,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    n_models = reward_train.shape[1]
    scores = np.zeros((x_test.shape[0], n_models), dtype=np.float32)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = reward_train[:, i] - reward_train[:, j]
            mask = np.abs(diff) >= args.min_margin
            if not np.any(mask):
                prob_i = np.full(x_test.shape[0], 0.5, dtype=np.float32)
            else:
                y = (diff[mask] > 0).astype(np.int64)
                margins = np.abs(diff[mask]).astype(np.float32)
                sample_weight = np.power(np.maximum(margins, 1e-6), args.margin_power)
                prob_i = predict_pair_proba(
                    x_train[mask],
                    y,
                    sample_weight,
                    x_test,
                    args,
                    seed + i * n_models + j,
                )
            scores[:, i] += prob_i
            scores[:, j] += 1.0 - prob_i

    return (scores / max(1, n_models - 1)).astype(np.float32)


def evaluate_reward(reward_df: pd.DataFrame, preds: List[str]) -> float:
    row_idx = np.arange(len(preds))
    col_idx = reward_df.columns.get_indexer(preds)
    return float(reward_df.to_numpy(dtype=np.float32)[row_idx, col_idx].mean())


def parse_cv_seeds(raw: str, seed: int, repeats: int) -> List[int]:
    if raw.strip():
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [seed + offset for offset in range(max(1, repeats))]


def train_predict_scores(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    text_col: str,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    vectorizer = make_vectorizer(args)
    x_train = vectorizer.fit_transform(train_df[text_col].astype(str))
    x_target = vectorizer.transform(target_df[text_col].astype(str))
    reward_train = reward_df.to_numpy(dtype=np.float32)
    return pairwise_vote_scores(x_train, reward_train, x_target, args, seed)


def run_cv(train_df: pd.DataFrame, reward_df: pd.DataFrame, args: argparse.Namespace) -> None:
    text_col = find_text_column(train_df)
    seeds = parse_cv_seeds(args.cv_seeds, args.seed, args.cv_repeats)
    scores = []
    for seed in seeds:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=args.val_size,
            random_state=seed,
            shuffle=True,
        )
        fit_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        fit_reward = reward_df.iloc[train_idx].reset_index(drop=True)
        val_reward = reward_df.iloc[val_idx].reset_index(drop=True)
        pred_scores = train_predict_scores(fit_df, val_df, fit_reward, text_col, args, seed)
        preds = [reward_df.columns[idx] for idx in pred_scores.argmax(axis=1)]
        score = evaluate_reward(val_reward, preds)
        scores.append(score)
        print(f"Local Reward_{args.alpha:.2f} seed={seed}: {score:.6f}")

    values = np.array(scores, dtype=np.float32)
    print(
        "Local Reward_{:.2f} summary: mean={:.6f} std={:.6f} min={:.6f} max={:.6f} n={}".format(
            args.alpha,
            float(values.mean()),
            float(values.std(ddof=0)),
            float(values.min()),
            float(values.max()),
            len(values),
        )
    )


def make_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
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


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args()
    if args.ngram_min > args.ngram_max:
        raise ValueError("--ngram-min must be <= --ngram-max")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    if args.limit > 0:
        train_df = train_df.head(args.limit).copy()
        test_df = test_df.head(args.limit).copy()

    models = parse_model_names(train_df.columns.tolist())
    reward_df = compute_reward(train_df, models, args.alpha)

    if args.cv:
        run_cv(train_df, reward_df, args)
        return

    text_col = find_text_column(train_df)
    if text_col not in test_df.columns:
        raise ValueError(f"test.csv is missing text column {text_col!r}.")

    scores = train_predict_scores(train_df, test_df, reward_df, text_col, args, args.seed)
    preds = [models[idx] for idx in scores.argmax(axis=1)]
    make_submission(test_df, preds, args.out)
    print(f"Wrote {args.out} with {len(preds)} rows.")
    print(pd.Series(preds).value_counts().sort_index().to_string())

    if args.score_out:
        save_score_npz(args.score_out, scores, test_df["ID"], models)
        print(f"Wrote {args.score_out} with score shape {scores.shape}.")


if __name__ == "__main__":
    main()
