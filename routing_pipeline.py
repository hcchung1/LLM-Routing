import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


@dataclass
class RewardConfig:
    alpha: float = 0.85
    cost_norm: str = "row-max"  # row-max|minmax-global|minmax-per-model|zscore-global|none


def parse_model_names(columns: List[str]) -> List[str]:
    models = []
    for col in columns:
        if col.endswith("_performance") and col.startswith("Model_"):
            models.append(col.replace("_performance", ""))
    return sorted(models)


def normalize_cost(cost: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "row-max":
        row_max = cost.max(axis=1).replace(0, 1.0)
        return cost.div(row_max, axis=0)
    if method == "none":
        return cost
    if method == "minmax-global":
        cmin = cost.min().min()
        cmax = cost.max().max()
        denom = (cmax - cmin) if cmax != cmin else 1.0
        return (cost - cmin) / denom
    if method == "zscore-global":
        mean = cost.stack().mean()
        std = cost.stack().std() or 1.0
        return (cost - mean) / std
    if method == "minmax-per-model":
        cmin = cost.min()
        cmax = cost.max()
        denom = (cmax - cmin).replace(0, 1.0)
        return (cost - cmin) / denom
    raise ValueError(f"Unknown cost_norm: {method}")


def compute_reward(
    perf: pd.DataFrame, cost: pd.DataFrame, cfg: RewardConfig
) -> pd.DataFrame:
    # Reward_{0.85} = 0.85 * P - 0.15 * (C / C_max) per sample.
    cost_n = normalize_cost(cost, cfg.cost_norm)
    alpha = cfg.alpha
    reward = alpha * perf - (1.0 - alpha) * cost_n
    return reward


def build_labels(reward: pd.DataFrame) -> pd.Series:
    return reward.idxmax(axis=1)


def baseline_global_best_label(reward: pd.DataFrame) -> str:
    return reward.mean(axis=0).idxmax()


def evaluate_reward(
    reward: pd.DataFrame, pred_models: pd.Series
) -> float:
    idx = np.arange(len(pred_models))
    return reward.to_numpy()[idx, reward.columns.get_indexer(pred_models)].mean()


def train_tfidf_logreg(
    train_df: pd.DataFrame,
    labels: pd.Series,
    max_features: int = 40000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        max_features=max_features, min_df=min_df, ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(train_df["query"])
    model = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        solver="lbfgs",
    )
    model.fit(X, labels)
    return vectorizer, model


def crossval_tfidf_logreg(
    train_df: pd.DataFrame,
    labels: pd.Series,
    reward: pd.DataFrame,
    n_splits: int = 5,
) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(train_df, labels):
        X_train = train_df.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        X_val = train_df.iloc[val_idx]
        vectorizer, model = train_tfidf_logreg(X_train, y_train)
        Xv = vectorizer.transform(X_val["query"])
        preds = model.predict(Xv)
        score = evaluate_reward(reward.iloc[val_idx], pd.Series(preds))
        scores.append(score)
    return float(np.mean(scores))


def predict_tfidf_logreg(
    train_df: pd.DataFrame, labels: pd.Series, test_df: pd.DataFrame
) -> pd.Series:
    vectorizer, model = train_tfidf_logreg(train_df, labels)
    Xt = vectorizer.transform(test_df["query"])
    preds = model.predict(Xt)
    return pd.Series(preds, index=test_df.index)


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def make_submission(test_df: pd.DataFrame, preds: pd.Series, out_path: str) -> None:
    sub = pd.DataFrame({"ID": test_df["ID"], "pred_model": preds})
    sub.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--mode", choices=["baseline", "tfidf"], default="tfidf")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--cost-norm",
        default="row-max",
        choices=["row-max", "none", "minmax-global", "minmax-per-model", "zscore-global"],
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--export-config", default="")
    args = parser.parse_args()

    if args.export_config:
        cfg = RewardConfig(alpha=args.alpha, cost_norm=args.cost_norm)
        with open(args.export_config, "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, indent=2)
        return

    train_df, test_df = load_data(args.train, args.test)
    models = parse_model_names(train_df.columns.tolist())

    perf_cols = [f"{m}_performance" for m in models]
    cost_cols = [f"{m}_cost" for m in models]
    perf = train_df[perf_cols].copy()
    cost = train_df[cost_cols].copy()
    perf.columns = models
    cost.columns = models

    cfg = RewardConfig(alpha=args.alpha, cost_norm=args.cost_norm)
    reward = compute_reward(perf, cost, cfg)
    labels = build_labels(reward)

    if args.mode == "baseline":
        best_model = baseline_global_best_label(reward)
        preds = pd.Series([best_model] * len(test_df), index=test_df.index)
        make_submission(test_df, preds, args.out)
        return

    if args.mode == "tfidf":
        if args.cv > 1:
            cv_score = crossval_tfidf_logreg(train_df, labels, reward, n_splits=args.cv)
            print(f"CV reward: {cv_score:.6f}")
        preds = predict_tfidf_logreg(train_df, labels, test_df)
        make_submission(test_df, preds, args.out)
        return


if __name__ == "__main__":
    main()
