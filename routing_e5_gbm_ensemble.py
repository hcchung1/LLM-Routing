import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor


@dataclass
class RewardConfig:
    alpha: float = 0.85
    cost_norm: str = "global-max"


def parse_model_names(columns: List[str]) -> List[str]:
    models = []
    for col in columns:
        if col.startswith("Model_") and col.endswith("_performance"):
            models.append(col.replace("_performance", ""))
    return sorted(models)


def load_data(train_path: str, test_path: str, limit: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if limit > 0:
        train_df = train_df.head(limit).copy()
        test_df = test_df.head(limit).copy()
    return train_df, test_df


def extract_perf_cost(train_df: pd.DataFrame, models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf = train_df[[f"{m}_performance" for m in models]].copy()
    cost = train_df[[f"{m}_cost" for m in models]].copy()
    perf.columns = models
    cost.columns = models
    return perf, cost


def cost_denominator(cost: pd.DataFrame, method: str) -> float:
    if method == "global-max":
        denom = float(cost.max().max())
    elif method == "mean-row-max":
        denom = float(cost.max(axis=1).mean())
    elif method == "none":
        denom = 1.0
    else:
        raise ValueError(f"Unknown cost_norm denominator: {method}")
    return denom if denom != 0 else 1.0


def normalize_cost(cost: pd.DataFrame, cfg: RewardConfig) -> pd.DataFrame:
    if cfg.cost_norm == "row-max":
        row_max = cost.max(axis=1).replace(0, 1.0)
        return cost.div(row_max, axis=0)
    if cfg.cost_norm in {"global-max", "mean-row-max", "none"}:
        return cost / cost_denominator(cost, cfg.cost_norm)
    raise ValueError(f"Unknown cost_norm: {cfg.cost_norm}")


def compute_reward(perf: pd.DataFrame, cost: pd.DataFrame, cfg: RewardConfig) -> pd.DataFrame:
    return cfg.alpha * perf - (1.0 - cfg.alpha) * normalize_cost(cost, cfg)


def make_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
    pd.DataFrame({"ID": test_df["ID"], "pred_model": preds}).to_csv(out_path, index=False)


def cache_paths(cache_dir: Path, model_name: str, train_path: str, test_path: str, limit: int) -> Tuple[Path, Path]:
    safe_model = model_name.replace("/", "__")
    train_key = Path(train_path).stem
    test_key = Path(test_path).stem
    limit_key = f"_limit{limit}" if limit > 0 else ""
    return (
        cache_dir / f"{safe_model}_{train_key}{limit_key}_train.npy",
        cache_dir / f"{safe_model}_{test_key}{limit_key}_test.npy",
    )


def encode_queries(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache, test_cache = cache_paths(cache_dir, args.embedding_model, args.train, args.test, args.limit)

    if args.use_cache and train_cache.exists() and test_cache.exists():
        logger.info("Loading cached embeddings from {}", cache_dir)
        return np.load(train_cache), np.load(test_cache)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required. Install it with `pip install sentence-transformers`."
        ) from exc

    logger.info("Loading embedding model: {}", args.embedding_model)
    embedder = SentenceTransformer(args.embedding_model, device=args.device or None)
    train_texts = ["query: " + q for q in train_df["query"].astype(str).tolist()]
    test_texts = ["query: " + q for q in test_df["query"].astype(str).tolist()]

    logger.info("Encoding train queries")
    x_train = embedder.encode(
        train_texts,
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    logger.info("Encoding test queries")
    x_test = embedder.encode(
        test_texts,
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if args.use_cache:
        np.save(train_cache, x_train)
        np.save(test_cache, x_test)
        logger.info("Saved embeddings to {}", cache_dir)
    return x_train, x_test


def aligned_predict_proba(model, x_test: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x_test)
    out = np.zeros((x_test.shape[0], n_classes), dtype=np.float32)
    for src_idx, class_id in enumerate(model.classes_):
        out[:, int(class_id)] = raw[:, src_idx]
    return out


def train_classifier_ensemble(
    x_train: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    args,
) -> np.ndarray:
    logger.info("Training GradientBoostingClassifier")
    gbm = GradientBoostingClassifier(
        n_estimators=args.gbm_estimators,
        max_depth=args.gbm_depth,
        learning_rate=args.gbm_lr,
        random_state=args.seed,
    )
    gbm.fit(x_train, y)

    logger.info("Training RandomForestClassifier")
    rf = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_depth=None,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    rf.fit(x_train, y)

    logger.info("Training LogisticRegression")
    lr = LogisticRegression(
        C=args.lr_c,
        max_iter=args.lr_max_iter,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    lr.fit(x_train, y)

    prob = (
        aligned_predict_proba(gbm, x_test, n_classes)
        + aligned_predict_proba(rf, x_test, n_classes)
        + aligned_predict_proba(lr, x_test, n_classes)
    ) / 3.0
    return prob.astype(np.float32)


def train_reward_regressor(
    x_train: np.ndarray,
    reward: np.ndarray,
    x_test: np.ndarray,
    args,
) -> np.ndarray:
    logger.info("Training reward regressor for scheme-3 ensembling")
    reg = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=args.reg_estimators,
            max_depth=args.reg_depth,
            learning_rate=args.reg_lr,
            random_state=args.seed,
        ),
        n_jobs=args.reg_n_jobs,
    )
    reg.fit(x_train, reward)
    return reg.predict(x_test).astype(np.float32)


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
    parser = argparse.ArgumentParser(description="Scheme 1: E5 embeddings + GBM/RF/LR ensemble.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_e5_ensemble.csv")
    parser.add_argument("--prob-out", default="")
    parser.add_argument("--reward-out", default="e5_gbm_reward.npz")
    parser.add_argument("--metadata-out", default="e5_gbm_metadata.json")
    parser.add_argument("--embedding-model", default="intfloat/e5-large-v2")
    parser.add_argument("--cache-dir", default="e5_embedding_cache")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--device", default="", help="SentenceTransformer device, e.g. cpu or cuda.")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--cost-norm",
        default="global-max",
        choices=["global-max", "mean-row-max", "row-max", "none"],
    )
    parser.add_argument("--gbm-estimators", type=int, default=500)
    parser.add_argument("--gbm-depth", type=int, default=5)
    parser.add_argument("--gbm-lr", type=float, default=0.05)
    parser.add_argument("--rf-estimators", type=int, default=500)
    parser.add_argument("--lr-c", type=float, default=1.0)
    parser.add_argument("--lr-max-iter", type=int, default=1000)
    parser.add_argument("--reg-estimators", type=int, default=300)
    parser.add_argument("--reg-depth", type=int, default=4)
    parser.add_argument("--reg-lr", type=float, default=0.05)
    parser.add_argument("--reg-n-jobs", type=int, default=-1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test limit for train/test rows.")
    parser.add_argument("--no-reward-regression", dest="reward_regression", action="store_false")
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_df, test_df = load_data(args.train, args.test, limit=args.limit)
    model_names = parse_model_names(train_df.columns.tolist())
    perf, cost = extract_perf_cost(train_df, model_names)
    cfg = RewardConfig(alpha=args.alpha, cost_norm=args.cost_norm)
    reward = compute_reward(perf, cost, cfg)
    reward_array = reward.to_numpy(dtype=np.float32)
    y = reward_array.argmax(axis=1).astype(np.int64)

    x_train, x_test = encode_queries(train_df, test_df, args)

    prob = train_classifier_ensemble(x_train, y, x_test, len(model_names), args)
    preds = [model_names[idx] for idx in prob.argmax(axis=1)]
    make_submission(test_df, preds, args.out)
    logger.info("Saved scheme-1 submission to {}", args.out)

    if args.prob_out:
        save_score_npz(args.prob_out, prob, test_df["ID"], model_names)
        logger.info("Saved classifier probabilities to {}", args.prob_out)

    if args.reward_regression:
        pred_reward = train_reward_regressor(x_train, reward_array, x_test, args)
        save_score_npz(args.reward_out, pred_reward, test_df["ID"], model_names)
        logger.info("Saved reward-regression scores to {}", args.reward_out)

    if args.metadata_out:
        metadata = {
            "model_names": model_names,
            "embedding_model": args.embedding_model,
            "alpha": args.alpha,
            "cost_norm": args.cost_norm,
        }
        Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
