import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ScoreBlock:
    name: str
    scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-view reward router with local CV board and champion-prior candidates."
    )
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--base-submission", default="output/submission_starcoder2_qlora.csv")
    parser.add_argument("--out-dir", default="output/multiview_reward")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--knn-k", default="5,20,50")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--candidate-top-n", type=int, default=5)
    parser.add_argument(
        "--base-weights",
        default="0.15,0.25,0.35,0.45",
        help="Comma-separated champion one-hot prior weights for candidate blends.",
    )
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_ints(raw: str) -> List[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_floats(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_model_names(columns: Sequence[str]) -> List[str]:
    models = []
    for col in columns:
        if col.startswith("Model_") and col.endswith("_performance"):
            model = col.removesuffix("_performance")
            if f"{model}_cost" in columns:
                models.append(model)
    if not models:
        raise ValueError("No model performance/cost column pairs found.")
    return sorted(models)


def compute_reward(df: pd.DataFrame, models: List[str], alpha: float) -> pd.DataFrame:
    perf = df[[f"{model}_performance" for model in models]].copy()
    cost = df[[f"{model}_cost" for model in models]].copy()
    perf.columns = models
    cost.columns = models
    denom = float(cost.max().max())
    if denom <= 0:
        denom = 1.0
    return alpha * perf - (1.0 - alpha) * (cost / denom)


def evaluate_reward(reward_df: pd.DataFrame, preds: Sequence[str]) -> float:
    row_idx = np.arange(len(preds))
    col_idx = reward_df.columns.get_indexer(list(preds))
    if np.any(col_idx < 0):
        missing = sorted({pred for pred, idx in zip(preds, col_idx) if idx < 0})
        raise ValueError(f"Unknown prediction labels: {missing}")
    return float(reward_df.to_numpy(dtype=np.float32)[row_idx, col_idx].mean())


def make_submission(test_df: pd.DataFrame, preds: Sequence[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": test_df["ID"], "pred_model": list(preds)}).to_csv(out_path, index=False)


def save_score_npz(out_path: Path, scores: np.ndarray, ids: pd.Series, model_names: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=scores.astype(np.float32),
        ids=ids.to_numpy(),
        model_names=np.array(model_names),
    )


def load_base_one_hot(path: str, test_df: pd.DataFrame, model_names: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    expected_columns = ["ID", "pred_model"]
    if df.columns.tolist() != expected_columns:
        raise ValueError(f"{path} columns must be exactly {expected_columns}; got {df.columns.tolist()}.")
    if len(df) < len(test_df):
        raise ValueError(f"{path} has {len(df)} rows, expected at least {len(test_df)}.")
    if len(df) > len(test_df):
        df = df.head(len(test_df)).copy()
    if not df["ID"].reset_index(drop=True).equals(test_df["ID"].reset_index(drop=True)):
        raise ValueError(f"{path} ID order does not match test.csv.")
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    missing = sorted({label for label in df["pred_model"].astype(str).unique().tolist() if label not in model_to_idx})
    if missing:
        raise ValueError(f"{path} contains unknown model labels: {missing}")
    one_hot = np.zeros((len(df), len(model_names)), dtype=np.float32)
    for row_idx, label in enumerate(df["pred_model"].astype(str).tolist()):
        one_hot[row_idx, model_to_idx[label]] = 1.0
    return one_hot


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    mins = scores.min(axis=1, keepdims=True)
    maxs = scores.max(axis=1, keepdims=True)
    denom = np.where((maxs - mins) > 1e-8, maxs - mins, 1.0)
    return ((scores - mins) / denom).astype(np.float32)


def row_softmax(scores: np.ndarray, temperature: float = 0.08) -> np.ndarray:
    scaled = scores.astype(np.float32) / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def vectorizer_specs(max_features: int) -> List[Tuple[str, TfidfVectorizer]]:
    return [
        (
            "word12",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=max_features,
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                norm="l2",
            ),
        ),
        (
            "charwb35",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_features,
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                norm="l2",
            ),
        ),
        (
            "char35",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=max_features,
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                norm="l2",
            ),
        ),
    ]


def topk_weighted_reward(
    x_train: sparse.spmatrix,
    x_query: sparse.spmatrix,
    reward_train: np.ndarray,
    k: int,
    batch_size: int,
) -> np.ndarray:
    n_query = x_query.shape[0]
    n_train = x_train.shape[0]
    k = max(1, min(k, n_train))
    reward_train = reward_train.astype(np.float32, copy=False)
    global_mean = reward_train.mean(axis=0).astype(np.float32)
    scores = np.zeros((n_query, reward_train.shape[1]), dtype=np.float32)

    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = (x_query[start:end] @ x_train.T).toarray().astype(np.float32, copy=False)
        if k < n_train:
            top_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        else:
            top_idx = np.tile(np.arange(n_train), (end - start, 1))
        top_sim = np.take_along_axis(sim, top_idx, axis=1)
        denom = top_sim.sum(axis=1, keepdims=True)
        weighted = np.einsum("bk,bkm->bm", top_sim, reward_train[top_idx], optimize=True)
        valid = denom[:, 0] > 1e-8
        block = scores[start:end]
        block[valid] = weighted[valid] / denom[valid]
        block[~valid] = global_mean
    return scores


def ridge_reward_scores(
    x_train: sparse.spmatrix,
    x_query: sparse.spmatrix,
    reward_train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    reg = Ridge(alpha=alpha, random_state=0)
    reg.fit(x_train, reward_train.astype(np.float32))
    return reg.predict(x_query).astype(np.float32)


def cluster_prior_scores(
    x_train: sparse.spmatrix,
    x_query: sparse.spmatrix,
    reward_train: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    n_clusters = max(1, min(n_clusters, x_train.shape[0]))
    if n_clusters == 1:
        return np.tile(reward_train.mean(axis=0, keepdims=True), (x_query.shape[0], 1)).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=1024,
        n_init=3,
        reassignment_ratio=0.01,
    )
    train_clusters = kmeans.fit_predict(x_train)
    query_clusters = kmeans.predict(x_query)
    global_mean = reward_train.mean(axis=0).astype(np.float32)
    cluster_means = np.tile(global_mean[None, :], (n_clusters, 1)).astype(np.float32)
    for cluster_id in range(n_clusters):
        mask = train_clusters == cluster_id
        if np.any(mask):
            cluster_means[cluster_id] = reward_train[mask].mean(axis=0)
    return cluster_means[query_clusters].astype(np.float32)


def build_score_blocks(
    train_texts: pd.Series,
    query_texts: pd.Series,
    reward_train: np.ndarray,
    args: argparse.Namespace,
    ks: List[int],
) -> List[ScoreBlock]:
    blocks: List[ScoreBlock] = []
    for view_name, vectorizer in vectorizer_specs(args.max_features):
        print(f"Fitting vectorizer {view_name} on {len(train_texts)} rows")
        x_train = vectorizer.fit_transform(train_texts.astype(str))
        x_query = vectorizer.transform(query_texts.astype(str))
        for k in ks:
            print(f"  scoring {view_name}_knn{k}")
            scores = topk_weighted_reward(x_train, x_query, reward_train, k, args.batch_size)
            blocks.append(ScoreBlock(f"{view_name}_knn{k}", scores))
        print(f"  scoring {view_name}_ridge")
        blocks.append(ScoreBlock(f"{view_name}_ridge", ridge_reward_scores(x_train, x_query, reward_train, args.ridge_alpha)))
        if args.clusters > 0:
            print(f"  scoring {view_name}_cluster{args.clusters}")
            blocks.append(
                ScoreBlock(
                    f"{view_name}_cluster{args.clusters}",
                    cluster_prior_scores(x_train, x_query, reward_train, args.clusters),
                )
            )
    return blocks


def average_blocks(blocks: Iterable[ScoreBlock], names: Optional[Sequence[str]] = None) -> np.ndarray:
    selected = list(blocks)
    if names is not None:
        wanted = set(names)
        selected = [block for block in selected if block.name in wanted]
    if not selected:
        raise ValueError("No score blocks selected for averaging.")
    return np.mean([normalize_scores(block.scores) for block in selected], axis=0).astype(np.float32)


def preds_from_scores(scores: np.ndarray, model_names: List[str]) -> List[str]:
    return [model_names[idx] for idx in scores.argmax(axis=1)]


def evaluate_blocks(
    blocks: List[ScoreBlock],
    reward_val: pd.DataFrame,
    model_names: List[str],
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for block in blocks:
        preds = preds_from_scores(block.scores, model_names)
        rows.append(
            {
                "seed": seed,
                "method": block.name,
                "local_reward": evaluate_reward(reward_val, preds),
                "n_blocks": 1,
            }
        )

    ranked = sorted(rows, key=lambda row: float(row["local_reward"]), reverse=True)
    for top_n in [3, 5, 8, 12]:
        chosen = [str(row["method"]) for row in ranked[: min(top_n, len(ranked))]]
        avg_scores = average_blocks(blocks, chosen)
        preds = preds_from_scores(avg_scores, model_names)
        rows.append(
            {
                "seed": seed,
                "method": f"cv_top{len(chosen)}_avg",
                "local_reward": evaluate_reward(reward_val, preds),
                "n_blocks": len(chosen),
            }
        )
    return rows


def run_cv(
    train_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    model_names: List[str],
    args: argparse.Namespace,
    ks: List[int],
    out_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seeds = parse_ints(args.seeds)
    for seed in seeds:
        fit_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=args.val_size,
            random_state=seed,
            shuffle=True,
        )
        fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        fit_reward = reward_df.iloc[fit_idx].reset_index(drop=True)
        val_reward = reward_df.iloc[val_idx].reset_index(drop=True)
        oracle_preds = val_reward.idxmax(axis=1).tolist()
        rows.append(
            {
                "seed": seed,
                "method": "oracle",
                "local_reward": evaluate_reward(val_reward, oracle_preds),
                "n_blocks": 0,
            }
        )
        blocks = build_score_blocks(fit_df["query"], val_df["query"], fit_reward.to_numpy(dtype=np.float32), args, ks)
        rows.extend(evaluate_blocks(blocks, val_reward, model_names, seed))

    board = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_dir / "local_cv_board_long.csv", index=False)
    summary = (
        board.groupby("method", as_index=False)
        .agg(
            local_reward_mean=("local_reward", "mean"),
            local_reward_std=("local_reward", "std"),
            local_reward_min=("local_reward", "min"),
            local_reward_max=("local_reward", "max"),
            n=("local_reward", "size"),
            n_blocks=("n_blocks", "max"),
        )
        .sort_values("local_reward_mean", ascending=False)
    )
    summary.to_csv(out_dir / "local_cv_board_summary.csv", index=False)
    print("Local CV summary:")
    print(summary.head(20).to_string(index=False))
    return summary


def select_final_methods(summary: Optional[pd.DataFrame], blocks: List[ScoreBlock], limit: int) -> List[str]:
    if summary is None or summary.empty:
        priority_prefixes = ["word12_knn", "charwb35_knn", "char35_knn", "ridge"]
        names = []
        for prefix in priority_prefixes:
            names.extend([block.name for block in blocks if prefix in block.name])
        return list(dict.fromkeys(names))[:limit]

    block_names = {block.name for block in blocks}
    ranked = summary[summary["method"].isin(block_names)].sort_values("local_reward_mean", ascending=False)
    names = ranked["method"].astype(str).head(limit).tolist()
    if len(names) < min(limit, len(blocks)):
        for block in blocks:
            if block.name not in names:
                names.append(block.name)
            if len(names) >= limit:
                break
    return names


def write_candidates(
    test_df: pd.DataFrame,
    model_names: List[str],
    blocks: List[ScoreBlock],
    selected_methods: List[str],
    base_one_hot: np.ndarray,
    base_weights: List[float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    base_preds = preds_from_scores(base_one_hot, model_names)

    candidate_sizes = []
    for top_n in [1, 3, 5, len(selected_methods)]:
        size = min(top_n, len(selected_methods))
        if size > 0 and size not in candidate_sizes:
            candidate_sizes.append(size)

    for top_n in candidate_sizes:
        chosen = selected_methods[: min(top_n, len(selected_methods))]
        if not chosen:
            continue
        reward_scores = average_blocks(blocks, chosen)
        reward_name = f"mv_top{len(chosen)}"
        save_score_npz(out_dir / f"{reward_name}_scores.npz", reward_scores, test_df["ID"], model_names)
        raw_preds = preds_from_scores(reward_scores, model_names)
        raw_path = out_dir / f"submission_{reward_name}.csv"
        make_submission(test_df, raw_preds, raw_path)
        rows.append(
            {
                "candidate": raw_path.name,
                "path": str(raw_path),
                "methods": ",".join(chosen),
                "base_weight": 0.0,
                "changed_vs_base": int(pd.Series(raw_preds).ne(pd.Series(base_preds)).sum()),
            }
        )

        prob_scores = row_softmax(reward_scores)
        for base_weight in base_weights:
            mixed = (1.0 - base_weight) * prob_scores + base_weight * base_one_hot
            preds = preds_from_scores(mixed, model_names)
            out_path = out_dir / f"submission_{reward_name}_base{int(round(base_weight * 100)):02d}.csv"
            make_submission(test_df, preds, out_path)
            rows.append(
                {
                    "candidate": out_path.name,
                    "path": str(out_path),
                    "methods": ",".join(chosen),
                    "base_weight": base_weight,
                    "changed_vs_base": int(pd.Series(preds).ne(pd.Series(base_preds)).sum()),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "candidate_summary.csv", index=False)
    print("Candidate summary:")
    print(pd.DataFrame(rows).to_string(index=False))


def validate_outputs(out_dir: Path, test_df: pd.DataFrame) -> None:
    for path in sorted(out_dir.glob("submission_*.csv")):
        df = pd.read_csv(path)
        if df.columns.tolist() != ["ID", "pred_model"]:
            raise ValueError(f"{path} has bad columns: {df.columns.tolist()}")
        if len(df) != len(test_df):
            raise ValueError(f"{path} has {len(df)} rows, expected {len(test_df)}")
        if not df["ID"].reset_index(drop=True).equals(test_df["ID"].reset_index(drop=True)):
            raise ValueError(f"{path} ID order does not match test.csv")
        if df.isna().sum().sum() > 0:
            raise ValueError(f"{path} contains null values")
    print(f"Validated submission CSV files under {out_dir}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ks = parse_ints(args.knn_k)
    base_weights = parse_floats(args.base_weights)
    if any(weight < 0.0 or weight >= 1.0 for weight in base_weights):
        raise ValueError("--base-weights must be in [0, 1).")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    if args.limit > 0:
        train_df = train_df.head(args.limit).copy()
        test_df = test_df.head(args.limit).copy()
    if "query" not in train_df.columns or "query" not in test_df.columns:
        raise ValueError("train/test must contain a query column.")

    model_names = parse_model_names(train_df.columns.tolist())
    reward_df = compute_reward(train_df, model_names, args.alpha)

    summary = None
    if not args.no_cv:
        summary = run_cv(train_df, reward_df, model_names, args, ks, out_dir)

    full_blocks = build_score_blocks(
        train_df["query"],
        test_df["query"],
        reward_df.to_numpy(dtype=np.float32),
        args,
        ks,
    )
    selected_methods = select_final_methods(summary, full_blocks, args.candidate_top_n)
    pd.DataFrame({"rank": np.arange(1, len(selected_methods) + 1), "method": selected_methods}).to_csv(
        out_dir / "selected_methods.csv",
        index=False,
    )
    print("Selected final methods:")
    for rank, method in enumerate(selected_methods, start=1):
        print(f"  {rank}. {method}")

    base_one_hot = load_base_one_hot(args.base_submission, test_df, model_names)
    write_candidates(test_df, model_names, full_blocks, selected_methods, base_one_hot, base_weights, out_dir)
    validate_outputs(out_dir, test_df)


if __name__ == "__main__":
    main()
