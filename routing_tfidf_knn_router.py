import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


TASK_NAMES = [
    "coding",
    "math",
    "multilingual",
    "science",
    "factual_qa",
    "creative_language",
    "agentic",
    "reasoning",
    "general",
]

CODING_RE = re.compile(
    r"```|<code>|<issue>|traceback|diff --git|\b(def|class|import|function)\b|"
    r"\b(pytest|github|bug|exception|error|leetcode|codeforces|atcoder)\b|"
    r"sample input|sample output|constraints\s*\n",
    re.IGNORECASE | re.MULTILINE,
)
MATH_RE = re.compile(
    r"\\frac|\\sqrt|\\sum|\\int|\$|\b(prove|solve|equation|integer|prime|"
    r"probability|triangle|matrix|derivative|integral|geometry|algebra)\b",
    re.IGNORECASE,
)
MULTILINGUAL_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
SCIENCE_RE = re.compile(
    r"\b(physics|chemistry|biology|molecule|protein|gene|cell|disease|"
    r"reaction|velocity|force|pressure|temperature|organism|experiment)\b",
    re.IGNORECASE,
)
FACTUAL_QA_RE = re.compile(
    r"^\s*(who|what|when|where|which|how many|how much|in what|why)\b",
    re.IGNORECASE,
)
CREATIVE_RE = re.compile(
    r"\b(write|rewrite|summarize|summary|story|poem|haiku|creative|compose|"
    r"translate|tone|style|email|essay)\b",
    re.IGNORECASE,
)
AGENTIC_RE = re.compile(
    r"\b(order|cancel|refund|return|tracking|booking|reservation|profile|"
    r"cart|customer|agent|tool|api call|function call)\b",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"\b(riddle|logic puzzle|deduce|infer|correct order|arrange|constraint|"
    r"reason step by step|which statement)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight TF-IDF kNN LLM router.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_tfidf_knn.csv")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--knn-weight", type=float, default=0.70)
    parser.add_argument("--prior-smoothing", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--score-out", default="", help="Optional .npz path for test reward scores.")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument(
        "--cv-seeds",
        default="",
        help="Comma-separated CV seeds. Overrides --cv-repeats when set.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_text_column(df: pd.DataFrame) -> str:
    for col in ["query", "prompt", "Question", "question", "text"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a text column. Expected one of: query, prompt, Question, question, text.")


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


def compute_reward(train_df: pd.DataFrame, models: List[str], alpha: float) -> pd.DataFrame:
    perf = train_df[[f"{model}_performance" for model in models]].copy()
    cost = train_df[[f"{model}_cost" for model in models]].copy()
    perf.columns = models
    cost.columns = models

    global_cmax = float(cost.max().max())
    if global_cmax <= 0:
        global_cmax = 1.0
    reward = alpha * perf - (1.0 - alpha) * (cost / global_cmax)
    return reward


def infer_task(text: str) -> str:
    text = str(text)
    lowered = text.lower()

    if CODING_RE.search(text):
        return "coding"
    if MATH_RE.search(text):
        return "math"
    if MULTILINGUAL_RE.search(text):
        return "multilingual"
    if SCIENCE_RE.search(text):
        return "science"
    if FACTUAL_QA_RE.search(text):
        return "factual_qa"
    if CREATIVE_RE.search(text):
        return "creative_language"
    if AGENTIC_RE.search(text):
        return "agentic"
    if REASONING_RE.search(text) or ("step by step" in lowered and "explain" not in lowered):
        return "reasoning"
    return "general"


def infer_tasks(texts: pd.Series) -> np.ndarray:
    return np.array([infer_task(text) for text in texts.astype(str).tolist()], dtype=object)


def build_prior_table(
    train_tasks: np.ndarray,
    reward: np.ndarray,
    smoothing: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    global_prior = reward.mean(axis=0).astype(np.float32)
    priors: Dict[str, np.ndarray] = {}

    for task in TASK_NAMES:
        mask = train_tasks == task
        count = int(mask.sum())
        if count == 0:
            priors[task] = global_prior
            continue
        task_sum = reward[mask].sum(axis=0)
        priors[task] = ((task_sum + smoothing * global_prior) / (count + smoothing)).astype(np.float32)

    return priors, global_prior


def lookup_priors(tasks: np.ndarray, priors: Dict[str, np.ndarray], global_prior: np.ndarray) -> np.ndarray:
    return np.vstack([priors.get(str(task), global_prior) for task in tasks]).astype(np.float32)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=200000,
        sublinear_tf=True,
        strip_accents="unicode",
        norm="l2",
    )


def knn_reward_scores(
    train_texts: pd.Series,
    query_texts: pd.Series,
    reward: np.ndarray,
    k: int,
    batch_size: int,
) -> np.ndarray:
    vectorizer = make_vectorizer()
    x_train = vectorizer.fit_transform(train_texts.astype(str))
    x_query = vectorizer.transform(query_texts.astype(str))

    n_query = x_query.shape[0]
    n_train = x_train.shape[0]
    k = max(1, min(k, n_train))
    scores = np.zeros((n_query, reward.shape[1]), dtype=np.float32)
    reward = reward.astype(np.float32, copy=False)

    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = (x_query[start:end] @ x_train.T).toarray().astype(np.float32, copy=False)

        if k < n_train:
            top_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        else:
            top_idx = np.tile(np.arange(n_train), (end - start, 1))

        top_sim = np.take_along_axis(sim, top_idx, axis=1)
        denom = top_sim.sum(axis=1, keepdims=True)
        weighted = np.einsum("bk,bkm->bm", top_sim, reward[top_idx], optimize=True)

        valid = denom[:, 0] > 1e-12
        scores[start:end][valid] = weighted[valid] / denom[valid]
        if np.any(~valid):
            scores[start:end][~valid] = reward.mean(axis=0)

    return scores


def router_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    text_col: str,
    k: int,
    knn_weight: float,
    prior_smoothing: float,
    batch_size: int,
) -> List[str]:
    reward = reward_df.to_numpy(dtype=np.float32)
    train_tasks = infer_tasks(train_df[text_col])
    test_tasks = infer_tasks(test_df[text_col])
    priors, global_prior = build_prior_table(train_tasks, reward, prior_smoothing)

    knn_scores = knn_reward_scores(train_df[text_col], test_df[text_col], reward, k, batch_size)
    prior_scores = lookup_priors(test_tasks, priors, global_prior)
    return (knn_weight * knn_scores + (1.0 - knn_weight) * prior_scores).astype(np.float32)


def predict_router(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    text_col: str,
    k: int,
    knn_weight: float,
    prior_smoothing: float,
    batch_size: int,
) -> List[str]:
    final_scores = router_scores(
        train_df,
        test_df,
        reward_df,
        text_col,
        k,
        knn_weight,
        prior_smoothing,
        batch_size,
    )
    best_idx = final_scores.argmax(axis=1)
    return [reward_df.columns[idx] for idx in best_idx]


def evaluate_reward(reward_df: pd.DataFrame, preds: List[str]) -> float:
    row_idx = np.arange(len(preds))
    col_idx = reward_df.columns.get_indexer(preds)
    return float(reward_df.to_numpy(dtype=np.float32)[row_idx, col_idx].mean())


def parse_cv_seeds(raw: str, seed: int, repeats: int) -> List[int]:
    if raw.strip():
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    repeats = max(1, repeats)
    return [seed + offset for offset in range(repeats)]


def run_one_cv_split(
    train_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> float:
    text_col = find_text_column(train_df)
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

    preds = predict_router(
        fit_df,
        val_df,
        fit_reward,
        text_col,
        args.k,
        args.knn_weight,
        args.prior_smoothing,
        args.batch_size,
    )
    return evaluate_reward(val_reward, preds)


def run_cv(train_df: pd.DataFrame, reward_df: pd.DataFrame, args: argparse.Namespace) -> None:
    seeds = parse_cv_seeds(args.cv_seeds, args.seed, args.cv_repeats)
    scores = []
    for seed in seeds:
        score = run_one_cv_split(train_df, reward_df, args, seed)
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


def write_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
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


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train)
    models = parse_model_names(train_df.columns.tolist())
    reward_df = compute_reward(train_df, models, args.alpha)

    if args.cv:
        run_cv(train_df, reward_df, args)
        return

    test_df = pd.read_csv(args.test)
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
    preds = [reward_df.columns[idx] for idx in scores.argmax(axis=1)]
    write_submission(test_df, preds, args.out)
    if args.score_out:
        save_score_npz(args.score_out, scores, test_df["ID"], list(reward_df.columns))
        print(f"Wrote {args.score_out} with score shape {scores.shape}.")
    print(f"Wrote {args.out} with {len(preds)} rows.")


if __name__ == "__main__":
    main()
