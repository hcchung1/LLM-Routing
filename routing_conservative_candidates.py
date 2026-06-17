import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SOURCES: List[Tuple[str, str, List[int]]] = [
    ("lb_top4", "output/leaderboard_vote_top4_scores.npz", [5, 10, 22]),
    ("lb_top5", "output/leaderboard_vote_top5_scores.npz", [5, 10, 20, 40]),
    ("pairwise", "output/pairwise_ranker_scores.npz", [5, 10, 20]),
    ("lora_gbm", "output/lora_gbm_final_scores.npz", [5, 10, 20]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conservative override candidates from the best known submission."
    )
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--base-submission", default="output/submission_starcoder2_qlora.csv")
    parser.add_argument("--out-dir", default="output/conservative_candidates")
    parser.add_argument("--prefix", default="submission_conservative")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="Optional source specs: label:path:k1,k2. Defaults to known score files.",
    )
    parser.add_argument(
        "--summary-out",
        default="output/conservative_candidates/summary.csv",
        help="CSV summary of generated candidates.",
    )
    return parser.parse_args()


def parse_sources(raw_sources: List[str]) -> List[Tuple[str, str, List[int]]]:
    if not raw_sources:
        return DEFAULT_SOURCES

    parsed = []
    for spec in raw_sources:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad source spec {spec!r}; expected label:path:k1,k2.")
        label, path, counts_raw = parts
        counts = [int(value) for value in counts_raw.split(",") if value.strip()]
        if not label:
            raise ValueError(f"Bad source spec {spec!r}; label is empty.")
        if not counts or any(count <= 0 for count in counts):
            raise ValueError(f"Bad source spec {spec!r}; override counts must be positive.")
        parsed.append((label, path, counts))
    return parsed


def load_submission(path: str, test_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = ["ID", "pred_model"]
    if df.columns.tolist() != expected_columns:
        raise ValueError(f"{path} columns must be exactly {expected_columns}; got {df.columns.tolist()}.")
    if len(df) != len(test_df):
        raise ValueError(f"{path} has {len(df)} rows, expected {len(test_df)}.")
    if not df["ID"].reset_index(drop=True).equals(test_df["ID"].reset_index(drop=True)):
        raise ValueError(f"{path} ID order does not match test.csv.")
    return df.copy()


def load_scores(path: str, test_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    score_path = Path(path)
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    data = np.load(score_path, allow_pickle=True)
    if "scores" not in data or "model_names" not in data:
        raise ValueError(f"{path} must contain `scores` and `model_names` arrays.")

    scores = np.asarray(data["scores"], dtype=np.float32)
    model_names = [str(value) for value in data["model_names"]]
    if scores.ndim != 2:
        raise ValueError(f"{path} scores must be 2D; got {scores.shape}.")
    if scores.shape[0] != len(test_df):
        raise ValueError(f"{path} has {scores.shape[0]} score rows, expected {len(test_df)}.")
    if scores.shape[1] != len(model_names):
        raise ValueError(f"{path} has {scores.shape[1]} score columns, expected {len(model_names)}.")
    if "ids" in data and not pd.Series(data["ids"]).reset_index(drop=True).equals(
        test_df["ID"].reset_index(drop=True)
    ):
        raise ValueError(f"{path} ID order does not match test.csv.")
    return scores, model_names


def top_override_candidates(
    base_preds: pd.Series,
    scores: np.ndarray,
    model_names: List[str],
) -> pd.DataFrame:
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    missing = sorted({str(value) for value in base_preds.unique().tolist() if str(value) not in model_to_idx})
    if missing:
        raise ValueError(f"Base submission contains labels missing from score columns: {missing}")

    top_idx = scores.argmax(axis=1)
    top_scores = scores[np.arange(len(scores)), top_idx]
    second_scores = np.partition(scores, -2, axis=1)[:, -2] if scores.shape[1] > 1 else np.zeros(len(scores))
    base_idx = np.array([model_to_idx[str(value)] for value in base_preds.tolist()], dtype=np.int64)
    base_scores = scores[np.arange(len(scores)), base_idx]

    rows = pd.DataFrame(
        {
            "row_idx": np.arange(len(scores), dtype=np.int64),
            "base_pred": base_preds.astype(str).to_numpy(),
            "score_pred": [model_names[idx] for idx in top_idx],
            "margin_vs_base": (top_scores - base_scores).astype(np.float32),
            "top_gap": (top_scores - second_scores).astype(np.float32),
        }
    )
    rows = rows[rows["base_pred"] != rows["score_pred"]].copy()
    rows = rows.sort_values(["margin_vs_base", "top_gap"], ascending=False).reset_index(drop=True)
    return rows


def write_candidate(
    test_df: pd.DataFrame,
    base_sub: pd.DataFrame,
    candidates: pd.DataFrame,
    count: int,
    out_path: Path,
    debug_path: Path,
) -> Dict[str, object]:
    selected = candidates.head(count).copy()
    final = base_sub.copy()
    for _, row in selected.iterrows():
        final.loc[int(row["row_idx"]), "pred_model"] = row["score_pred"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    debug = pd.concat(
        [test_df[["ID"]].reset_index(drop=True), base_sub[["pred_model"]].rename(columns={"pred_model": "final_pred"})],
        axis=1,
    )
    debug["changed"] = final["pred_model"].ne(base_sub["pred_model"]).to_numpy()
    debug["final_pred"] = final["pred_model"].to_numpy()
    if not selected.empty:
        selected_debug = selected.copy()
        selected_debug["ID"] = test_df.loc[selected_debug["row_idx"], "ID"].to_numpy()
        selected_debug.to_csv(debug_path, index=False)

    transitions = {}
    for _, row in selected.iterrows():
        key = f"{row['base_pred']}->{row['score_pred']}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "file_name": out_path.name,
        "path": str(out_path),
        "debug_path": str(debug_path) if not selected.empty else "",
        "requested_overrides": count,
        "actual_overrides": int(final["pred_model"].ne(base_sub["pred_model"]).sum()),
        "transition_summary": "; ".join(f"{key}:{value}" for key, value in sorted(transitions.items())),
    }


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test)
    base_sub = load_submission(args.base_submission, test_df)
    out_dir = Path(args.out_dir)
    sources = parse_sources(args.sources)

    summary_rows: List[Dict[str, object]] = []
    for label, score_path, counts in sources:
        scores, model_names = load_scores(score_path, test_df)
        candidates = top_override_candidates(base_sub["pred_model"], scores, model_names)
        print(f"{label}: {len(candidates)} possible overrides from {score_path}")
        for count in counts:
            safe_count = min(count, len(candidates))
            out_path = out_dir / f"{args.prefix}_{label}_top{safe_count}.csv"
            debug_path = out_dir / f"{args.prefix}_{label}_top{safe_count}_debug.csv"
            row = write_candidate(test_df, base_sub, candidates, safe_count, out_path, debug_path)
            row["source"] = label
            row["score_path"] = score_path
            summary_rows.append(row)
            print(f"  wrote {out_path} with {row['actual_overrides']} overrides")

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
