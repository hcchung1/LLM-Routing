import argparse
import inspect
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Value
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class RewardConfig:
    alpha: float = 0.85
    cost_norm: str = "global-max"


def parse_model_names(columns: List[str]) -> List[str]:
    models = []
    for col in columns:
        if col.endswith("_performance") and col.startswith("Model_"):
            models.append(col.replace("_performance", ""))
    return sorted(models)


def normalize_cost(cost: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "global-max":
        denom = float(cost.max().max())
        return cost / (denom if denom > 0 else 1.0)
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


def compute_reward(perf: pd.DataFrame, cost: pd.DataFrame, cfg: RewardConfig) -> pd.DataFrame:
    # Kaggle Reward_alpha uses the global maximum cost as the cost denominator.
    cost_n = normalize_cost(cost, cfg.cost_norm)
    alpha = cfg.alpha
    return alpha * perf - (1.0 - alpha) * cost_n


def build_labels(reward: pd.DataFrame) -> pd.Series:
    return reward.idxmax(axis=1)


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def make_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
    sub = pd.DataFrame({"ID": test_df["ID"], "pred_model": preds})
    sub.to_csv(out_path, index=False)


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    val_size = max(1, int(len(df) * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def batch_tokenize(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> Dict[str, List[List[int]]]:
    model_inputs = {name: [] for name in tokenizer.model_input_names}
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        output = tokenizer(
            batch,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        for name in model_inputs:
            if name in output:
                model_inputs[name].extend(output[name])
    return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-small")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--cost-norm",
        default="global-max",
        choices=["global-max", "row-max", "none", "minmax-global", "minmax-per-model", "zscore-global"],
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
    )
    args = parser.parse_args()

    logger.info("Loading data")
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

    label2id: Dict[str, int] = {m: i for i, m in enumerate(models)}
    id2label: Dict[int, str] = {i: m for m, i in label2id.items()}

    label_ids = labels.map(label2id)
    if label_ids.isna().any():
        missing = label_ids.isna().sum()
        raise ValueError(f"Found {missing} unmapped labels.")
    train_df = train_df[["query"]].copy()
    train_df["labels"] = label_ids.astype(int)
    logger.info(
        "Label stats: min={}, max={}, classes={}",
        int(train_df["labels"].min()),
        int(train_df["labels"].max()),
        train_df["labels"].nunique(),
    )
    train_split, val_split = split_train_val(train_df, args.val_ratio, args.seed)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        logger.info("HF token detected from environment")
    else:
        logger.warning("HF token not found; downloads may be rate-limited")

    logger.info("Loading tokenizer: {}", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)

    logger.info("Tokenizing train split")
    train_tokens = batch_tokenize(
        train_split["query"].tolist(),
        tokenizer,
        max_length=args.max_length,
        batch_size=256,
        desc="Tokenizing train",
    )
    train_tokens["labels"] = train_split["labels"].tolist()

    logger.info("Tokenizing val split")
    val_tokens = batch_tokenize(
        val_split["query"].tolist(),
        tokenizer,
        max_length=args.max_length,
        batch_size=256,
        desc="Tokenizing val",
    )
    val_tokens["labels"] = val_split["labels"].tolist()

    logger.info("Tokenizing test set")
    test_tokens = batch_tokenize(
        test_df["query"].tolist(),
        tokenizer,
        max_length=args.max_length,
        batch_size=256,
        desc="Tokenizing test",
    )

    train_ds = Dataset.from_dict(train_tokens).cast_column("labels", Value("int64")).with_format("torch")
    val_ds = Dataset.from_dict(val_tokens).cast_column("labels", Value("int64")).with_format("torch")
    test_ds = Dataset.from_dict(test_tokens).with_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    precision = args.precision
    if args.fp16:
        logger.warning("`--fp16` is deprecated; use `--precision fp16` to force.")
        precision = "fp16"

    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            precision = "fp32"

    use_fp16 = precision == "fp16" and torch.cuda.is_available()
    use_bf16 = precision == "bf16" and torch.cuda.is_available()
    if precision == "fp16" and not torch.cuda.is_available():
        logger.warning("FP16 requested but CUDA is unavailable; falling back to fp32.")
    if precision == "bf16" and not torch.cuda.is_available():
        logger.warning("BF16 requested but CUDA is unavailable; falling back to fp32.")

    logger.info("Loading model")
    torch_dtype = torch.float32 if precision == "fp32" else None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(models),
            id2label=id2label,
            label2id=label2id,
            use_safetensors=True,
            dtype=torch_dtype,
            token=hf_token,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(models),
            id2label=id2label,
            label2id=label2id,
            use_safetensors=True,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model params: total={} ({:.2f}M), trainable={} ({:.2f}M)",
        total_params,
        total_params / 1e6,
        trainable_params,
        trainable_params / 1e6,
    )
    logger.info("Training precision: {}", precision)
    ta_kwargs = dict(
        output_dir="./transformer_runs",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        fp16=use_fp16,
    )
    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        ta_kwargs["evaluation_strategy"] = "epoch"
    if "eval_strategy" in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in ta_sig.parameters:
        ta_kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in ta_sig.parameters:
        ta_kwargs["logging_strategy"] = "epoch"
    if "logging_steps" in ta_sig.parameters:
        ta_kwargs["logging_steps"] = 10**9
    if "remove_unused_columns" in ta_sig.parameters:
        ta_kwargs["remove_unused_columns"] = False
    if "label_names" in ta_sig.parameters:
        ta_kwargs["label_names"] = ["labels"]
    training_args = TrainingArguments(**ta_kwargs)
    if "bf16" in ta_sig.parameters:
        training_args.bf16 = use_bf16

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    logger.info("Training start")
    trainer.train()

    logger.info("Predicting test set")
    preds = trainer.predict(test_ds)
    pred_ids = np.argmax(preds.predictions, axis=1)
    pred_labels = [id2label[i] for i in pred_ids]
    make_submission(test_df, pred_labels, args.out)
    logger.info("Saved submission to {}", args.out)


if __name__ == "__main__":
    main()
