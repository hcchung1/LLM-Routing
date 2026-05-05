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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


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


def compute_reward(perf: pd.DataFrame, cost: pd.DataFrame, cfg: RewardConfig) -> pd.DataFrame:
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


def build_prompt(query: str, model_names: List[str]) -> str:
    model_list = ", ".join(model_names)
    return (
        "You are a router that selects the best model for a user query. "
        "Choose exactly one model from the list.\n\n"
        f"Models: {model_list}\n\n"
        f"Query: {query}\n\n"
        "Answer with exactly one label (Model_A to Model_K).\n"
        "Label:"
    )


def truncate_prompt_ids(prompt_ids: List[int], label_ids: List[int], max_length: int) -> List[int]:
    reserve = len(label_ids) + 1
    if max_length <= reserve:
        return prompt_ids[:1]
    allowed = max_length - reserve
    if len(prompt_ids) <= allowed:
        return prompt_ids
    return prompt_ids[:allowed]


def build_causal_example(
    query: str,
    label: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    model_names: List[str],
) -> Dict[str, List[int]]:
    prompt = build_prompt(query, model_names)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    label_ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    prompt_ids = truncate_prompt_ids(prompt_ids, label_ids, max_length)
    input_ids = prompt_ids + label_ids
    if tokenizer.eos_token_id is not None:
        input_ids = input_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + label_ids
    if tokenizer.eos_token_id is not None:
        labels = labels + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_infer_prompt_ids(
    query: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    model_names: List[str],
) -> List[int]:
    prompt = build_prompt(query, model_names)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return prompt_ids[:max_length]


class CausalDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        def to_list(val: List[int] | torch.Tensor) -> List[int]:
            if torch.is_tensor(val):
                return val.tolist()
            return val

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            f_input_ids = to_list(f["input_ids"])
            f_attention_mask = to_list(f["attention_mask"])
            f_labels = to_list(f["labels"])
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f_input_ids + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f_attention_mask + [0] * pad_len)
            labels.append(f_labels + [-100] * pad_len)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


def score_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids_batch: List[List[int]],
    label_token_ids: Dict[str, List[int]],
    max_length: int,
) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    labels = list(label_token_ids.keys())

    sequences = []
    meta = []
    for q_idx, prompt_ids in enumerate(prompt_ids_batch):
        for label in labels:
            label_ids = label_token_ids[label]
            trunc_prompt = truncate_prompt_ids(prompt_ids, label_ids, max_length)
            input_ids = trunc_prompt + label_ids
            sequences.append(input_ids)
            meta.append((q_idx, label, len(trunc_prompt)))

    max_len = max(len(seq) for seq in sequences)
    input_ids = []
    attention_mask = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        input_ids.append(seq + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    scores = {}
    for idx, (q_idx, label, prompt_len) in enumerate(meta):
        seq = sequences[idx]
        label_len = len(seq) - prompt_len
        score = 0.0
        for pos in range(prompt_len, len(seq)):
            token_id = seq[pos]
            score += float(log_probs[idx, pos - 1, token_id].item())
        score = score / max(1, label_len)
        scores.setdefault(q_idx, {})[label] = score

    preds = []
    for q_idx in range(len(prompt_ids_batch)):
        best_label = max(scores[q_idx].items(), key=lambda kv: kv[1])[0]
        preds.append(best_label)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_gemma.csv")
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--cost-norm",
        default="row-max",
        choices=["row-max", "none", "minmax-global", "minmax-per-model", "zscore-global"],
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        logger.info("HF token detected from environment")
    else:
        logger.warning("HF token not found; downloads may be rate-limited")

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

    train_df = train_df[["query"]].copy()
    train_df["label"] = labels
    train_split, val_split = split_train_val(train_df, args.val_ratio, args.seed)

    logger.info("Loading tokenizer: {}", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Building train/val datasets")
    train_rows = []
    for row in tqdm(train_split.itertuples(index=False), total=len(train_split), desc="Train rows"):
        train_rows.append(
            build_causal_example(
                row.query,
                row.label,
                tokenizer,
                max_length=args.max_length,
                model_names=models,
            )
        )
    val_rows = []
    for row in tqdm(val_split.itertuples(index=False), total=len(val_split), desc="Val rows"):
        val_rows.append(
            build_causal_example(
                row.query,
                row.label,
                tokenizer,
                max_length=args.max_length,
                model_names=models,
            )
        )

    train_ds = Dataset.from_list(train_rows).with_format("torch")
    val_ds = Dataset.from_list(val_rows).with_format("torch")

    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    logger.info("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    use_fp16 = args.fp16 and torch.cuda.is_available()
    training_args_kwargs = dict(
        output_dir="./gemma_judge_runs",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        fp16=use_fp16,
        remove_unused_columns=False,
    )
    training_args_params = inspect.signature(TrainingArguments).parameters
    if "evaluation_strategy" in training_args_params:
        training_args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CausalDataCollator(tokenizer),
    )

    logger.info("Training start")
    trainer.train()

    logger.info("Scoring test set")
    label_token_ids = {
        label: tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        for label in models
    }
    test_prompts = [
        build_infer_prompt_ids(q, tokenizer, args.max_length, model_names=models)
        for q in test_df["query"].tolist()
    ]

    preds = []
    batch = 4
    for i in tqdm(range(0, len(test_prompts), batch), desc="Scoring"):
        batch_prompts = test_prompts[i : i + batch]
        preds.extend(
            score_batch(
                model,
                tokenizer,
                batch_prompts,
                label_token_ids,
                max_length=args.max_length,
            )
        )

    make_submission(test_df, preds, args.out)
    logger.info("Saved submission to {}", args.out)


if __name__ == "__main__":
    main()
