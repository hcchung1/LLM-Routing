import argparse
import gc
import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class RewardConfig:
    alpha: float = 0.85
    cost_norm: str = "mean-row-max"


@dataclass(frozen=True)
class TaskProfile:
    domain: str
    subtask: str
    difficulty: str

    @property
    def profile_key(self) -> str:
        return f"{self.domain}/{self.subtask}/{self.difficulty}"

    @property
    def subtask_key(self) -> str:
        return f"{self.domain}/{self.subtask}"


TaskPriorTable = Dict[str, Dict[str, Tuple[int, np.ndarray]]]


REPO_CODING_RE = re.compile(
    r"<issue>|<code>|partial code base|traceback|\bdiff --git\b|\bpytest\b|"
    r"\bgithub\b|\bbug\b|\bregression\b|\bexception\b|\berror\b",
    re.IGNORECASE | re.MULTILINE,
)
COMPETITIVE_CODING_RE = re.compile(
    r"Sample Input|Sample Output|\bInput\s*\n|\bOutput\s*\n|Constraints\s*\n|"
    r"Standard Input|AtCoder|Codeforces",
    re.IGNORECASE | re.MULTILINE,
)
CODE_SNIPPET_RE = re.compile(
    r"```|\bdef\s+\w+\s*\(|\bclass\s+\w+|\bimport\s+\w+|"
    r"\bpublic\s+class\b|\bfunction\s+\w+",
    re.IGNORECASE | re.MULTILINE,
)
MULTILINGUAL_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
MATH_RE = re.compile(
    r"\\frac|\\sqrt|\$|\bFind\b|\bLet\b|\bprobability\b|\binteger\b|"
    r"\btriangle\b|\bparabola\b|\bprime\b|\bdivisor\b|\\log_",
    re.IGNORECASE,
)
FACTUAL_QA_RE = re.compile(
    r"^(Who|What|When|Where|Which|How many|In what|The .* called)\b",
    re.IGNORECASE,
)


def contains_any(text: str, terms: List[str]) -> bool:
    return any(term in text for term in terms)


def infer_task_profile(query: str) -> TaskProfile:
    """Approximate TRouter's domain/subtask/difficulty task profile locally."""
    text = str(query)
    lowered = text.lower()

    if re.search(r"^\s*ARC-AGI Task\b", text, re.IGNORECASE):
        domain, subtask = "arc_agi", "grid_abstraction"
    elif REPO_CODING_RE.search(text):
        domain, subtask = "coding", "repository_issue"
    elif COMPETITIVE_CODING_RE.search(text):
        domain, subtask = "coding", "competitive_programming"
    elif CODE_SNIPPET_RE.search(text):
        domain, subtask = "coding", "code_generation"
    elif MULTILINGUAL_RE.search(text):
        if contains_any(lowered, ["haiku", "poem", "rewrite", "translate", "summarize"]):
            domain, subtask = "creative_language", "multilingual_writing"
        else:
            domain, subtask = "multilingual", "non_english"
    elif MATH_RE.search(text):
        domain, subtask = "math", "symbolic_reasoning"
    elif contains_any(
        lowered,
        [
            "physics",
            "force",
            "velocity",
            "pressure",
            "molecule",
            "protein",
            "e. coli",
            "gene",
            "cell",
            "disease",
            "chemical",
            "co_2",
            "resistance",
        ],
    ):
        domain, subtask = "science", "science_qa"
    elif FACTUAL_QA_RE.search(text.strip()):
        domain, subtask = "factual_qa", "short_answer"
    elif contains_any(lowered, ["summarize", "summary", "article", "rewrite", "write a", "story", "poem", "haiku", "creative"]):
        domain, subtask = "creative_language", "writing_summarization"
    elif contains_any(lowered, ["order", "cancel", "return", "refund", "tracking", "profile", "agent correctly handles"]):
        domain, subtask = "agentic", "instruction_following"
    elif contains_any(lowered, ["riddle", "correct order", "hint:", "logic puzzle"]):
        domain, subtask = "reasoning", "puzzle"
    else:
        domain, subtask = "general", "general"

    length = len(text)
    if length >= 2500:
        difficulty = "hard"
    elif length >= 700:
        difficulty = "medium"
    else:
        difficulty = "easy"

    return TaskProfile(domain=domain, subtask=subtask, difficulty=difficulty)


def infer_task_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiles = [infer_task_profile(query) for query in df["query"].astype(str).tolist()]
    return pd.DataFrame(
        {
            "task_domain": [profile.domain for profile in profiles],
            "task_subtask": [profile.subtask for profile in profiles],
            "task_difficulty": [profile.difficulty for profile in profiles],
            "task_profile": [profile.profile_key for profile in profiles],
            "task_subtask_key": [profile.subtask_key for profile in profiles],
        }
    )


def log_task_summary(name: str, profiles: pd.DataFrame) -> None:
    if profiles.empty:
        return
    counts = profiles["task_subtask_key"].value_counts().head(12)
    formatted = ", ".join(f"{key}={count}" for key, count in counts.items())
    logger.info("{} task profiles: {}", name, formatted)


def build_task_reward_priors(
    profiles: pd.DataFrame,
    reward: pd.DataFrame,
    smoothing: float,
) -> TaskPriorTable:
    global_prior = reward.mean(axis=0).to_numpy(dtype=np.float32)
    stats: TaskPriorTable = {
        "profile": {},
        "subtask": {},
        "domain": {},
        "global": {"__global__": (len(reward), global_prior)},
    }
    group_specs = {
        "profile": profiles["task_profile"],
        "subtask": profiles["task_subtask_key"],
        "domain": profiles["task_domain"],
    }
    for level, keys in group_specs.items():
        for key, idx in keys.groupby(keys).groups.items():
            positions = list(idx)
            count = len(positions)
            mean_scores = reward.iloc[positions].mean(axis=0).to_numpy(dtype=np.float32)
            prior_scores = (count * mean_scores + smoothing * global_prior) / (count + smoothing)
            stats[level][str(key)] = (count, prior_scores.astype(np.float32))
    return stats


def task_prior_scores(
    profiles: pd.DataFrame,
    priors: TaskPriorTable,
    min_count: int,
) -> np.ndarray:
    global_scores = priors["global"]["__global__"][1]
    rows = []
    for _, row in profiles.iterrows():
        candidates = (
            ("profile", row["task_profile"]),
            ("subtask", row["task_subtask_key"]),
            ("domain", row["task_domain"]),
        )
        selected = global_scores
        for level, key in candidates:
            count_and_scores = priors.get(level, {}).get(str(key))
            if count_and_scores is not None and count_and_scores[0] >= min_count:
                selected = count_and_scores[1]
                break
        rows.append(selected)
    return np.vstack(rows).astype(np.float32) if rows else np.empty((0, len(global_scores)), dtype=np.float32)


def patch_bnb_params4bit() -> None:
    """Compatibility shim for older bitsandbytes/transformers combinations."""
    try:
        from bitsandbytes.nn import Params4bit
    except Exception:
        return

    try:
        sig = inspect.signature(Params4bit.__new__)
    except (TypeError, ValueError):
        return

    if "_is_hf_initialized" in sig.parameters:
        return

    orig_new = Params4bit.__new__

    def _new(cls, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return orig_new(cls, *args, **kwargs)

    Params4bit.__new__ = staticmethod(_new)


def parse_model_names(columns: List[str]) -> List[str]:
    models = []
    for col in columns:
        if col.startswith("Model_") and col.endswith("_performance"):
            models.append(col.replace("_performance", ""))
    return sorted(models)


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def extract_perf_cost(train_df: pd.DataFrame, models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf_cols = [f"{m}_performance" for m in models]
    cost_cols = [f"{m}_cost" for m in models]
    perf = train_df[perf_cols].copy()
    cost = train_df[cost_cols].copy()
    perf.columns = models
    cost.columns = models
    return perf, cost


def cost_denominator(cost: pd.DataFrame, method: str) -> float:
    if method == "mean-row-max":
        denom = float(cost.max(axis=1).mean())
    elif method == "global-max":
        denom = float(cost.max().max())
    elif method == "none":
        denom = 1.0
    else:
        raise ValueError(f"Unknown global denominator method: {method}")
    return denom if denom != 0 else 1.0


def normalize_cost(cost: pd.DataFrame, cfg: RewardConfig) -> pd.DataFrame:
    if cfg.cost_norm == "row-max":
        row_max = cost.max(axis=1).replace(0, 1.0)
        return cost.div(row_max, axis=0)
    if cfg.cost_norm in {"mean-row-max", "global-max", "none"}:
        return cost / cost_denominator(cost, cfg.cost_norm)
    raise ValueError(f"Unknown cost_norm: {cfg.cost_norm}")


def compute_reward(perf: pd.DataFrame, cost: pd.DataFrame, cfg: RewardConfig) -> pd.DataFrame:
    # For Reward_alpha = alpha * mean(P) - (1 - alpha) * mean(C) / mean(C_max),
    # per-row routing is equivalent to maximizing alpha * P_i,m - beta * C_i,m / D.
    return cfg.alpha * perf - (1.0 - cfg.alpha) * normalize_cost(cost, cfg)


def labels_from_reward(reward: pd.DataFrame) -> pd.Series:
    return reward.idxmax(axis=1)


def evaluate_policy_reward(
    perf: pd.DataFrame,
    cost: pd.DataFrame,
    pred_models: List[str],
    cfg: RewardConfig,
) -> float:
    col_idx = perf.columns.get_indexer(pred_models)
    row_idx = np.arange(len(pred_models))
    selected_perf = perf.to_numpy(dtype=np.float32)[row_idx, col_idx]
    selected_cost = cost.to_numpy(dtype=np.float32)[row_idx, col_idx]

    if cfg.cost_norm == "row-max":
        denom = cost.max(axis=1).replace(0, 1.0).to_numpy(dtype=np.float32)
        cost_term = np.mean(selected_cost / denom)
    elif cfg.cost_norm in {"mean-row-max", "global-max", "none"}:
        denom = cost_denominator(cost, cfg.cost_norm)
        cost_term = float(np.mean(selected_cost) / denom)
    else:
        raise ValueError(f"Unknown cost_norm: {cfg.cost_norm}")

    return float(cfg.alpha * np.mean(selected_perf) - (1.0 - cfg.alpha) * cost_term)


def split_train_val_indices(n_rows: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    val_size = max(1, int(n_rows * val_ratio)) if val_ratio > 0 else 0
    if val_size == 0:
        return idx, np.array([], dtype=np.int64)
    return idx[val_size:], idx[:val_size]


def batch_tokenize(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> Dict[str, List[List[int]]]:
    names = tokenizer.model_input_names
    encoded = {name: [] for name in names}
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start : start + batch_size]
        output = tokenizer(
            batch,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        for name in names:
            if name in output:
                encoded[name].extend(output[name])
    return encoded


def build_regression_dataset(
    df: pd.DataFrame,
    target: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
    tokenize_batch_size: int,
    desc: str,
) -> Dataset:
    tokens = batch_tokenize(
        df["query"].astype(str).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=tokenize_batch_size,
        desc=desc,
    )
    tokens["labels"] = target.to_numpy(dtype=np.float32).tolist()
    return Dataset.from_dict(tokens).with_format("torch")


def build_inference_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
    tokenize_batch_size: int,
    desc: str,
) -> Dataset:
    tokens = batch_tokenize(
        df["query"].astype(str).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=tokenize_batch_size,
        desc=desc,
    )
    return Dataset.from_dict(tokens).with_format("torch")


class RegressionDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = None
        if "labels" in features[0]:
            labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        if labels is not None:
            if torch.is_tensor(labels[0]):
                batch["labels"] = torch.stack([label.float() for label in labels])
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def make_submission(test_df: pd.DataFrame, preds: List[str], out_path: str) -> None:
    sub = pd.DataFrame({"ID": test_df["ID"], "pred_model": preds})
    sub.to_csv(out_path, index=False)


def get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def load_tokenizer(model_name: str, hf_token: Optional[str]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_bnb_config(args) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit:
        return None
    patch_bnb_params4bit()
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def parse_target_modules(raw: str) -> List[str]:
    if raw == "attention":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if raw == "all":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_training_args(args, output_dir: str, do_eval: bool) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        fp16=args.precision == "fp16" and torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim=args.optim,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch" if do_eval else "no"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch" if do_eval else "no"
    if "save_total_limit" in sig.parameters:
        kwargs["save_total_limit"] = 1
    if "gradient_checkpointing" in sig.parameters:
        kwargs["gradient_checkpointing"] = args.gradient_checkpointing

    return TrainingArguments(**kwargs)


def load_sequence_regressor(
    model_name: str,
    models: List[str],
    args,
    hf_token: Optional[str],
    for_training: bool,
):
    id2label = {idx: name for idx, name in enumerate(models)}
    label2id = {name: idx for idx, name in id2label.items()}
    quant_config = build_bnb_config(args)

    max_memory = None
    if args.max_gpu_mem:
        max_memory = {0: args.max_gpu_mem, "cpu": args.max_cpu_mem}

    model_kwargs = dict(
        token=hf_token,
        num_labels=len(models),
        id2label=id2label,
        label2id=label2id,
        problem_type="regression",
        ignore_mismatched_sizes=True,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        max_memory=max_memory,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.config.problem_type = "regression"
    model.config.use_cache = False

    if for_training and args.load_in_4bit:
        prepare_kwargs = {}
        prepare_sig = inspect.signature(prepare_model_for_kbit_training)
        if "use_gradient_checkpointing" in prepare_sig.parameters:
            prepare_kwargs["use_gradient_checkpointing"] = args.gradient_checkpointing
        model = prepare_model_for_kbit_training(model, **prepare_kwargs)

    if for_training and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def attach_lora(model, args):
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.lora_target_modules),
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def compute_regression_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    mse = float(np.mean((predictions - labels) ** 2))
    mae = float(np.mean(np.abs(predictions - labels)))
    return {"mse": mse, "mae": mae}


def train_or_predict_one_target(
    target_name: str,
    model_name: str,
    adapter_path: Optional[str],
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    train_target: Optional[pd.DataFrame],
    val_target: Optional[pd.DataFrame],
    models: List[str],
    args,
    hf_token: Optional[str],
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    logger.info("Loading tokenizer for {}: {}", target_name, model_name)
    tokenizer = load_tokenizer(model_name, hf_token)

    train_ds = None
    val_ds = None
    if args.mode == "train-predict":
        if train_target is None:
            raise ValueError("train_target is required in train-predict mode")
        train_ds = build_regression_dataset(
            train_df,
            train_target,
            tokenizer,
            max_length=args.max_length,
            tokenize_batch_size=args.tokenize_batch_size,
            desc=f"Tokenizing {target_name} train",
        )
        if val_df is not None and val_target is not None and len(val_df) > 0:
            val_ds = build_regression_dataset(
                val_df,
                val_target,
                tokenizer,
                max_length=args.max_length,
                tokenize_batch_size=args.tokenize_batch_size,
                desc=f"Tokenizing {target_name} val",
            )

    test_ds = build_inference_dataset(
        test_df,
        tokenizer,
        max_length=args.max_length,
        tokenize_batch_size=args.tokenize_batch_size,
        desc=f"Tokenizing {target_name} test",
    )

    logger.info("Loading {} model", target_name)
    model = load_sequence_regressor(
        model_name=model_name,
        models=models,
        args=args,
        hf_token=hf_token,
        for_training=args.mode == "train-predict",
    )

    if args.mode == "train-predict":
        model = attach_lora(model, args)
    else:
        if not adapter_path:
            raise ValueError(f"--{target_name}-adapter is required for predict-only mode")
        logger.info("Loading {} adapter from {}", target_name, adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    data_collator = RegressionDataCollator(tokenizer)
    training_args = make_training_args(
        args,
        output_dir=str(output_dir / f"{target_name}_trainer"),
        do_eval=val_ds is not None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_regression_metrics if val_ds is not None else None,
    )

    if args.mode == "train-predict":
        logger.info("Training {} adapter", target_name)
        trainer.train()
        save_path = adapter_path or str(output_dir / f"{target_name}_adapter")
        logger.info("Saving {} adapter to {}", target_name, save_path)
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

    val_predictions = None
    if val_ds is not None:
        logger.info("Predicting {} validation scores", target_name)
        val_predictions = trainer.predict(val_ds).predictions.astype(np.float32)

    logger.info("Predicting {} test scores", target_name)
    test_predictions = trainer.predict(test_ds).predictions.astype(np.float32)

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_predictions, test_predictions


def choose_models(
    perf_pred: np.ndarray,
    cost_pred_norm: np.ndarray,
    model_names: List[str],
    alpha: float,
    clip_predictions: bool,
    task_profiles: Optional[pd.DataFrame] = None,
    task_reward_priors: Optional[TaskPriorTable] = None,
    task_prior_weight: float = 0.0,
    coding_task_prior_weight: Optional[float] = None,
    task_prior_min_count: int = 20,
) -> List[str]:
    perf_scores = perf_pred
    cost_scores = cost_pred_norm
    if clip_predictions:
        perf_scores = np.clip(perf_scores, 0.0, 1.0)
        cost_scores = np.clip(cost_scores, 0.0, None)

    reward_scores = alpha * perf_scores - (1.0 - alpha) * cost_scores
    if task_profiles is not None and task_reward_priors is not None and task_prior_weight > 0:
        prior_scores = task_prior_scores(
            task_profiles,
            task_reward_priors,
            min_count=task_prior_min_count,
        )
        if prior_scores.shape == reward_scores.shape:
            row_weights = np.full(len(task_profiles), task_prior_weight, dtype=np.float32)
            if coding_task_prior_weight is not None:
                coding_mask = task_profiles["task_domain"].eq("coding").to_numpy()
                row_weights[coding_mask] = coding_task_prior_weight
            row_weights = np.clip(row_weights, 0.0, 1.0)[:, None]
            reward_scores = (1.0 - row_weights) * reward_scores + row_weights * prior_scores
    pred_ids = np.argmax(reward_scores, axis=1)
    return [model_names[idx] for idx in pred_ids]


def save_score_debug(
    out_dir: Path,
    prefix: str,
    ids: pd.Series,
    model_names: List[str],
    perf_pred: np.ndarray,
    cost_pred_norm: np.ndarray,
    preds: List[str],
    task_profiles: Optional[pd.DataFrame] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({"ID": ids, "pred_model": preds})
    if task_profiles is not None:
        for col in ["task_domain", "task_subtask", "task_difficulty", "task_profile"]:
            rows[col] = task_profiles[col].to_list()
    for idx, model_name in enumerate(model_names):
        rows[f"{model_name}_pred_perf"] = perf_pred[:, idx]
        rows[f"{model_name}_pred_cost_norm"] = cost_pred_norm[:, idx]
    rows.to_csv(out_dir / f"{prefix}_score_debug.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--out", default="submission_starcoder2_qlora.csv")
    parser.add_argument("--output-dir", default="starcoder2_qlora_runs")
    parser.add_argument("--mode", choices=["train-predict", "predict-only"], default="train-predict")
    parser.add_argument("--perf-model", default="withmartian/starcoder2-3b-bcbToppers-perf")
    parser.add_argument("--cost-model", default="withmartian/starcoder2-3b-bcbToppers-cost")
    parser.add_argument("--coding-perf-model", default="withmartian/starcoder2-3b-bcbToppers-perf")
    parser.add_argument("--coding-cost-model", default="withmartian/starcoder2-3b-bcbToppers-cost")
    parser.add_argument("--perf-adapter", default="")
    parser.add_argument("--cost-adapter", default="")
    parser.add_argument("--coding-perf-adapter", default="")
    parser.add_argument("--coding-cost-adapter", default="")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--cost-norm",
        default="mean-row-max",
        choices=["mean-row-max", "row-max", "global-max", "none"],
        help="mean-row-max matches Reward = alpha * P_bar - beta * C_bar / Cmax_bar.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="attention",
        help="'attention', 'all', or a comma-separated module list.",
    )
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--max-gpu-mem", default="")
    parser.add_argument("--max-cpu-mem", default="48GiB")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--no-clip-preds", dest="clip_preds", action="store_false")
    parser.add_argument("--task-aware", action="store_true", default=True)
    parser.add_argument("--no-task-aware", dest="task_aware", action="store_false")
    parser.add_argument("--task-prior-weight", type=float, default=0.15)
    parser.add_argument("--coding-task-prior-weight", type=float, default=0.30)
    parser.add_argument("--task-prior-min-count", type=int, default=20)
    parser.add_argument("--task-prior-smoothing", type=float, default=50.0)
    parser.add_argument("--coding-specialist", action="store_true", default=True)
    parser.add_argument("--no-coding-specialist", dest="coding_specialist", action="store_false")
    parser.add_argument("--min-coding-train-rows", type=int, default=128)
    parser.add_argument("--save-score-debug", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "predict-only" and (not args.perf_adapter or not args.cost_adapter):
        raise ValueError("predict-only mode requires both --perf-adapter and --cost-adapter")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = get_hf_token()
    if hf_token:
        logger.info("HF token detected from environment")
    else:
        logger.warning("HF token not found; public model downloads should still work")

    logger.info("Loading dataset")
    train_df, test_df = load_data(args.train, args.test)
    model_names = parse_model_names(train_df.columns.tolist())
    perf, cost = extract_perf_cost(train_df, model_names)

    cfg = RewardConfig(alpha=args.alpha, cost_norm=args.cost_norm)
    oracle_reward = compute_reward(perf, cost, cfg)
    oracle_labels = labels_from_reward(oracle_reward)
    oracle_score = evaluate_policy_reward(perf, cost, oracle_labels.tolist(), cfg)
    logger.info("Train oracle reward with {} normalization: {:.6f}", args.cost_norm, oracle_score)

    cost_norm = normalize_cost(cost, cfg)
    train_idx, val_idx = split_train_val_indices(len(train_df), args.val_ratio, args.seed)
    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    val_part = train_df.iloc[val_idx].reset_index(drop=True) if len(val_idx) else None
    full_profiles = infer_task_profiles(train_df)
    train_profiles = infer_task_profiles(train_part)
    val_profiles = infer_task_profiles(val_part) if val_part is not None else None
    test_profiles = infer_task_profiles(test_df)
    log_task_summary("Train", full_profiles)
    log_task_summary("Test", test_profiles)

    perf_train = perf.iloc[train_idx].reset_index(drop=True)
    cost_train = cost_norm.iloc[train_idx].reset_index(drop=True)
    perf_val = perf.iloc[val_idx].reset_index(drop=True) if len(val_idx) else None
    cost_val = cost_norm.iloc[val_idx].reset_index(drop=True) if len(val_idx) else None

    perf_adapter = args.perf_adapter or str(out_dir / "perf_adapter")
    cost_adapter = args.cost_adapter or str(out_dir / "cost_adapter")
    coding_perf_adapter = args.coding_perf_adapter or str(out_dir / "coding_perf_adapter")
    coding_cost_adapter = args.coding_cost_adapter or str(out_dir / "coding_cost_adapter")
    coding_specialist_available = args.coding_specialist
    if args.mode == "predict-only" and args.coding_specialist:
        missing_coding_adapters = [
            path for path in [coding_perf_adapter, coding_cost_adapter] if not Path(path).exists()
        ]
        if missing_coding_adapters:
            coding_specialist_available = False
            logger.warning(
                "Skipping coding specialist in predict-only mode because adapters are missing: {}",
                ", ".join(missing_coding_adapters),
            )

    train_task_priors = None
    test_task_priors = None
    if args.task_aware:
        train_reward_for_priors = args.alpha * perf_train - (1.0 - args.alpha) * cost_train
        train_task_priors = build_task_reward_priors(
            train_profiles,
            train_reward_for_priors,
            smoothing=args.task_prior_smoothing,
        )
        test_task_priors = build_task_reward_priors(
            full_profiles,
            oracle_reward,
            smoothing=args.task_prior_smoothing,
        )

    perf_val_pred, perf_test_pred = train_or_predict_one_target(
        target_name="perf",
        model_name=args.perf_model,
        adapter_path=perf_adapter,
        output_dir=out_dir,
        train_df=train_part,
        val_df=val_part,
        test_df=test_df,
        train_target=perf_train,
        val_target=perf_val,
        models=model_names,
        args=args,
        hf_token=hf_token,
    )

    cost_val_pred, cost_test_pred = train_or_predict_one_target(
        target_name="cost",
        model_name=args.cost_model,
        adapter_path=cost_adapter,
        output_dir=out_dir,
        train_df=train_part,
        val_df=val_part,
        test_df=test_df,
        train_target=cost_train,
        val_target=cost_val,
        models=model_names,
        args=args,
        hf_token=hf_token,
    )

    coding_train_mask = train_profiles["task_domain"].eq("coding").to_numpy()
    test_coding_mask = test_profiles["task_domain"].eq("coding").to_numpy()
    val_coding_mask = val_profiles["task_domain"].eq("coding").to_numpy() if val_profiles is not None else None
    coding_train_count = int(coding_train_mask.sum())
    should_run_coding_specialist = (
        coding_specialist_available
        and coding_train_count >= args.min_coding_train_rows
        and bool(test_coding_mask.any())
    )
    if should_run_coding_specialist:
        logger.info(
            "Training/predicting coding specialist with {} rows using {} and {}",
            coding_train_count,
            args.coding_perf_model,
            args.coding_cost_model,
        )
        coding_train_part = train_part.loc[coding_train_mask].reset_index(drop=True)
        coding_perf_train = perf_train.loc[coding_train_mask].reset_index(drop=True)
        coding_cost_train = cost_train.loc[coding_train_mask].reset_index(drop=True)

        coding_val_part = None
        coding_perf_val = None
        coding_cost_val = None
        if val_part is not None and val_coding_mask is not None and val_coding_mask.any():
            coding_val_part = val_part.loc[val_coding_mask].reset_index(drop=True)
            coding_perf_val = perf_val.loc[val_coding_mask].reset_index(drop=True) if perf_val is not None else None
            coding_cost_val = cost_val.loc[val_coding_mask].reset_index(drop=True) if cost_val is not None else None

        coding_test_part = test_df.loc[test_coding_mask].reset_index(drop=True)

        coding_perf_val_pred, coding_perf_test_pred = train_or_predict_one_target(
            target_name="coding_perf",
            model_name=args.coding_perf_model,
            adapter_path=coding_perf_adapter,
            output_dir=out_dir,
            train_df=coding_train_part,
            val_df=coding_val_part,
            test_df=coding_test_part,
            train_target=coding_perf_train,
            val_target=coding_perf_val,
            models=model_names,
            args=args,
            hf_token=hf_token,
        )

        coding_cost_val_pred, coding_cost_test_pred = train_or_predict_one_target(
            target_name="coding_cost",
            model_name=args.coding_cost_model,
            adapter_path=coding_cost_adapter,
            output_dir=out_dir,
            train_df=coding_train_part,
            val_df=coding_val_part,
            test_df=coding_test_part,
            train_target=coding_cost_train,
            val_target=coding_cost_val,
            models=model_names,
            args=args,
            hf_token=hf_token,
        )

        perf_test_pred = perf_test_pred.copy()
        cost_test_pred = cost_test_pred.copy()
        perf_test_pred[test_coding_mask] = coding_perf_test_pred
        cost_test_pred[test_coding_mask] = coding_cost_test_pred

        if (
            perf_val_pred is not None
            and cost_val_pred is not None
            and val_coding_mask is not None
            and val_coding_mask.any()
            and coding_perf_val_pred is not None
            and coding_cost_val_pred is not None
        ):
            perf_val_pred = perf_val_pred.copy()
            cost_val_pred = cost_val_pred.copy()
            perf_val_pred[val_coding_mask] = coding_perf_val_pred
            cost_val_pred[val_coding_mask] = coding_cost_val_pred
    elif args.coding_specialist:
        logger.info(
            "Skipping coding specialist: coding_train_rows={}, test_coding_rows={}",
            coding_train_count,
            int(test_coding_mask.sum()),
        )

    if perf_val_pred is not None and cost_val_pred is not None and len(val_idx):
        val_preds = choose_models(
            perf_val_pred,
            cost_val_pred,
            model_names,
            alpha=args.alpha,
            clip_predictions=args.clip_preds,
            task_profiles=val_profiles if args.task_aware else None,
            task_reward_priors=train_task_priors,
            task_prior_weight=args.task_prior_weight if args.task_aware else 0.0,
            coding_task_prior_weight=args.coding_task_prior_weight,
            task_prior_min_count=args.task_prior_min_count,
        )
        val_perf = perf.iloc[val_idx].reset_index(drop=True)
        val_cost = cost.iloc[val_idx].reset_index(drop=True)
        val_reward = evaluate_policy_reward(val_perf, val_cost, val_preds, cfg)
        val_oracle = evaluate_policy_reward(
            val_perf,
            val_cost,
            labels_from_reward(compute_reward(val_perf, val_cost, cfg)).tolist(),
            cfg,
        )
        logger.info("Validation router reward: {:.6f}", val_reward)
        logger.info("Validation oracle reward: {:.6f}", val_oracle)
        if args.save_score_debug:
            save_score_debug(
                out_dir,
                "val",
                train_df.iloc[val_idx]["ID"].reset_index(drop=True),
                model_names,
                perf_val_pred,
                cost_val_pred,
                val_preds,
                task_profiles=val_profiles,
            )

    test_preds = choose_models(
        perf_test_pred,
        cost_test_pred,
        model_names,
        alpha=args.alpha,
        clip_predictions=args.clip_preds,
        task_profiles=test_profiles if args.task_aware else None,
        task_reward_priors=test_task_priors,
        task_prior_weight=args.task_prior_weight if args.task_aware else 0.0,
        coding_task_prior_weight=args.coding_task_prior_weight,
        task_prior_min_count=args.task_prior_min_count,
    )
    make_submission(test_df, test_preds, args.out)
    logger.info("Saved submission to {}", args.out)

    if args.save_score_debug:
        save_score_debug(
            out_dir,
            "test",
            test_df["ID"],
            model_names,
            perf_test_pred,
            cost_test_pred,
            test_preds,
            task_profiles=test_profiles,
        )


if __name__ == "__main__":
    main()
