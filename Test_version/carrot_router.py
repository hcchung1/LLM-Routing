import argparse
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_PROFILES: Dict[str, Dict[str, object]] = {
    "qwen3-4b-1080ti": {
        "model_name": "Qwen/Qwen3-4B-Base",
        "max_length": 512,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,
        "compute_dtype": "float16",
    },
    "qwen3-8b-l4": {
        "model_name": "Qwen/Qwen3-8B-Base",
        "max_length": 512,
        "batch_size": 1,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,
        "compute_dtype": "bfloat16",
    },
}

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

warnings.filterwarnings(
    "ignore",
    message=r"_check_is_size will be removed in a future PyTorch release.*",
    category=FutureWarning,
    module=r"bitsandbytes\.backends\.cuda\.ops",
)


@dataclass
class CarrotMetadata:
    profile: str
    model_name: str
    model_names: List[str]
    text_column: str
    max_length: int
    cost_mean: List[float]
    cost_std: List[float]
    global_max_cost: float
    cost_weight: float
    seed: int
    ranking_loss_weight: float = 1.0
    ranking_temperature: float = 0.05
    ranking_min_gap: float = 1e-4
    row_weight_floor: float = 0.05


@dataclass
class SoftOracleMetadata:
    profile: str
    model_name: str
    model_names: List[str]
    text_column: str
    max_length: int
    cost_denominator_method: str
    cost_denominator: float
    soft_label_temperature: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent CARROT router: estimate per-model performance and cost, "
            "then apply the paper's plug-in routing rule."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train CARROT performance and cost predictors.")
    add_common_data_args(train_parser)
    add_model_args(train_parser)
    train_parser.add_argument("--output-dir", default="carrot_runs/qwen3-4b-1080ti")
    train_parser.add_argument("--head", choices=["both", "performance", "cost"], default="both")
    train_parser.add_argument("--epochs", type=float, default=5.0)
    train_parser.add_argument("--val-size", type=float, default=0.1)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--logging-steps", type=int, default=50)
    train_parser.add_argument("--save-total-limit", type=int, default=1)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--resume-from-checkpoint", default="")
    train_parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--ranking-temperature", type=float, default=0.05)
    train_parser.add_argument("--ranking-min-gap", type=float, default=1e-4)
    train_parser.add_argument("--row-weight-floor", type=float, default=0.05)
    train_parser.add_argument("--audit-out", default="")

    soft_train_parser = subparsers.add_parser(
        "train-soft-oracle",
        help="Train a regret-weighted soft-oracle model classifier.",
    )
    add_common_data_args(soft_train_parser)
    add_model_args(soft_train_parser)
    soft_train_parser.add_argument(
        "--output-dir",
        default="carrot_runs/qwen3-4b-soft-oracle",
    )
    soft_train_parser.add_argument("--epochs", type=float, default=5.0)
    soft_train_parser.add_argument("--val-size", type=float, default=0.1)
    soft_train_parser.add_argument("--weight-decay", type=float, default=0.01)
    soft_train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    soft_train_parser.add_argument("--logging-steps", type=int, default=50)
    soft_train_parser.add_argument("--save-total-limit", type=int, default=1)
    soft_train_parser.add_argument("--lora-r", type=int, default=16)
    soft_train_parser.add_argument("--lora-alpha", type=int, default=32)
    soft_train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    soft_train_parser.add_argument("--resume-from-checkpoint", default="")
    soft_train_parser.add_argument("--soft-label-temperature", type=float, default=0.05)
    soft_train_parser.add_argument(
        "--cost-denominator",
        choices=["mean-row-max", "global-max"],
        default="mean-row-max",
    )
    soft_train_parser.add_argument("--audit-out", default="")

    predict_parser = subparsers.add_parser("predict", help="Route queries with trained CARROT predictors.")
    add_common_data_args(predict_parser)
    add_model_args(predict_parser)
    predict_parser.add_argument("--adapter-dir", required=True)
    predict_parser.add_argument("--out", default="submission_carrot.csv")
    predict_parser.add_argument("--score-out", default="")
    predict_parser.add_argument("--prediction-out", default="")
    predict_parser.add_argument("--cost-weight", type=float, default=None)
    predict_parser.add_argument("--calibration", default="")
    predict_parser.add_argument("--knn-correction", default="")
    predict_parser.add_argument("--knn-weight", type=float, default=0.0)
    predict_parser.add_argument("--embedding-batch-size", type=int, default=32)

    soft_predict_parser = subparsers.add_parser(
        "predict-soft-oracle",
        help="Route queries with a trained soft-oracle classifier.",
    )
    add_common_data_args(soft_predict_parser)
    add_model_args(soft_predict_parser)
    soft_predict_parser.add_argument("--adapter-dir", required=True)
    soft_predict_parser.add_argument("--out", default="submission_carrot_soft_oracle.csv")
    soft_predict_parser.add_argument("--score-out", default="")
    soft_predict_parser.add_argument("--prediction-out", default="")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Fit per-model score biases and optional KNN residual correction from a validation audit.",
    )
    calibrate_parser.add_argument("--audit", required=True)
    calibrate_parser.add_argument("--out", default="carrot_calibration.json")
    calibrate_parser.add_argument("--max-bias", type=float, default=0.05)
    calibrate_parser.add_argument("--bias-step", type=float, default=0.002)
    calibrate_parser.add_argument("--passes", type=int, default=3)
    calibrate_parser.add_argument("--shrink", type=float, default=0.5)
    calibrate_parser.add_argument("--knn-out", default="")
    calibrate_parser.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    calibrate_parser.add_argument("--embedding-batch-size", type=int, default=32)
    calibrate_parser.add_argument("--knn-k", type=int, default=50)

    list_parser = subparsers.add_parser("list-profiles", help="Print built-in hardware profiles.")
    list_parser.set_defaults(profile="")

    return parser.parse_args()


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--test", default="dataset/test.csv")
    parser.add_argument("--text-column", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="qwen3-4b-1080ti")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16"], default="")
    parser.add_argument("--hf-token", default="")


def resolve_profile(args: argparse.Namespace) -> Dict[str, object]:
    profile = dict(MODEL_PROFILES[args.profile])
    overrides = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "compute_dtype": args.compute_dtype,
    }
    for key, value in overrides.items():
        if value not in ("", 0, 0.0):
            profile[key] = value
    return profile


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_model_names(columns: Sequence[str]) -> List[str]:
    models = []
    for column in columns:
        if column.startswith("Model_") and column.endswith("_performance"):
            model = column.removesuffix("_performance")
            if f"{model}_cost" in columns:
                models.append(model)
    if not models:
        raise ValueError("No Model_*_performance and Model_*_cost column pairs were found.")
    return sorted(models)


def find_text_column(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Text column {requested!r} is not present.")
        return requested
    for column in ["query", "prompt", "Question", "question", "text"]:
        if column in df.columns:
            return column
    raise ValueError("Could not infer the text column.")


def read_csv(path: str, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit > 0:
        df = df.head(limit).copy()
    return df


def extract_targets(df: pd.DataFrame, model_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    performance = df[[f"{model}_performance" for model in model_names]].to_numpy(dtype=np.float32)
    cost = df[[f"{model}_cost" for model in model_names]].to_numpy(dtype=np.float32)
    return performance, cost


def fit_cost_scaler(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = cost.mean(axis=0, keepdims=True).astype(np.float32)
    std = cost.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-8] = 1.0
    return ((cost - mean) / std).astype(np.float32), mean, std


def compute_true_reward(
    performance: np.ndarray,
    cost: np.ndarray,
    global_max_cost: float,
) -> np.ndarray:
    return (
        0.85 * performance
        - 0.15 * cost / max(float(global_max_cost), 1e-12)
    ).astype(np.float32)


def compute_cost_denominator(cost: np.ndarray, method: str) -> float:
    if method == "mean-row-max":
        denominator = float(np.max(cost, axis=1).mean())
    elif method == "global-max":
        denominator = float(np.max(cost))
    else:
        raise ValueError(f"Unknown cost denominator method: {method}")
    return max(denominator, 1e-12)


def softmax_numpy(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("--soft-label-temperature must be greater than zero.")
    scaled = np.asarray(values, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def regret_gap_weights(reward: np.ndarray) -> np.ndarray:
    if reward.shape[1] < 2:
        raise ValueError("Regret gap weights require at least two candidate models.")
    top_two = np.partition(reward, -2, axis=1)[:, -2:]
    gaps = top_two[:, 1] - top_two[:, 0]
    return np.clip(gaps, 0.0, 1.0).astype(np.float32)


def pack_soft_oracle_labels(
    soft_targets: np.ndarray,
    reward: np.ndarray,
    row_weights: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [soft_targets, reward, row_weights[:, None]],
        axis=1,
    ).astype(np.float32)


def unpack_soft_oracle_labels(
    labels: np.ndarray,
    n_models: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    soft_targets = labels[:, :n_models]
    reward = labels[:, n_models : 2 * n_models]
    row_weights = labels[:, 2 * n_models]
    return soft_targets, reward, row_weights


def informative_row_weights(reward: np.ndarray, floor: float) -> np.ndarray:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("--row-weight-floor must be between 0 and 1.")
    reward_range = reward.max(axis=1) - reward.min(axis=1)
    scale = float(np.quantile(reward_range, 0.9))
    if scale <= 1e-12:
        return np.ones(len(reward), dtype=np.float32)
    weights = np.clip(reward_range / scale, 0.0, 1.0)
    return (floor + (1.0 - floor) * weights).astype(np.float32)


def pack_performance_labels(
    performance: np.ndarray,
    reward: np.ndarray,
    row_weights: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [performance, reward, row_weights[:, None]],
        axis=1,
    ).astype(np.float32)


def unpack_performance_labels(
    labels: np.ndarray,
    n_models: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    performance = labels[:, :n_models]
    reward = labels[:, n_models : 2 * n_models]
    row_weights = labels[:, 2 * n_models]
    return performance, reward, row_weights


def split_indices(n_rows: int, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    n_val = max(1, int(round(n_rows * val_size)))
    return indices[n_val:], indices[:n_val]


def require_training_packages() -> None:
    missing = []
    for package, import_name in [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required packages: {joined}. Install requirements.txt first.")


def torch_dtype(name: str):
    import torch

    return torch.float16 if name == "float16" else torch.bfloat16


def validate_device(profile: Dict[str, object]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CARROT 3B/8B QLoRA training and inference require a CUDA GPU.")
    if profile["compute_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires BF16. Use qwen3-4b-1080ti or --compute-dtype float16.")


def load_tokenizer(model_name: str, hf_token: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token or None,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_qlora_model(
    model_name: str,
    num_labels: int,
    problem_type: str,
    profile: Dict[str, object],
    hf_token: str,
    trainable: bool,
    lora_config: Optional[Dict[str, object]] = None,
):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    dtype = torch_dtype(str(profile["compute_dtype"]))
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=dtype,
        token=hf_token or None,
        trust_remote_code=False,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    if not trainable:
        return model

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = lora_config or {}
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)),
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


class TextTargetDataset:
    def __init__(
        self,
        texts: Sequence[str],
        targets: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = [str(text) for text in texts]
        self.targets = np.asarray(targets, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {"text": self.texts[index], "labels": self.targets[index]}


class TextDataset:
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int) -> None:
        self.texts = [str(text) for text in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {"text": self.texts[index]}


class CarrotCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, object]:
        import torch

        labels = None
        if "labels" in features[0]:
            labels = np.stack([np.asarray(feature.pop("labels"), dtype=np.float32) for feature in features])
        texts = [str(feature["text"]) for feature in features]
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def training_arguments(
    output_dir: Path,
    args: argparse.Namespace,
    profile: Dict[str, object],
    head_name: str,
):
    import inspect
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": float(profile["learning_rate"]),
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": int(profile["batch_size"]),
        "per_device_eval_batch_size": int(profile["eval_batch_size"]),
        "gradient_accumulation_steps": int(profile["gradient_accumulation_steps"]),
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "linear",
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": (
            "eval_routing_reward"
            if head_name in {"performance", "soft-oracle"}
            else "eval_loss"
        ),
        "greater_is_better": head_name in {"performance", "soft-oracle"},
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "fp16": profile["compute_dtype"] == "float16",
        "bf16": profile["compute_dtype"] == "bfloat16",
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    kwargs["save_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def make_performance_metrics(n_models: int):
    def compute_metrics(eval_prediction) -> Dict[str, float]:
        logits = np.asarray(eval_prediction.predictions, dtype=np.float32)
        labels = np.asarray(eval_prediction.label_ids, dtype=np.float32)
        true_performance, true_reward, _ = unpack_performance_labels(labels, n_models)
        predicted_performance = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        true_normalized_cost = np.maximum(
            0.0,
            (0.85 * true_performance - true_reward) / 0.15,
        )
        predicted_scores = 0.85 * predicted_performance - 0.15 * true_normalized_cost
        selected = predicted_scores.argmax(axis=1)
        rows = np.arange(len(selected))
        routing_reward = float(true_reward[rows, selected].mean())
        oracle_reward = float(true_reward.max(axis=1).mean())
        regret = oracle_reward - routing_reward
        return {
            "routing_reward": routing_reward,
            "oracle_reward": oracle_reward,
            "routing_regret": regret,
        }

    return compute_metrics


def routing_aware_loss(
    logits,
    labels,
    n_models: int,
    ranking_loss_weight: float,
    ranking_temperature: float,
    ranking_min_gap: float,
):
    import torch
    import torch.nn.functional as functional

    true_performance = labels[:, :n_models]
    true_reward = labels[:, n_models : 2 * n_models]
    row_weights = labels[:, 2 * n_models]
    logits = logits.float()

    bce_per_model = functional.binary_cross_entropy_with_logits(
        logits,
        true_performance,
        reduction="none",
    )
    bce_per_row = bce_per_model.mean(dim=1)
    bce_loss = (bce_per_row * row_weights).sum() / row_weights.sum().clamp_min(1e-6)

    predicted_performance = torch.sigmoid(logits)
    true_normalized_cost = torch.clamp(
        (0.85 * true_performance - true_reward) / 0.15,
        min=0.0,
    )
    predicted_scores = 0.85 * predicted_performance - 0.15 * true_normalized_cost
    true_gap = true_reward.unsqueeze(2) - true_reward.unsqueeze(1)
    predicted_gap = predicted_scores.unsqueeze(2) - predicted_scores.unsqueeze(1)
    upper_triangle = torch.triu(
        torch.ones(
            (n_models, n_models),
            dtype=torch.bool,
            device=logits.device,
        ),
        diagonal=1,
    )
    valid_pairs = upper_triangle.unsqueeze(0) & (true_gap.abs() >= ranking_min_gap)
    pair_weights = true_gap.abs()
    pair_loss = functional.softplus(
        -true_gap.sign() * predicted_gap / ranking_temperature
    )
    weighted_pair_loss = pair_loss * pair_weights * row_weights[:, None, None]
    pair_denominator = (
        pair_weights * row_weights[:, None, None] * valid_pairs
    ).sum()
    if pair_denominator.item() > 0:
        ranking_loss = (weighted_pair_loss * valid_pairs).sum() / pair_denominator
    else:
        ranking_loss = logits.new_zeros(())
    total_loss = bce_loss + ranking_loss_weight * ranking_loss
    return total_loss, bce_loss.detach(), ranking_loss.detach()


def soft_oracle_loss(logits, labels, n_models: int):
    import torch.nn.functional as functional

    soft_targets, _, row_weights = (
        labels[:, :n_models],
        labels[:, n_models : 2 * n_models],
        labels[:, 2 * n_models],
    )
    per_row = functional.kl_div(
        functional.log_softmax(logits.float(), dim=1),
        soft_targets,
        reduction="none",
    ).sum(dim=1)
    denominator = row_weights.sum().clamp_min(1e-6)
    return (per_row * row_weights).sum() / denominator


def make_soft_oracle_metrics(n_models: int):
    def compute_metrics(eval_prediction) -> Dict[str, float]:
        logits = np.asarray(eval_prediction.predictions, dtype=np.float32)
        labels = np.asarray(eval_prediction.label_ids, dtype=np.float32)
        _, true_reward, _ = unpack_soft_oracle_labels(labels, n_models)
        selected = logits.argmax(axis=1)
        oracle = true_reward.argmax(axis=1)
        rows = np.arange(len(selected))
        routing_reward = float(true_reward[rows, selected].mean())
        oracle_reward = float(true_reward[rows, oracle].mean())
        return {
            "routing_reward": routing_reward,
            "oracle_reward": oracle_reward,
            "routing_regret": oracle_reward - routing_reward,
            "top1_accuracy": float((selected == oracle).mean()),
        }

    return compute_metrics


def build_soft_oracle_trainer(
    model,
    trainer_args,
    train_dataset,
    val_dataset,
    collator,
    n_models: int,
):
    from transformers import Trainer

    class SoftOracleTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = soft_oracle_loss(outputs.logits, labels, n_models)
            if return_outputs:
                return loss, outputs
            return loss

    return SoftOracleTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_soft_oracle_metrics(n_models),
    )


def build_routing_aware_trainer(
    model,
    trainer_args,
    train_dataset,
    val_dataset,
    collator,
    args: argparse.Namespace,
    n_models: int,
):
    from transformers import Trainer

    class RoutingAwareTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss, _, _ = routing_aware_loss(
                outputs.logits,
                labels,
                n_models,
                args.ranking_loss_weight,
                args.ranking_temperature,
                args.ranking_min_gap,
            )
            if return_outputs:
                return loss, outputs
            return loss

    return RoutingAwareTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_performance_metrics(n_models),
    )


def write_validation_audit(
    trainer,
    val_dataset,
    val_indices: np.ndarray,
    val_texts: Sequence[str],
    model_names: Sequence[str],
    global_max_cost: float,
    out_path: Path,
) -> None:
    prediction = trainer.predict(val_dataset)
    logits = np.asarray(prediction.predictions, dtype=np.float32)
    labels = np.asarray(prediction.label_ids, dtype=np.float32)
    true_performance, true_reward, row_weights = unpack_performance_labels(
        labels,
        len(model_names),
    )
    predicted_performance = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    true_normalized_cost = np.maximum(
        0.0,
        (0.85 * true_performance - true_reward) / 0.15,
    )
    predicted_scores = 0.85 * predicted_performance - 0.15 * true_normalized_cost
    selected = predicted_scores.argmax(axis=1)
    oracle = true_reward.argmax(axis=1)
    rows = np.arange(len(selected))
    margins = np.partition(predicted_scores, -2, axis=1)
    margins = margins[:, -1] - margins[:, -2]
    audit_reward = float(true_reward[rows, selected].mean())
    oracle_reward = float(true_reward[rows, oracle].mean())
    per_row_regret = true_reward[rows, oracle] - true_reward[rows, selected]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        logits=logits,
        predicted_performance=predicted_performance,
        predicted_scores=predicted_scores,
        true_performance=true_performance,
        true_reward=true_reward,
        true_normalized_cost=true_normalized_cost,
        row_weights=row_weights,
        selected=selected,
        oracle=oracle,
        margins=margins,
        val_indices=val_indices,
        texts=np.asarray(list(val_texts), dtype=object),
        model_names=np.asarray(model_names),
        global_max_cost=np.asarray([global_max_cost], dtype=np.float32),
    )
    summary = {
        "routing_reward": audit_reward,
        "oracle_reward": oracle_reward,
        "regret": oracle_reward - audit_reward,
        "selection_counts": {
            model: int((selected == index).sum())
            for index, model in enumerate(model_names)
        },
        "oracle_counts": {
            model: int((oracle == index).sum())
            for index, model in enumerate(model_names)
        },
        "conditional_regret_by_selected_model": {
            model: (
                float(per_row_regret[selected == index].mean())
                if np.any(selected == index)
                else None
            )
            for index, model in enumerate(model_names)
        },
        "margin_quantiles": {
            str(quantile): float(np.quantile(margins, quantile))
            for quantile in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        },
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def write_soft_oracle_audit(
    trainer,
    val_dataset,
    val_indices: np.ndarray,
    val_texts: Sequence[str],
    model_names: Sequence[str],
    cost_denominator_method: str,
    cost_denominator: float,
    soft_label_temperature: float,
    out_path: Path,
) -> None:
    prediction = trainer.predict(val_dataset)
    logits = np.asarray(prediction.predictions, dtype=np.float32)
    labels = np.asarray(prediction.label_ids, dtype=np.float32)
    soft_targets, true_reward, row_weights = unpack_soft_oracle_labels(
        labels,
        len(model_names),
    )
    probabilities = softmax_numpy(logits, 1.0)
    selected = logits.argmax(axis=1)
    oracle = true_reward.argmax(axis=1)
    rows = np.arange(len(selected))
    sorted_logits = np.partition(logits, -2, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    reward_gaps = regret_gap_weights(true_reward)
    routing_reward = float(true_reward[rows, selected].mean())
    oracle_reward = float(true_reward[rows, oracle].mean())
    per_row_regret = true_reward[rows, oracle] - true_reward[rows, selected]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        logits=logits,
        probabilities=probabilities,
        soft_targets=soft_targets,
        true_reward=true_reward,
        row_weights=row_weights,
        reward_gaps=reward_gaps,
        selected=selected,
        oracle=oracle,
        margins=margins,
        val_indices=val_indices,
        texts=np.asarray(list(val_texts), dtype=object),
        model_names=np.asarray(model_names),
        cost_denominator_method=np.asarray(cost_denominator_method),
        cost_denominator=np.asarray([cost_denominator], dtype=np.float32),
        soft_label_temperature=np.asarray([soft_label_temperature], dtype=np.float32),
    )
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    summary = {
        "routing_reward": routing_reward,
        "oracle_reward": oracle_reward,
        "regret": oracle_reward - routing_reward,
        "top1_accuracy": float((selected == oracle).mean()),
        "cost_denominator_method": cost_denominator_method,
        "cost_denominator": cost_denominator,
        "soft_label_temperature": soft_label_temperature,
        "selection_counts": {
            model: int((selected == index).sum())
            for index, model in enumerate(model_names)
        },
        "oracle_counts": {
            model: int((oracle == index).sum())
            for index, model in enumerate(model_names)
        },
        "conditional_regret_by_selected_model": {
            model: (
                float(per_row_regret[selected == index].mean())
                if np.any(selected == index)
                else None
            )
            for index, model in enumerate(model_names)
        },
        "row_weight_quantiles": {
            str(quantile): float(np.quantile(row_weights, quantile))
            for quantile in quantiles
        },
        "reward_gap_quantiles": {
            str(quantile): float(np.quantile(reward_gaps, quantile))
            for quantile in quantiles
        },
        "margin_quantiles": {
            str(quantile): float(np.quantile(margins, quantile))
            for quantile in quantiles
        },
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def train_head(
    head_name: str,
    texts: Sequence[str],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    profile: Dict[str, object],
    hf_token: str,
    global_max_cost: float,
    model_names: Sequence[str],
) -> None:
    from transformers import Trainer

    problem_type = "multi_label_classification" if head_name == "performance" else "regression"
    model_name = str(profile["model_name"])
    tokenizer = load_tokenizer(model_name, hf_token)
    model = load_qlora_model(
        model_name=model_name,
        num_labels=len(model_names),
        problem_type=problem_type,
        profile=profile,
        hf_token=hf_token,
        trainable=True,
        lora_config={
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    )
    train_texts = [texts[index] for index in train_idx]
    val_texts = [texts[index] for index in val_idx]
    train_dataset = TextTargetDataset(
        train_texts,
        targets[train_idx],
        tokenizer,
        int(profile["max_length"]),
    )
    val_dataset = TextTargetDataset(
        val_texts,
        targets[val_idx],
        tokenizer,
        int(profile["max_length"]),
    )
    head_dir = output_dir / head_name
    trainer_args = training_arguments(head_dir / "checkpoints", args, profile, head_name)
    collator = CarrotCollator(tokenizer, int(profile["max_length"]))
    if head_name == "performance":
        trainer = build_routing_aware_trainer(
            model,
            trainer_args,
            train_dataset,
            val_dataset,
            collator,
            args,
            len(model_names),
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(head_dir))
    tokenizer.save_pretrained(str(head_dir))
    if head_name == "performance":
        audit_path = Path(args.audit_out) if args.audit_out else output_dir / "validation_audit.npz"
        write_validation_audit(
            trainer,
            val_dataset,
            val_idx,
            val_texts,
            model_names,
            global_max_cost,
            audit_path,
        )


def save_metadata(path: Path, metadata: CarrotMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")


def load_metadata(adapter_dir: Path) -> CarrotMetadata:
    path = adapter_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing CARROT metadata: {path}")
    return CarrotMetadata(**json.loads(path.read_text(encoding="utf-8")))


def load_soft_oracle_metadata(adapter_dir: Path) -> SoftOracleMetadata:
    path = adapter_dir / "soft_oracle_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing soft-oracle metadata: {path}")
    return SoftOracleMetadata(**json.loads(path.read_text(encoding="utf-8")))


def run_train(args: argparse.Namespace) -> None:
    require_training_packages()
    set_seed(args.seed)
    profile = resolve_profile(args)
    validate_device(profile)

    train_df = read_csv(args.train, args.limit)
    model_names = parse_model_names(train_df.columns.tolist())
    text_column = find_text_column(train_df, args.text_column)
    performance, cost = extract_targets(train_df, model_names)
    standardized_cost, cost_mean, cost_std = fit_cost_scaler(cost)
    global_max_cost = max(float(cost.max()), 1e-12)
    true_reward = compute_true_reward(performance, cost, global_max_cost)
    row_weights = informative_row_weights(true_reward, args.row_weight_floor)
    performance_targets = pack_performance_labels(
        performance,
        true_reward,
        row_weights,
    )
    train_idx, val_idx = split_indices(len(train_df), args.val_size, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")

    metadata = CarrotMetadata(
        profile=args.profile,
        model_name=str(profile["model_name"]),
        model_names=model_names,
        text_column=text_column,
        max_length=int(profile["max_length"]),
        cost_mean=cost_mean.reshape(-1).tolist(),
        cost_std=cost_std.reshape(-1).tolist(),
        global_max_cost=global_max_cost,
        cost_weight=0.15,
        seed=args.seed,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_temperature=args.ranking_temperature,
        ranking_min_gap=args.ranking_min_gap,
        row_weight_floor=args.row_weight_floor,
    )
    save_metadata(output_dir / "metadata.json", metadata)

    if args.head in {"both", "performance"}:
        train_head(
            "performance",
            train_df[text_column].astype(str).tolist(),
            performance_targets,
            train_idx,
            val_idx,
            output_dir,
            args,
            profile,
            hf_token,
            global_max_cost,
            model_names,
        )
    if args.head in {"both", "cost"}:
        train_head(
            "cost",
            train_df[text_column].astype(str).tolist(),
            standardized_cost,
            train_idx,
            val_idx,
            output_dir,
            args,
            profile,
            hf_token,
            global_max_cost,
            model_names,
        )


def run_train_soft_oracle(args: argparse.Namespace) -> None:
    require_training_packages()
    set_seed(args.seed)
    profile = resolve_profile(args)
    validate_device(profile)

    train_df = read_csv(args.train, args.limit)
    model_names = parse_model_names(train_df.columns.tolist())
    text_column = find_text_column(train_df, args.text_column)
    performance, cost = extract_targets(train_df, model_names)
    cost_denominator = compute_cost_denominator(cost, args.cost_denominator)
    true_reward = compute_true_reward(performance, cost, cost_denominator)
    soft_targets = softmax_numpy(true_reward, args.soft_label_temperature)
    gap = regret_gap_weights(true_reward)
    scale = float(np.quantile(gap, 0.9))
    if scale <= 1e-12:
        row_weights = np.ones_like(gap, dtype=np.float32)
    else:
        row_weights = 0.10 + 0.90 * np.clip(gap / scale, 0.0, 1.0)
    row_weights = row_weights.astype(np.float32)
    targets = pack_soft_oracle_labels(soft_targets, true_reward, row_weights)
    train_idx, val_idx = split_indices(len(train_df), args.val_size, args.seed)

    output_dir = Path(args.output_dir)
    classifier_dir = output_dir / "classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
    metadata = SoftOracleMetadata(
        profile=args.profile,
        model_name=str(profile["model_name"]),
        model_names=model_names,
        text_column=text_column,
        max_length=int(profile["max_length"]),
        cost_denominator_method=args.cost_denominator,
        cost_denominator=cost_denominator,
        soft_label_temperature=args.soft_label_temperature,
        seed=args.seed,
    )
    save_metadata(output_dir / "soft_oracle_metadata.json", metadata)

    tokenizer = load_tokenizer(str(profile["model_name"]), hf_token)
    model = load_qlora_model(
        model_name=str(profile["model_name"]),
        num_labels=len(model_names),
        problem_type="multi_label_classification",
        profile=profile,
        hf_token=hf_token,
        trainable=True,
        lora_config={
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    )
    texts = train_df[text_column].astype(str).tolist()
    train_dataset = TextTargetDataset(
        [texts[index] for index in train_idx],
        targets[train_idx],
        tokenizer,
        int(profile["max_length"]),
    )
    val_dataset = TextTargetDataset(
        [texts[index] for index in val_idx],
        targets[val_idx],
        tokenizer,
        int(profile["max_length"]),
    )
    trainer_args = training_arguments(
        classifier_dir / "checkpoints",
        args,
        profile,
        "soft-oracle",
    )
    trainer = build_soft_oracle_trainer(
        model,
        trainer_args,
        train_dataset,
        val_dataset,
        CarrotCollator(tokenizer, int(profile["max_length"])),
        len(model_names),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(classifier_dir))
    tokenizer.save_pretrained(str(classifier_dir))
    audit_path = (
        Path(args.audit_out)
        if args.audit_out
        else output_dir / "validation_audit.npz"
    )
    write_soft_oracle_audit(
        trainer,
        val_dataset,
        val_idx,
        [texts[index] for index in val_idx],
        model_names,
        args.cost_denominator,
        cost_denominator,
        args.soft_label_temperature,
        audit_path,
    )
    print(f"Saved soft-oracle classifier to {classifier_dir}")


def predict_head(
    head_name: str,
    texts: Sequence[str],
    adapter_dir: Path,
    metadata: CarrotMetadata,
    profile: Dict[str, object],
    hf_token: str,
) -> np.ndarray:
    import torch
    from peft import PeftModel
    from transformers import Trainer, TrainingArguments

    head_dir = adapter_dir / head_name
    if not head_dir.exists():
        raise FileNotFoundError(f"Missing {head_name} adapter: {head_dir}")
    problem_type = "multi_label_classification" if head_name == "performance" else "regression"
    tokenizer = load_tokenizer(metadata.model_name, hf_token)
    base_model = load_qlora_model(
        model_name=metadata.model_name,
        num_labels=len(metadata.model_names),
        problem_type=problem_type,
        profile=profile,
        hf_token=hf_token,
        trainable=False,
    )
    model = PeftModel.from_pretrained(base_model, str(head_dir), is_trainable=False)
    model.eval()
    dataset = TextDataset(texts, tokenizer, metadata.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_dir / "_predict_tmp"),
            per_device_eval_batch_size=int(profile["eval_batch_size"]),
            report_to="none",
            remove_unused_columns=False,
            fp16=profile["compute_dtype"] == "float16",
            bf16=profile["compute_dtype"] == "bfloat16",
        ),
        data_collator=CarrotCollator(tokenizer, metadata.max_length),
    )
    logits = np.asarray(trainer.predict(dataset).predictions, dtype=np.float32)
    del trainer, model, base_model
    torch.cuda.empty_cache()
    if head_name == "performance":
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return logits


def predict_soft_oracle_logits(
    texts: Sequence[str],
    adapter_dir: Path,
    metadata: SoftOracleMetadata,
    profile: Dict[str, object],
    hf_token: str,
) -> np.ndarray:
    import torch
    from peft import PeftModel
    from transformers import Trainer, TrainingArguments

    classifier_dir = adapter_dir / "classifier"
    if not classifier_dir.exists():
        raise FileNotFoundError(f"Missing soft-oracle adapter: {classifier_dir}")
    tokenizer = load_tokenizer(metadata.model_name, hf_token)
    base_model = load_qlora_model(
        model_name=metadata.model_name,
        num_labels=len(metadata.model_names),
        problem_type="multi_label_classification",
        profile=profile,
        hf_token=hf_token,
        trainable=False,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(classifier_dir),
        is_trainable=False,
    )
    model.eval()
    dataset = TextDataset(texts, tokenizer, metadata.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_dir / "_predict_soft_oracle_tmp"),
            per_device_eval_batch_size=int(profile["eval_batch_size"]),
            report_to="none",
            remove_unused_columns=False,
            fp16=profile["compute_dtype"] == "float16",
            bf16=profile["compute_dtype"] == "bfloat16",
        ),
        data_collator=CarrotCollator(tokenizer, metadata.max_length),
    )
    logits = np.asarray(trainer.predict(dataset).predictions, dtype=np.float32)
    del trainer, model, base_model
    torch.cuda.empty_cache()
    return logits


def carrot_scores(
    performance: np.ndarray,
    standardized_cost: np.ndarray,
    metadata: CarrotMetadata,
    cost_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= cost_weight <= 1.0:
        raise ValueError("--cost-weight must be between 0 and 1.")
    cost_mean = np.asarray(metadata.cost_mean, dtype=np.float32)[None, :]
    cost_std = np.asarray(metadata.cost_std, dtype=np.float32)[None, :]
    predicted_cost = np.maximum(0.0, cost_mean + cost_std * standardized_cost)
    normalized_cost = predicted_cost / max(metadata.global_max_cost, 1e-12)
    scores = (1.0 - cost_weight) * performance - cost_weight * normalized_cost
    return scores.astype(np.float32), predicted_cost.astype(np.float32)


def routing_reward_from_scores(scores: np.ndarray, true_reward: np.ndarray) -> float:
    selected = scores.argmax(axis=1)
    rows = np.arange(len(selected))
    return float(true_reward[rows, selected].mean())


def fit_score_biases(
    predicted_scores: np.ndarray,
    true_reward: np.ndarray,
    max_bias: float,
    step: float,
    passes: int,
    shrink: float,
) -> Tuple[np.ndarray, float]:
    if max_bias < 0 or step <= 0 or passes <= 0:
        raise ValueError("Bias search requires max_bias >= 0, step > 0, and passes > 0.")
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("--shrink must be between 0 and 1.")
    n_models = predicted_scores.shape[1]
    biases = np.zeros(n_models, dtype=np.float32)
    grid = np.arange(-max_bias, max_bias + step * 0.5, step, dtype=np.float32)
    best_reward = routing_reward_from_scores(predicted_scores, true_reward)
    for _ in range(passes):
        changed = False
        for model_index in range(n_models):
            original = float(biases[model_index])
            local_best_bias = original
            local_best_reward = best_reward
            for candidate in grid:
                biases[model_index] = candidate
                reward = routing_reward_from_scores(
                    predicted_scores + biases[None, :],
                    true_reward,
                )
                if reward > local_best_reward + 1e-12:
                    local_best_reward = reward
                    local_best_bias = float(candidate)
            biases[model_index] = local_best_bias
            if local_best_bias != original:
                changed = True
                best_reward = local_best_reward
        if not changed:
            break
    biases *= shrink
    final_reward = routing_reward_from_scores(
        predicted_scores + biases[None, :],
        true_reward,
    )
    return biases, final_reward


def load_calibration(path: str, model_names: Sequence[str]) -> np.ndarray:
    if not path:
        return np.zeros(len(model_names), dtype=np.float32)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source_models = [str(model) for model in data["model_names"]]
    source_biases = np.asarray(data["score_biases"], dtype=np.float32)
    bias_by_model = dict(zip(source_models, source_biases))
    missing = [model for model in model_names if model not in bias_by_model]
    if missing:
        raise ValueError(f"Calibration is missing models: {missing}")
    return np.asarray([bias_by_model[model] for model in model_names], dtype=np.float32)


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers is required for KNN correction.") from exc
    model = SentenceTransformer(model_name)
    prefix = "query: " if "e5" in model_name.lower() else ""
    inputs = [prefix + str(text) for text in texts]
    return np.asarray(
        model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )


def knn_residual_scores(
    texts: Sequence[str],
    correction_path: str,
    batch_size: int,
) -> np.ndarray:
    correction = np.load(correction_path, allow_pickle=True)
    train_embeddings = np.asarray(correction["embeddings"], dtype=np.float32)
    residuals = np.asarray(correction["residuals"], dtype=np.float32)
    model_name = str(correction["embedding_model"].item())
    k = int(correction["k"].item())
    query_embeddings = encode_texts(texts, model_name, batch_size)
    similarities = query_embeddings @ train_embeddings.T
    k = max(1, min(k, train_embeddings.shape[0]))
    neighbor_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
    neighbor_similarities = np.take_along_axis(similarities, neighbor_indices, axis=1)
    neighbor_similarities = np.maximum(neighbor_similarities, 0.0)
    denominator = neighbor_similarities.sum(axis=1, keepdims=True)
    weighted = np.einsum(
        "bk,bkm->bm",
        neighbor_similarities,
        residuals[neighbor_indices],
        optimize=True,
    )
    fallback = residuals.mean(axis=0, keepdims=True)
    return np.where(
        denominator > 1e-12,
        weighted / np.maximum(denominator, 1e-12),
        fallback,
    ).astype(np.float32)


def run_calibrate(args: argparse.Namespace) -> None:
    audit = np.load(args.audit, allow_pickle=True)
    predicted_scores = np.asarray(audit["predicted_scores"], dtype=np.float32)
    true_reward = np.asarray(audit["true_reward"], dtype=np.float32)
    model_names = [str(model) for model in audit["model_names"]]
    baseline_reward = routing_reward_from_scores(predicted_scores, true_reward)
    biases, calibrated_reward = fit_score_biases(
        predicted_scores,
        true_reward,
        args.max_bias,
        args.bias_step,
        args.passes,
        args.shrink,
    )
    output = {
        "model_names": model_names,
        "score_biases": biases.tolist(),
        "validation_reward_before": baseline_reward,
        "validation_reward_after": calibrated_reward,
        "max_bias": args.max_bias,
        "bias_step": args.bias_step,
        "passes": args.passes,
        "shrink": args.shrink,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Validation reward: {baseline_reward:.6f} -> {calibrated_reward:.6f}; "
        f"saved calibration to {args.out}"
    )

    if args.knn_out:
        texts = [str(text) for text in audit["texts"]]
        embeddings = encode_texts(
            texts,
            args.embedding_model,
            args.embedding_batch_size,
        )
        calibrated_scores = predicted_scores + biases[None, :]
        residuals = true_reward - calibrated_scores
        knn_path = Path(args.knn_out)
        knn_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            knn_path,
            embeddings=embeddings,
            residuals=residuals.astype(np.float32),
            model_names=np.asarray(model_names),
            embedding_model=np.asarray(args.embedding_model),
            k=np.asarray(args.knn_k, dtype=np.int64),
        )
        print(f"Saved validation residual KNN correction to {args.knn_out}")


def evaluate_if_labeled(
    df: pd.DataFrame,
    model_names: Sequence[str],
    predictions: Sequence[str],
    global_max_cost: float,
) -> None:
    required = {
        column
        for model in model_names
        for column in [f"{model}_performance", f"{model}_cost"]
    }
    if not required.issubset(df.columns):
        return
    performance, cost = extract_targets(df, model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    selected = np.asarray([model_to_idx[model] for model in predictions], dtype=np.int64)
    rows = np.arange(len(df))
    reward = 0.85 * performance[rows, selected] - 0.15 * cost[rows, selected] / global_max_cost
    print(f"CARROT Reward_0.85: {float(reward.mean()):.6f}")


def run_predict(args: argparse.Namespace) -> None:
    require_training_packages()
    set_seed(args.seed)
    adapter_dir = Path(args.adapter_dir)
    metadata = load_metadata(adapter_dir)
    profile = resolve_profile(args)
    profile["model_name"] = metadata.model_name
    profile["max_length"] = metadata.max_length
    validate_device(profile)

    test_df = read_csv(args.test, args.limit)
    text_column = find_text_column(test_df, args.text_column or metadata.text_column)
    texts = test_df[text_column].astype(str).tolist()
    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
    performance = predict_head("performance", texts, adapter_dir, metadata, profile, hf_token)
    standardized_cost = predict_head("cost", texts, adapter_dir, metadata, profile, hf_token)
    cost_weight = metadata.cost_weight if args.cost_weight is None else args.cost_weight
    base_scores, predicted_cost = carrot_scores(
        performance,
        standardized_cost,
        metadata,
        cost_weight,
    )
    score_biases = load_calibration(args.calibration, metadata.model_names)
    scores = base_scores + score_biases[None, :]
    if args.knn_correction:
        if not 0.0 <= args.knn_weight <= 1.0:
            raise ValueError("--knn-weight must be between 0 and 1.")
        correction = np.load(args.knn_correction, allow_pickle=True)
        correction_models = [str(model) for model in correction["model_names"]]
        if correction_models != metadata.model_names:
            raise ValueError("KNN correction model order does not match CARROT metadata.")
        residual_scores = knn_residual_scores(
            texts,
            args.knn_correction,
            args.embedding_batch_size,
        )
        scores = scores + args.knn_weight * residual_scores
    selected_idx = scores.argmax(axis=1)
    predictions = [metadata.model_names[index] for index in selected_idx]

    if "ID" not in test_df.columns:
        raise ValueError("The test CSV must contain an ID column.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": test_df["ID"], "pred_model": predictions}).to_csv(args.out, index=False)

    if args.score_out:
        score_path = Path(args.score_out)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            scores=scores,
            base_scores=base_scores,
            performance=performance.astype(np.float32),
            predicted_cost=predicted_cost,
            score_biases=score_biases,
            ids=test_df["ID"].to_numpy(),
            model_names=np.asarray(metadata.model_names),
        )
    if args.prediction_out:
        prediction_path = Path(args.prediction_out)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {}
        for idx, model in enumerate(metadata.model_names):
            columns[f"{model}_pred_performance"] = performance[:, idx]
            columns[f"{model}_pred_cost"] = predicted_cost[:, idx]
            columns[f"{model}_carrot_score"] = scores[:, idx]
        pd.DataFrame(columns).to_csv(prediction_path, index=False)

    evaluate_if_labeled(
        test_df,
        metadata.model_names,
        predictions,
        metadata.global_max_cost,
    )
    print(f"Saved CARROT submission to {args.out}")


def run_predict_soft_oracle(args: argparse.Namespace) -> None:
    require_training_packages()
    set_seed(args.seed)
    adapter_dir = Path(args.adapter_dir)
    metadata = load_soft_oracle_metadata(adapter_dir)
    profile = resolve_profile(args)
    profile["model_name"] = metadata.model_name
    profile["max_length"] = metadata.max_length
    validate_device(profile)

    test_df = read_csv(args.test, args.limit)
    if "ID" not in test_df.columns:
        raise ValueError("The test CSV must contain an ID column.")
    text_column = find_text_column(
        test_df,
        args.text_column or metadata.text_column,
    )
    texts = test_df[text_column].astype(str).tolist()
    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
    logits = predict_soft_oracle_logits(
        texts,
        adapter_dir,
        metadata,
        profile,
        hf_token,
    )
    probabilities = softmax_numpy(logits, 1.0)
    selected_idx = logits.argmax(axis=1)
    predictions = [metadata.model_names[index] for index in selected_idx]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"ID": test_df["ID"], "pred_model": predictions}
    ).to_csv(output_path, index=False)

    if args.score_out:
        score_path = Path(args.score_out)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            logits=logits,
            probabilities=probabilities,
            ids=test_df["ID"].to_numpy(),
            model_names=np.asarray(metadata.model_names),
        )
    if args.prediction_out:
        prediction_path = Path(args.prediction_out)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {"ID": test_df["ID"].to_numpy()}
        for index, model in enumerate(metadata.model_names):
            columns[f"{model}_logit"] = logits[:, index]
            columns[f"{model}_probability"] = probabilities[:, index]
        pd.DataFrame(columns).to_csv(prediction_path, index=False)

    evaluate_if_labeled(
        test_df,
        metadata.model_names,
        predictions,
        metadata.cost_denominator,
    )
    print(f"Saved soft-oracle submission to {args.out}")


def main() -> None:
    args = parse_args()
    if args.command == "list-profiles":
        print(json.dumps(MODEL_PROFILES, indent=2))
    elif args.command == "train":
        run_train(args)
    elif args.command == "train-soft-oracle":
        run_train_soft_oracle(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "predict-soft-oracle":
        run_predict_soft_oracle(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
