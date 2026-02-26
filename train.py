"""
Korean Essay Rater - 최종 학습 스크립트
========================================
lora-self-consistency-aes 아키텍처를 발전시킨 한국어 에세이 자동 채점 모델 학습.

주요 개선사항:
1. 데이터 경량화: 루브릭 분리로 ~60% 토큰 절약 -> max_seq_length 1536으로 축소
2. 동적 Feature 샘플링: 매 step마다 kiwi 기반 Feature 추출/3개 랜덤 샘플링
3. 모델: EXAONE-3.5-7.8B (한국어 특화 LLM, 32층 4096 hidden, 48층보다 빠름)
4. LoRA 최적화: r=32, alpha=64, 확장된 target modules (q/k/v/o/gate/up/down)
5. 학습 최적화: Cosine scheduler, warmup, gradient clipping
6. EMO + NTL(MSE) + CBFL 3중 auxiliary loss with EMA-stabilized dynamic weighting
7. Label smoothing 0.05 for regularization

사용법:
    python train.py --device_id 0

    # 전체 옵션:
    python train.py \\
        --device_id 0 \\
        --base_model_name /home/khko/models/exaone \\
        --data_dir ./data \\
        --epochs 8 \\
        --batch_size 4 \\
        --grad_accum 8 \\
        --lr 2e-4 \\
        --no_wandb
"""

import os
import sys
import random
import json
import argparse
import datetime
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from sklearn.metrics import cohen_kappa_score
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import get_last_checkpoint

from modules.number_tokenizer import AutoNumberTokenizer
from modules.custom_trainer import CustomTrainer
from modules.aes_dataset import DynamicFeatureAESDataset, AESCollatorMTL
from modules.inference_module import run_inference
from modules.evaluate_module import evaluate_results


# ──────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)


def make_output_dir(tag: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"./runs/{sanitize_name(tag)}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def compute_score_distribution(data_path: str) -> torch.Tensor:
    """학습 데이터에서 점수 분포(1-9) 계산."""
    counts = torch.zeros(9, dtype=torch.float32)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            scores = item.get("scores", [])
            for score in scores:
                s = int(score)
                if 1 <= s <= 9:
                    counts[s - 1] += 1
    return counts


def detect_precomputed_features(data_path: str, probe_lines: int = 32) -> bool:
    """JSONL 상위 일부를 확인해 features_dict 사전 추출 여부를 판단."""
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= probe_lines:
                    break
                sample = json.loads(line)
                if isinstance(sample.get("features_dict"), dict):
                    return True
    except Exception:
        return False
    return False


def _extract_scores_from_text(text: str, num_rubrics: int = 8) -> Optional[List[int]]:
    nums = [int(x) for x in re.findall(r"\d+", text)]
    nums = [n for n in nums if 1 <= n <= 9]
    if len(nums) < num_rubrics:
        return None
    return nums[:num_rubrics]


def _safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    try:
        score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        return 0.0
    if score is None or np.isnan(score):
        # 단일 클래스 케이스에서 NaN이 나올 수 있으므로 안전 처리
        if np.all(y_true == y_true[0]) and np.all(y_pred == y_pred[0]):
            return 1.0 if y_true[0] == y_pred[0] else 0.0
        return 0.0
    return float(score)


def build_qwk_compute_metrics(tokenizer, num_rubrics: int = 8):
    """생성(generate) 없이 score token 구간의 argmax 예측으로 QWK 계산."""

    def compute_metrics(eval_pred) -> Dict[str, float]:
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, (tuple, list)):
            pred_ids = pred_ids[0]
        if isinstance(label_ids, (tuple, list)):
            # label_names=["labels","ntl_labels","emo_labels"] 기준
            ntl_labels = label_ids[1] if len(label_ids) > 1 else label_ids[0]
        else:
            ntl_labels = label_ids

        pred_ids = np.asarray(pred_ids)
        ntl_labels = np.asarray(ntl_labels)

        y_true, y_pred = [], []
        for i in range(ntl_labels.shape[0]):
            mask = ntl_labels[i] != -100
            if not np.any(mask):
                continue
            gold_text = tokenizer.decode(ntl_labels[i][mask].tolist(), skip_special_tokens=True)
            pred_text = tokenizer.decode(pred_ids[i][mask].tolist(), skip_special_tokens=True)

            gold_scores = _extract_scores_from_text(gold_text, num_rubrics=num_rubrics)
            pred_scores = _extract_scores_from_text(pred_text, num_rubrics=num_rubrics)
            if gold_scores is None or pred_scores is None:
                continue
            y_true.append(gold_scores)
            y_pred.append(pred_scores)

        if not y_true:
            return {
                "average_qwk": 0.0,
                "overall_qwk": 0.0,
                "qwk_parse_rate": 0.0,
            }

        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        qwk_each = []
        metrics: Dict[str, float] = {}
        for r in range(num_rubrics):
            qwk_r = _safe_qwk(y_true[:, r], y_pred[:, r])
            qwk_each.append(qwk_r)
            metrics[f"qwk_r{r+1}"] = qwk_r

        metrics["average_qwk"] = float(np.mean(qwk_each))
        metrics["overall_qwk"] = _safe_qwk(y_true.reshape(-1), y_pred.reshape(-1))
        metrics["qwk_parse_rate"] = float(len(y_true) / max(1, ntl_labels.shape[0]))
        return metrics

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    # full logits gather를 피하고 token-id argmax만 모아 평가 메모리/시간 절약
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def init_wandb(tag: str, output_dir: str, no_wandb: bool = False):
    if no_wandb:
        return
    project_name = sanitize_name(f"korean_essay_rater_{tag}")
    run_name = sanitize_name(f"{tag}_{datetime.datetime.now().strftime('%m%d_%H%M')}")
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_RUN_NAME"] = run_name
    wandb.init(project=project_name, name=run_name, dir=output_dir, reinit=True)
    print(f"WandB -> Project: {project_name}, Run: {run_name}")


# ──────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────
def train(args):
    set_seed(42)

    model_name = os.path.expanduser(args.base_model_name)

    # Output directory
    if args.resume_checkpoint:
        output_dir = args.resume_checkpoint
        ckpt_to_resume = get_last_checkpoint(output_dir)
        if ckpt_to_resume is None:
            raise ValueError(f"No checkpoint found: {output_dir}")
    else:
        output_dir = make_output_dir(f"ce+ntl+emo+cbfl")
        ckpt_to_resume = None

    use_bf16 = torch.cuda.is_bf16_supported()
    max_seq_length = args.max_seq_length

    init_wandb("ce+ntl+emo+cbfl", output_dir, no_wandb=args.no_wandb)

    # ── Dataset ──────────────────────────────────
    train_data_path = os.path.join(args.data_dir, "train.jsonl")
    auto_precomputed = detect_precomputed_features(train_data_path)
    use_precomputed_features = args.use_precomputed_features or auto_precomputed
    if args.dataloader_num_workers >= 0:
        num_workers = args.dataloader_num_workers
    else:
        num_workers = args.auto_num_workers_precomputed if use_precomputed_features else 0

    print("Loading datasets with dynamic feature sampling...")
    print(f"  precomputed_features={use_precomputed_features}, dataloader_num_workers={num_workers}")
    train_ds = DynamicFeatureAESDataset(
        data_path=train_data_path,
        num_features=3,
        cache_features=True,
        precomputed_features=use_precomputed_features,
    )
    valid_ds = DynamicFeatureAESDataset(
        data_path=os.path.join(args.data_dir, "valid.jsonl"),
        num_features=3,
        cache_features=True,
        precomputed_features=use_precomputed_features,
    )
    test_ds = DynamicFeatureAESDataset(
        data_path=os.path.join(args.data_dir, "test.jsonl"),
        num_features=3,
        cache_features=True,
        precomputed_features=use_precomputed_features,
    )
    print(f"  Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    # ── Tokenizer ────────────────────────────────
    tokenizer = AutoNumberTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = AESCollatorMTL(
        tokenizer,
        max_seq_length=max_seq_length,
        score_token_len=16,
        pad_to_max=args.pad_to_max,
        pack=args.pack_sequences,
    )
    # 평가 시에는 packing 비활성화 (compute_metrics가 1 sample = 1 row 가정)
    eval_data_collator = None
    if args.pack_sequences:
        eval_data_collator = AESCollatorMTL(
            tokenizer,
            max_seq_length=max_seq_length,
            score_token_len=16,
            pad_to_max=args.pad_to_max,
            pack=False,
        )
        print("  Sequence packing enabled (train only).")

    # ── Model + LoRA ─────────────────────────────
    print(f"Loading model: {model_name}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_load_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.attn_implementation == "auto":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                **model_load_kwargs, attn_implementation="flash_attention_2"
            )
            print("  Attention: flash_attention_2 enabled.")
        except Exception as e:
            print(f"  [WARN] flash_attention_2 unavailable -> fallback ({e})")
            model = AutoModelForCausalLM.from_pretrained(**model_load_kwargs)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                **model_load_kwargs, attn_implementation=args.attn_implementation
            )
            print(f"  Attention: {args.attn_implementation}")
        except Exception as e:
            print(f"  [WARN] attn_implementation={args.attn_implementation} failed -> fallback ({e})")
            model = AutoModelForCausalLM.from_pretrained(**model_load_kwargs)

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA: 확장된 target modules + 높은 rank
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if args.use_torch_compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"  torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"  [WARN] torch.compile disabled due to compatibility issue: {e}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Training Arguments ───────────────────────
    visible_gpus = torch.cuda.device_count()
    if visible_gpus > 1:
        print(f"  Multi-GPU detected: {visible_gpus} GPUs (DDP path)")
    else:
        print("  Single GPU detected.")

    fsdp_value = None
    if args.enable_fsdp:
        if visible_gpus > 1:
            fsdp_value = "full_shard auto_wrap"
            print("  FSDP enabled.")
        else:
            print("  [WARN] enable_fsdp=True but only 1 GPU visible -> FSDP disabled.")

    save_strategy = args.eval_strategy
    if args.save_strategy != "match_eval":
        save_strategy = args.save_strategy

    trainer_extra_kwargs = {}
    if args.eval_strategy == "steps":
        if args.eval_steps <= 0:
            raise ValueError("eval_strategy=steps requires --eval_steps > 0")
        trainer_extra_kwargs["eval_steps"] = args.eval_steps
        trainer_extra_kwargs["save_steps"] = args.eval_steps

    trainer_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        num_train_epochs=args.epochs,
        eval_strategy=args.eval_strategy,
        save_strategy=save_strategy,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=(not use_bf16),
        save_total_limit=2,
        report_to="wandb" if not args.no_wandb else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_average_qwk",
        greater_is_better=True,
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        label_names=["labels", "ntl_labels", "emo_labels"],
        ddp_find_unused_parameters=False if visible_gpus > 1 else None,
        fsdp=fsdp_value,
        **trainer_extra_kwargs,
    )

    # ── Score distribution for CBFL ──────────────
    class_counts = compute_score_distribution(train_data_path)
    print(f"Score distribution (1-9): {class_counts.tolist()}")

    # ── Trainer ──────────────────────────────────
    trainer = CustomTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=build_qwk_compute_metrics(tokenizer, num_rubrics=8),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        num_tokenizer=tokenizer,
        ntl_weight=-1,      # dynamic mode
        emo_weight=-1,       # dynamic mode
        emo_topk=args.emo_topk,
        cb_weight=1.0,
        cb_beta=0.9999,
        cb_gamma=2.0,
        class_counts=class_counts,
        ema_decay=0.99,
        label_smoothing=args.label_smoothing,
        aux_warmup_ratio=args.aux_warmup_ratio,
        emo_every_n_steps=args.emo_every_n_steps,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        eval_data_collator=eval_data_collator,
    )

    # ── Training ─────────────────────────────────
    if args.dry_run:
        print("\nDry run successful. Model, tokenizer, and data loaded. Skipping training.")
        return

    if ckpt_to_resume:
        print(f"Resuming from: {ckpt_to_resume}")
        trainer.train(resume_from_checkpoint=ckpt_to_resume)
    else:
        trainer.train()

    if not args.no_wandb:
        wandb.finish()

    print(f"Training complete. Saving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Inference ────────────────────────────────
    if not args.skip_inference:
        print("\nRunning inference on test set...")
        # test_ds를 instruction/output dict 형태로 변환
        test_data_for_inference = []
        for i in range(len(test_ds)):
            sample = test_ds[i]
            test_data_for_inference.append({
                "instruction": sample["instruction"],
                "output": sample["output"],
            })

        csv_path = run_inference(
            model=trainer.model,
            tokenizer=tokenizer,
            test_dataset=test_data_for_inference,
            out_dir=output_dir,
        )
        evaluate_results(str(csv_path), save_dir=output_dir)

    # ── Config 저장 ──────────────────────────────
    config = {
        "base_model": args.base_model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": lora_config.target_modules,
        "max_seq_length": max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "loss_type": "mse",
        "label_smoothing": args.label_smoothing,
        "losses": ["CE", "NTL", "EMO", "CBFL"],
        "dynamic_feature_sampling": True,
        "num_features": 3,
        "precomputed_features": use_precomputed_features,
        "pad_to_max": args.pad_to_max,
        "pack_sequences": args.pack_sequences,
        "optim": args.optim,
        "eval_metric": "average_qwk",
    }
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nAll done. Results saved to: {output_dir}")


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean Essay Rater Training")

    # Model
    parser.add_argument("--base_model_name", type=str,
                        default="/home/khko/models/exaone",
                        help="베이스 모델 경로")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="경량화된 데이터 디렉토리")
    parser.add_argument("--max_seq_length", type=int, default=1536,
                        help="최대 시퀀스 길이")
    parser.add_argument("--use_precomputed_features", action="store_true",
                        help="JSONL의 features_dict를 사용해 kiwi 분석 생략")
    parser.add_argument("--dataloader_num_workers", type=int, default=-1,
                        help="DataLoader worker 수 (-1이면 자동: precomputed 시 auto_num_workers_precomputed)")
    parser.add_argument("--auto_num_workers_precomputed", type=int, default=4,
                        help="precomputed_features 사용 시 자동 worker 수")
    parser.add_argument("--pad_to_max", action="store_true",
                        help="동적 패딩 대신 max_seq_length 고정 패딩 사용")
    parser.add_argument("--pack_sequences", action="store_true",
                        help="Sequence packing으로 패딩 낭비를 줄여 throughput 향상")

    # Training
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_strategy", type=str, default="match_eval", choices=["match_eval", "epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--aux_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--emo_every_n_steps", type=int, default=2)
    parser.add_argument("--emo_topk", type=int, default=32)
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--attn_implementation", type=str, default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--enable_fsdp", action="store_true")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="옵티마이저 (HF TrainingArguments 형식)")

    # Infra
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="예: '0' 또는 '0,1'. 미지정 시 현재 환경값 유지")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train(args)
