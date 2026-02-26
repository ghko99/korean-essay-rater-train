"""
Enhanced Custom Trainer for Korean Essay AES
=============================================
기존 lora-self-consistency-aes의 CustomTrainer를 발전:

1. EMO + NTL + CBFL 3가지 auxiliary loss 유지
2. Dynamic loss weighting (CE 기반 자동 스케일링) 개선
   - EMA 기반 안정화: aux_dynamic 값의 급격한 변동 방지
3. Gradient norm 로깅 추가
4. Label smoothing 옵션 추가
"""

from transformers import Trainer
from typing import Any
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from .number_token_loss import NumberTokenLoss
from .class_balanced_focal_loss import ClassBalancedFocalLoss


class CustomTrainer(Trainer):
    def __init__(
        self, *args,
        ntl_weight: float = 0.3,
        emo_weight: float = 0.1,
        emo_topk: int = 32,
        num_tokenizer=None,
        cb_weight: float = 0.0,
        cb_beta: float = 0.999,
        cb_gamma: float = 2.0,
        class_counts=None,
        ema_decay: float = 0.99,
        label_smoothing: float = 0.0,
        aux_warmup_ratio: float = 0.2,
        emo_every_n_steps: int = 2,
        eval_data_collator=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._eval_data_collator = eval_data_collator
        self.ntl_weight = float(ntl_weight)
        self.emo_weight = float(emo_weight)
        self.emo_topk = int(emo_topk)
        self.cb_weight = float(cb_weight)
        self.ema_decay = float(ema_decay)
        self.label_smoothing = float(label_smoothing)
        self.aux_warmup_ratio = float(aux_warmup_ratio)
        self.emo_every_n_steps = int(emo_every_n_steps)

        self.num_tokenizer = num_tokenizer

        device = self.args.device
        vocab_size = self.model.config.vocab_size

        self.ntl_criterion = NumberTokenLoss(
            tokenizer=self.num_tokenizer, vocab_size=vocab_size,
            device=device, loss_function=torch.nn.functional.mse_loss
        )

        # Class-Balanced Focal Loss
        if self.cb_weight > 0 and class_counts is not None:
            self.cbfl_criterion = ClassBalancedFocalLoss(
                tokenizer=self.num_tokenizer,
                vocab_size=vocab_size,
                class_counts=class_counts,
                device=device,
                beta=cb_beta,
                gamma=cb_gamma,
            )
        else:
            self.cbfl_criterion = None

        self._last_logged_step = -1
        # EMA for aux_dynamic stabilization
        self._ema_aux_dynamic = None

    def get_eval_dataloader(self, eval_dataset=None):
        """평가 시에는 packing 없는 collator를 사용."""
        if self._eval_data_collator is not None:
            orig = self.data_collator
            self.data_collator = self._eval_data_collator
            loader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = orig
            return loader
        return super().get_eval_dataloader(eval_dataset)

    @staticmethod
    def _to_serializable(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().float().mean().item() if v.numel() > 1 else v.detach().cpu().item()
        if isinstance(v, dict):
            return {k: CustomTrainer._to_serializable(val) for k, val in v.items()}
        return v

    @staticmethod
    def _get_embedding_matrix(model) -> torch.Tensor:
        if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            return model.get_output_embeddings().weight
        if hasattr(model, "lm_head"):
            return model.lm_head.weight
        raise ValueError("Cannot find output embedding matrix.")

    def _emo_loss_topk(self, logits: torch.Tensor, emo_labels: torch.Tensor, model) -> torch.Tensor:
        """EMO Loss: Top-K embedding distance loss for feedback quality."""
        logits_s = logits[:, :-1, :]
        labels_s = emo_labels[:, 1:]
        mask = labels_s.ne(-100)

        if mask.sum().item() == 0:
            return logits_s.sum() * 0.0

        E = self._get_embedding_matrix(model)
        B, Tm1, V = logits_s.shape
        D = E.size(1)

        labels_safe = labels_s.masked_fill(~mask, 0)
        k = min(self.emo_topk, V)

        lse = torch.logsumexp(logits_s, dim=-1)
        top_logits, topi = torch.topk(logits_s, k=k, dim=-1)
        topv = torch.exp(top_logits - lse.unsqueeze(-1))

        gt = labels_safe.unsqueeze(-1)
        topv = topv.masked_fill(topi.eq(gt), 0.0)
        topv = topv * mask.unsqueeze(-1)

        N = B * Tm1
        topi_f = topi.reshape(N, k)
        topv_f = topv.reshape(N, k).to(dtype=E.dtype)
        labels_f = labels_safe.reshape(N)
        mask_f = mask.reshape(N)

        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        count = mask_f.sum().to(dtype=torch.float32).clamp_min(1.0)

        block_size = 128
        for s in range(0, N, block_size):
            e = min(s + block_size, N)
            idx_block = topi_f[s:e]
            w_block = topv_f[s:e]

            emb = E.index_select(0, idx_block.reshape(-1))
            emb = F.normalize(emb, p=2, dim=-1, eps=1e-12)
            emb = emb.view(e - s, k, D)
            q_block = (w_block.unsqueeze(-1) * emb).sum(dim=1)
            q_block = F.normalize(q_block, p=2, dim=-1, eps=1e-12)

            p_block = E.index_select(0, labels_f[s:e])
            p_block = F.normalize(p_block, p=2, dim=-1, eps=1e-12)

            cos = (p_block * q_block).sum(dim=-1)
            emo_tok = 1.0 - cos

            m = mask_f[s:e]
            if m.any():
                total = total + emo_tok[m].to(torch.float32).sum()

        return total / count

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ntl_labels = inputs.get("ntl_labels", inputs.get("labels", None))
        emo_labels = inputs.get("emo_labels", None)

        inputs_for_super = {k: v for k, v in inputs.items() if k not in ("ntl_labels", "emo_labels")}

        # 1) 기본 CE (with optional label smoothing)
        if self.label_smoothing > 0:
            base_loss, outputs = self._compute_ce_with_smoothing(
                model, inputs_for_super, self.label_smoothing
            )
        else:
            base_loss, outputs = super().compute_loss(
                model, inputs_for_super, return_outputs=True, **kwargs
            )

        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs must include 'logits'.")

        if not getattr(model.config, "is_encoder_decoder", False):
            logits_shifted = logits[:, :-1, :]
            labels_shifted = ntl_labels[:, 1:]
        else:
            logits_shifted, labels_shifted = logits, ntl_labels

        # 2) NTL
        if self.ntl_weight != 0:
            ntl_loss = self.ntl_criterion(logits_shifted, labels_shifted)
        else:
            ntl_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # 3) EMO (매 N step마다만 계산, 나머지는 스킵)
        if (emo_labels is not None and self.emo_weight != 0
                and self.state.global_step % self.emo_every_n_steps == 0):
            emo_loss = self._emo_loss_topk(logits, emo_labels, model)
        else:
            emo_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # 4) CBFL (OOM 방지: 2단계 분리)
        if self.cb_weight > 0 and self.cbfl_criterion is not None:
            digit_logits, score_indices = self.cbfl_criterion.extract_digit_logits(
                logits_shifted, labels_shifted
            )
            if digit_logits is not None:
                cbfl_loss = torch.utils.checkpoint.checkpoint(
                    self.cbfl_criterion.compute_focal_loss,
                    digit_logits,
                    score_indices,
                    use_reentrant=False,
                )
            else:
                cbfl_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            cbfl_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        eps = 1e-8

        # 5) Dynamic loss weighting with EMA stabilization
        aux_parts = []
        if self.ntl_weight != 0:
            aux_parts.append(ntl_loss)
        if self.emo_weight != 0:
            aux_parts.append(emo_loss)
        if self.cbfl_criterion is not None and self.cb_weight > 0:
            aux_parts.append(cbfl_loss)

        if aux_parts:
            aux_loss = sum(aux_parts)
            with torch.no_grad():
                raw_dynamic = base_loss.detach() / (aux_loss.detach() + eps)

                # EMA stabilization
                if self._ema_aux_dynamic is None:
                    self._ema_aux_dynamic = raw_dynamic.item()
                else:
                    self._ema_aux_dynamic = (
                        self.ema_decay * self._ema_aux_dynamic
                        + (1 - self.ema_decay) * raw_dynamic.item()
                    )
                aux_dynamic = torch.tensor(
                    self._ema_aux_dynamic, device=logits.device, dtype=logits.dtype
                )

            warmup_scale = 1.0
            max_steps = getattr(self.state, "max_steps", 0)
            if max_steps and max_steps > 0 and self.aux_warmup_ratio > 0:
                warmup_steps = max(1, int(max_steps * self.aux_warmup_ratio))
                warmup_scale = min(
                    1.0,
                    float(self.state.global_step + 1) / float(warmup_steps),
                )

            effective_aux = aux_dynamic * warmup_scale
            total_loss = 0.5 * (base_loss + effective_aux * aux_loss)
            current_aux_weight = float(effective_aux.item())
        else:
            total_loss = base_loss
            current_aux_weight = 0.0

        # 6) 로깅
        if self.model.training and self.state.global_step % self.args.logging_steps == 0:
            if self._last_logged_step != self.state.global_step:
                self.log({
                    "loss_ce":    self._to_serializable(base_loss),
                    "loss_ntl":   self._to_serializable(ntl_loss),
                    "loss_emo":   self._to_serializable(emo_loss),
                    "loss_cbfl":  self._to_serializable(cbfl_loss),
                    "loss_total": self._to_serializable(total_loss),
                    "aux_dynamic": current_aux_weight,
                })
                self._last_logged_step = self.state.global_step

        if isinstance(outputs, dict):
            outputs["ce_loss"] = base_loss.detach()
            outputs["ntl_loss"] = ntl_loss.detach()
            outputs["emo_loss"] = emo_loss.detach()
            outputs["cbfl_loss"] = cbfl_loss.detach()

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_ce_with_smoothing(self, model, inputs, smoothing):
        """Label smoothing CE loss."""
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        mask = shift_labels != -100
        if mask.sum() == 0:
            loss = shift_logits.sum() * 0.0
        else:
            valid_logits = shift_logits[mask]
            valid_labels = shift_labels[mask]
            log_probs = F.log_softmax(valid_logits, dim=-1)
            nll = F.nll_loss(log_probs, valid_labels, reduction='mean')
            smooth_loss = -log_probs.mean(dim=-1).mean()
            loss = (1.0 - smoothing) * nll + smoothing * smooth_loss

        return loss, outputs
