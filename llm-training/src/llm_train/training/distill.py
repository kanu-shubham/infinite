"""Knowledge distillation training.

Two modes:

``logit``
    Logit-level distillation. Given identical inputs, minimize
    ``alpha_ce * CE(student, labels) + alpha_kd * T^2 * KL(student_T || teacher_T)``
    where ``*_T`` denotes softening by temperature ``T``. This is the
    classic Hinton et al. formulation extended to causal LM with
    per-token averaging over non-masked positions only.

``sequence``
    Sequence-level distillation (Kim & Rush, 2016). The teacher generated
    the response offline; the student just does SFT on it. Equivalent to
    SFT — the distillation step is in the data preparation, not the loss.
    For this mode, run ``run_sft`` with teacher-generated data instead.

This module implements the ``logit`` mode. The entrypoint raises for
``sequence`` with guidance, since it's cleaner to keep SFT-equivalent
training in the SFT path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, Trainer

from llm_train.data.distill import build_distill_datasets
from llm_train.models.loader import apply_lora, load_model_and_tokenizer
from llm_train.training.collators import IGNORE_INDEX, CausalCollator
from llm_train.training.common import build_hf_training_args, snapshot_run
from llm_train.utils.logging import get_logger
from llm_train.utils.seed import set_global_seed

if TYPE_CHECKING:
    from llm_train.config import DistillConfig

log = get_logger(__name__)


class LogitDistillationTrainer(Trainer):
    """``transformers.Trainer`` with a custom loss that blends CE and KL.

    Teacher is frozen and runs in ``eval()`` mode — never updated, never
    subject to gradient checkpointing. For memory efficiency it lives in
    the same process and uses ``torch.inference_mode`` during the forward.
    """

    def __init__(
        self,
        *args: Any,
        teacher_model: PreTrainedModel,
        temperature: float,
        alpha_ce: float,
        alpha_kd: float,
        top_k_logits: int | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.temperature = float(temperature)
        self.alpha_ce = float(alpha_ce)
        self.alpha_kd = float(alpha_kd)
        self.top_k_logits = top_k_logits

    def _move_teacher(self, device: torch.device, dtype: torch.dtype) -> None:
        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device=device, dtype=dtype)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        student_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        ce_loss = student_outputs.loss  # already ignores IGNORE_INDEX

        # Shift so that at position t we predict token t+1.
        student_logits = student_outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss_mask = (shift_labels != IGNORE_INDEX).to(student_logits.dtype)

        self._move_teacher(student_logits.device, student_logits.dtype)
        with torch.inference_mode():
            teacher_logits = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[:, :-1, :]

        T = self.temperature
        s_log = F.log_softmax(student_logits / T, dim=-1)
        t_prob = F.softmax(teacher_logits / T, dim=-1)

        if self.top_k_logits is not None:
            # Restrict KL to the teacher's top-k tokens per position — cheaper
            # and reduces noise from the long tail.
            topv, topi = t_prob.topk(self.top_k_logits, dim=-1)
            topv = topv / topv.sum(dim=-1, keepdim=True)
            s_log_topk = s_log.gather(-1, topi)
            per_tok_kl = -(topv * s_log_topk).sum(dim=-1) - (
                -(topv * topv.clamp_min(1e-12).log()).sum(dim=-1)
            )
        else:
            per_tok_kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)

        # Mask out prompt/pad positions, average over valid tokens.
        denom = loss_mask.sum().clamp_min(1.0)
        kd_loss = (per_tok_kl * loss_mask).sum() / denom
        kd_loss = kd_loss * (T * T)  # temperature scaling (Hinton)

        loss = self.alpha_ce * ce_loss + self.alpha_kd * kd_loss

        # Log components (picked up by HF logging).
        self.log({"loss/ce": ce_loss.detach().item(), "loss/kd": kd_loss.detach().item()})
        return (loss, student_outputs) if return_outputs else loss


def run_distill(cfg: DistillConfig, *, source_config_path: str | None = None) -> str:
    if cfg.mode == "sequence":
        raise ValueError(
            "sequence-level distillation: generate data with the teacher "
            "(see scripts/generate_teacher_data.py) and run run_sft on it."
        )

    set_global_seed(cfg.training.seed)
    out = snapshot_run(cfg, cfg.training.output_dir, source_config_path=source_config_path)

    student, tokenizer = load_model_and_tokenizer(cfg.student, cfg.tokenizer, for_training=True)
    student = apply_lora(student, cfg.lora)

    teacher, _ = load_model_and_tokenizer(cfg.teacher, cfg.tokenizer, for_training=False)

    train_ds, eval_ds = build_distill_datasets(cfg.data, cfg.tokenizer, tokenizer)

    training_args = build_hf_training_args(cfg.training, cfg.optimizer)
    collator = CausalCollator(tokenizer=tokenizer)

    trainer = LogitDistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        teacher_model=teacher,
        temperature=cfg.temperature,
        alpha_ce=cfg.alpha_ce,
        alpha_kd=cfg.alpha_kd,
        top_k_logits=cfg.top_k_logits,
    )

    log.info(
        "Starting distillation: student=%s teacher=%s T=%.2f α_ce=%.2f α_kd=%.2f",
        cfg.student.name_or_path, cfg.teacher.name_or_path,
        cfg.temperature, cfg.alpha_ce, cfg.alpha_kd,
    )
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    log.info("Distillation complete → %s", out)
    return str(out)
