# Direct Preference Optimization (DPO)

DPO aligns a model to human preferences without the reward-model + PPO
machinery of classic RLHF. It starts from an **SFT checkpoint** and
uses `(prompt, chosen, rejected)` pairs to nudge the policy toward
preferred responses while a KL penalty (via the reference model) keeps
it from drifting too far.

## Objective

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}\left[
\log \sigma\!\left(
  \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}
- \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}
\right)\right]
$$

- $\pi_\theta$ — the policy being trained
- $\pi_\text{ref}$ — frozen reference (usually the SFT checkpoint)
- $y_w$ — chosen response, $y_l$ — rejected
- $\beta$ — controls the KL strength; higher $\beta$ keeps the policy
  closer to the reference.

When LoRA is enabled we **skip loading a second reference model** —
TRL toggles the adapter off to recover $\pi_\text{ref}$, cutting memory
roughly in half.

## Data format

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

`prompt` can also be a chat-formatted list of messages — the
tokenizer's chat template is applied automatically.

## Loss variants

Set via `loss_type:` in the config:

| Type        | Paper / notes                                                |
| ----------- | ------------------------------------------------------------ |
| `sigmoid`   | Original DPO (Rafailov et al., 2023). Default.               |
| `ipo`       | Azar et al. — fixes DPO's overconfidence on noisy preferences |
| `hinge`     | Hinge-loss variant; less sensitive to outliers               |
| `kto_pair`  | Kahneman-Tversky Optimization, pairwise form                 |

## Configuration

[`configs/dpo/qwen_0_5b.yaml`](../configs/dpo/qwen_0_5b.yaml). Key knobs:

| Field              | Notes                                                       |
| ------------------ | ----------------------------------------------------------- |
| `model.name_or_path` | Should point at your SFT checkpoint (merged or LoRA-wrapped) |
| `beta`             | 0.1 (stronger updates) – 0.5 (more conservative)            |
| `optimizer.learning_rate` | **5e-6 – 1e-5** — DPO is LR-sensitive; don't reuse SFT LR |
| `max_prompt_length` / `max_length` | Prompt truncated from the left; completions truncated from the right |
| `lora.enabled`     | True = implicit reference model (TRL toggles the adapter)   |

## Running

```bash
llm-train dpo --config configs/dpo/qwen_0_5b.yaml
```

## What "good" looks like

During training, TRL logs `rewards/chosen`, `rewards/rejected`, and
`rewards/accuracies` (fraction of pairs where chosen scored higher).
Healthy runs show:

- `rewards/accuracies` trending **up** toward 0.7–0.9
- `rewards/chosen - rewards/rejected` (the "margin") **widening**
- `loss` gently decreasing, not collapsing to zero

If `loss` → 0 and `rewards/accuracies` = 1.0 in 50 steps, you overfit —
reduce LR, add data, or lower epochs.

## Common pitfalls

- **Huge reward gap + garbage outputs** — the model found a shortcut
  (e.g. longer = preferred). Curate data to decorrelate length from
  quality, or enable length normalization (`loss_type: ipo`).
- **No improvement** — LR likely too low; try 1e-5.
- **Explicit ref model with LoRA** — wastes memory; leave `ref_model: null`.
- **Reference drift** — if you DPO multiple rounds, each round's
  reference should be the *previous round's* final model, not the
  original SFT.
