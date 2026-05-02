# LLM Alignment, From Scratch

A self-contained curriculum that implements every major post-training
alignment technique on the same toy world, in pure Python with manual
gradients. No `torch`, no `numpy`. ~1500 lines total. Runs in <30s.

The point isn't to scale — it's to see the math actually working.
Once you can read each loss and its gradient, scaling to a real LM is
the same equations applied to log-probs from a transformer.

## The toy world

```
contexts: greet, math_q, code_q, story_req
actions:  hi, hello!, ..., x=2, print(x), once_upon, rude!, off_topic
```

A hidden `TRUE_REWARD[c][a]` is the ground-truth quality of action `a`
for context `c`. Every stage tries to push a softmax policy toward
high expected `TRUE_REWARD` — but only via the alignment-style signals
each method actually has access to (demonstrations, preferences, AI
feedback, a teacher policy).

## Stages

| # | File | Concept | Signal it uses |
|---|---|---|---|
| 1 | `01_sft.py` | Supervised fine-tuning | demonstrations `(c, a*)` |
| 2 | `02_reward_model.py` | Bradley-Terry reward model | pairwise prefs `(c, a_w, a_l)` |
| 3 | `03_ppo.py` | RLHF / PPO | RM scalar + KL to ref |
| 4 | `04_dpo.py` | Direct preference opt. | pairwise prefs (no RM, no RL) |
| 5 | `05_ipo.py` | Identity pref. opt. | pairs (squared loss, no DPO collapse) |
| 6 | `06_kto.py` | Kahneman-Tversky opt. | unpaired binary 👍/👎 |
| 7 | `07_constitutional_rlaif.py` | CAI + RLAIF | a constitution + AI judge |
| 8 | `08_distillation_kd.py` | Hinton soft-label KD | teacher logits |
| 9 | `09_minilm_distillation.py` | Relation distillation | teacher value-relations |

## Run it

```bash
python alignment/run_all.py
```

Or run individual stages — order matters because later ones load
artifacts from earlier ones (e.g., `sft_policy.json`, `rm.json`).

```bash
python alignment/01_sft.py
python alignment/02_reward_model.py
python alignment/03_ppo.py
python alignment/04_dpo.py        # alternative to 03
python alignment/05_ipo.py
python alignment/06_kto.py
python alignment/07_constitutional_rlaif.py
python alignment/08_distillation_kd.py
python alignment/09_minilm_distillation.py
```

---

## SFT → RLHF pipeline

**SFT.** Imitate a noisy expert. Loss = `-log π(a*|c)`. Gradient =
`-(1[a==a*] - π(a|c))` for the same context, 0 elsewhere. SFT cannot
exceed the demonstrator: there's no signal for "this is wrong".

**Reward model.** Humans rank pairs. Under Bradley-Terry,
`P(w > l) = σ(r(c,w) - r(c,l))`. Fit `r_φ` by minimizing
`-log σ(r_φ(c,w) - r_φ(c,l))`. The reward is identifiable only up to a
per-context constant — that constant cancels in the loss.

**PPO.** Use `r_φ` as reward, but add a KL penalty to keep the policy
close to a frozen reference (the SFT model):

```
R(c, a) = r_φ(c, a) - β · ( log π_θ(a|c) - log π_ref(a|c) )
```

Optimize with the PPO clipped surrogate:

```
L_PPO = E[ min( ρ A,  clip(ρ, 1-ε, 1+ε) A ) ],   ρ = π_new/π_old
```

The clip makes each step a *trust-region* step: huge `ρ` doesn't
amplify gradients in the wrong direction, even with stale rollouts
re-used across PPO inner epochs.

Why people hate the pipeline:
- Trains *three* models (policy, value head, RM) plus a frozen ref.
- Lots of hyperparameters (β, ε, KL controller, advantage norm, etc.).
- On-policy sampling loop is slow and brittle; reward hacking is
  constant whack-a-mole.

---

## DPO, IPO, KTO

**DPO** (Rafailov et al. 2023). The KL-regularized RL optimum has the
closed form `π*(a|c) ∝ π_ref(a|c) · exp(r(c,a)/β)`. Inverting,
`r(c,a) = β · log(π/π_ref) + β·log Z(c)`. Plug into the BT log-loss;
`log Z` cancels because both sides share the same context:

```
L_DPO = -E[ log σ( β · ( log π_θ(a_w|c) - log π_θ(a_l|c)
                       - log π_ref(a_w|c) + log π_ref(a_l|c) ) ) ]
```

So you can train preferences with the *same machinery as SFT*. No RM,
no RL loop, no value head. **That is why DPO replaced PPO in many
shops:**
- Same training loop as SFT — your existing fine-tuning infra works.
- Stable, reproducible; no sampling-loop variance.
- Half the moving parts → fewer hyperparameters → faster iteration.

DPO's downsides — and what IPO/KTO fix:

**IPO** (Azar et al. 2023). DPO's `-log σ(βh)` keeps pushing the
margin `h` toward infinity even when the preference is noisy ~55/45.
Replace with a squared loss around a finite optimum:

```
L_IPO = ( h - 1/(2β) )^2
```

The optimal margin is finite → no collapse onto winners on noisy data.

**KTO** (Ethayarajh et al. 2024). Drops the requirement for *paired*
preferences. Real chat logs come with thumbs-up/thumbs-down — unpaired
binary feedback. KTO treats each (c, a, label) example with a value
function from prospect theory:

```
r_θ(c, a) = log π_θ(a|c) - log π_ref(a|c)        # implicit reward
z_ref     = β · E[ KL(π_θ || π_ref) ]            # baseline (running)

desirable :  L = w_D · ( 1 - σ( β·r_θ - z_ref ) )
undesirable: L = w_U · ( 1 - σ( z_ref - β·r_θ ) )
```

Asymmetry between gains (above baseline) and losses (below baseline) is
the Kahneman-Tversky bit. Practically: you can use any 👍/👎 signal,
no pairing required.

When to use which:
- **High-quality paired prefs?** DPO is fine.
- **Noisy / near-tied prefs?** IPO.
- **Unpaired binary signal (production logs)?** KTO.
- **Need on-policy correction or non-pref reward (length, safety
  classifier, code execution)?** Still PPO/GRPO.

---

## Constitutional AI / RLAIF

The bottleneck of RLHF is *human* feedback: slow, expensive, biased.
Two fixes that share the same insight — let an LLM grade itself.

**Constitutional AI** (Anthropic, Bai et al. 2022).
1. **SL-CAI**: prompt the model to critique its own response under a
   sampled principle from a written constitution, then to revise.
   SFT on (prompt, revised) pairs.
2. **RL-CAI**: for two candidate responses, prompt the model to
   choose which better satisfies a principle → AI preference label →
   RLHF (or DPO) on those AI labels.

**RLAIF** (Lee et al. 2023, Google) is the same RL phase, isolated
and benchmarked: AI preferences match human prefs closely on
summarization/helpfulness tasks at a fraction of the cost.

In `07_constitutional_rlaif.py`:
- The constitution is three rules ("be polite", "stay on topic", "be
  specific").
- The "AI judge" is a noisy version of TRUE_REWARD plus the
  constitution penalty.
- We run SL-CAI (revise → SFT) then RL-CAI (DPO on AI prefs).
- The script reports the constitution-violation rate falling at each
  step.

---

## Distillation

**Knowledge distillation** (Hinton 2015). The teacher's *full output
distribution* — not just its argmax — encodes "dark knowledge"
(relative ranking of the wrong answers, calibrated uncertainty,
class similarity). Train the student to match the softened teacher:

```
p_T = softmax(z_teacher / T),   p_S = softmax(z_student / T)
L_KD = T^2 · KL(p_T || p_S)
```

Often combined with a small CE-on-hard-labels term. In
`08_distillation_kd.py` we force a real capacity gap: the student is
**low-rank** (`logits[c,a] = U[c]·V[a]`, rank=2) while the teacher is
the rank-8 DPO policy. KD beats hard-label training on the same
student.

**MiniLM** (Wang et al. 2020). Output distillation only works if your
output space is the same. For a transformer student that's
*architecturally* smaller than the teacher, MiniLM also distills
*internal* relations: attention patterns and value-vector relation
matrices. The relation matrix is `softmax(M Mᵀ / √d)` where `M ∈
{Q, K, V}` for each head; its shape is (seq_len, seq_len) and is
**independent of hidden dim** — so a thin student can match a wide
teacher's relations directly.

`09_minilm_distillation.py` demonstrates this on the action-embedding
space: a rank-2 student learns to match a rank-8 teacher's
action-action relation matrix.

Other practical distillation knobs:
- **Hard vs soft KD**: temperatures 1–4 typical; softer = more
  emphasis on non-argmax structure.
- **Sequence-level KD**: replace teacher logits with teacher-generated
  *sequences*, then SFT. Cheap, surprisingly effective.
- **On-policy distillation**: student samples, teacher scores —
  matches RL pipelines while still being a supervised loss.
- **Distillation of preferences**: DPO on (student-vs-teacher) pairs
  labeled by the teacher → distillation aligned with preferences.

---

## Reading list (mapping code → papers)

| Code | Paper |
|---|---|
| `01_sft.py` | InstructGPT (Ouyang et al. 2022) §3.1 |
| `02_reward_model.py` + `03_ppo.py` | InstructGPT §3.2–3.3; PPO (Schulman 2017) |
| `04_dpo.py` | Rafailov et al. 2023 |
| `05_ipo.py` | Azar et al. 2023 ("A General Theoretical Paradigm…") |
| `06_kto.py` | Ethayarajh et al. 2024 |
| `07_constitutional_rlaif.py` | Bai et al. 2022 (CAI), Lee et al. 2023 (RLAIF) |
| `08_distillation_kd.py` | Hinton et al. 2015 |
| `09_minilm_distillation.py` | Wang et al. 2020 (MiniLM) |
