# Reinforcement Learning: End-to-End Production Project

> A beginner-to-production guide to **Reinforcement Learning** with **OpenAI Gym**, **DQN**, and **PPO**, built around a relatable real-life problem: **Smart Warehouse Inventory Management**.

This repository is written for absolute beginners in Machine Learning but is
structured as a real-world, production-grade codebase. By the end you will:

1. Understand the **intuition and math** of Reinforcement Learning.
2. Be comfortable using **OpenAI Gym (Gymnasium)**.
3. Implement **Deep Q-Network (DQN)** from scratch in PyTorch.
4. Implement **Proximal Policy Optimization (PPO)** from scratch in PyTorch.
5. Ship a **realistic production project** you can confidently talk about in interviews.

---

## 1. The Real-Life Problem (so you can relate)

Imagine you run a small warehouse for an e-commerce company like Amazon or Flipkart.
Every day you must decide **how much stock to order** for a popular product.

- Order **too much** -> you pay high **holding costs** (rent, spoilage, capital).
- Order **too little** -> you run out and customers go elsewhere (**stockout cost**).
- Demand is **uncertain** (weather, weekends, sales, competitors).

A human can write rules ("if stock < 20, order 50"), but those rules break when
demand patterns change. This is exactly the kind of **sequential decision-making
under uncertainty** that Reinforcement Learning is built for.

Our RL agent learns the optimal restocking policy by interacting with a simulated
warehouse environment and maximizing **long-term profit**.

We also include the classic **CartPole** tutorial so you can learn the mechanics
on a small problem first.

---

## 2. What is Reinforcement Learning? (plain English)

You already know this concept:

| Real life                   | RL term        |
| --------------------------- | -------------- |
| You (the decision maker)    | **Agent**      |
| The world you act in        | **Environment**|
| What you see                | **State / Observation** |
| What you do                 | **Action**     |
| Feedback (good / bad)       | **Reward**     |
| Your strategy               | **Policy**     |
| How good being in a state is| **Value**      |

The agent tries actions, gets rewards, and **updates its policy** to get more
reward in the future. That's it. Everything else (DQN, PPO, GAE, entropy bonus...)
is just a trick to make this learning stable, sample-efficient, and scalable.

Read [`docs/01-rl-fundamentals.md`](docs/01-rl-fundamentals.md) for a 15-minute
deep dive.

---

## 3. Project Layout

```
rl-project/
├── README.md                       <- you are here
├── requirements.txt
├── Dockerfile
├── docs/                           <- beginner-friendly theory
│   ├── 01-rl-fundamentals.md
│   ├── 02-openai-gym.md
│   ├── 03-dqn-explained.md
│   ├── 04-ppo-explained.md
│   ├── 05-real-world-inventory.md
│   └── 06-interview-prep.md
├── src/rl_project/
│   ├── agents/        <- DQN and PPO agents
│   ├── environments/  <- Custom Gym env for inventory
│   ├── networks/      <- Neural network architectures
│   ├── utils/         <- Replay buffer, rollout buffer, logger
│   ├── config/        <- YAML configs (hyperparameters)
│   └── training/      <- Training loops
├── tutorials/         <- Step-by-step scripts
│   ├── 01_cartpole_dqn.py
│   ├── 02_cartpole_ppo.py
│   └── 03_inventory_ppo.py
├── scripts/
│   ├── train.py       <- CLI entry point for training
│   ├── evaluate.py    <- Run a trained agent
│   └── serve.py       <- FastAPI model server (production)
└── tests/             <- Unit tests
```

---

## 4. Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the CartPole DQN tutorial (5 minutes)

```bash
python tutorials/01_cartpole_dqn.py
```

### Train PPO on the inventory environment

```bash
python scripts/train.py --algo ppo --env inventory --config src/rl_project/config/ppo_inventory.yaml
```

### Evaluate a trained policy

```bash
python scripts/evaluate.py --algo ppo --env inventory --checkpoint runs/ppo_inventory/best.pt
```

### Serve the trained policy as a REST API

```bash
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"stock": 40, "day_of_week": 2, "recent_demand_mean": 18, "recent_demand_std": 4}'
```

---

## 5. How to Read This Repo

If you are a beginner, go in this order:

1. `docs/01-rl-fundamentals.md` — What RL is, Markov Decision Processes.
2. `docs/02-openai-gym.md` — The Gym API (`reset`, `step`, spaces).
3. `tutorials/01_cartpole_dqn.py` — A runnable DQN on CartPole with comments.
4. `docs/03-dqn-explained.md` — The math and design choices behind DQN.
5. `tutorials/02_cartpole_ppo.py` — A runnable PPO on CartPole.
6. `docs/04-ppo-explained.md` — PPO theory.
7. `docs/05-real-world-inventory.md` — Our production problem framed as an MDP.
8. `tutorials/03_inventory_ppo.py` — Solving it end-to-end.
9. `docs/06-interview-prep.md` — Questions + answers you'll face.

---

## 6. Why This Project Is Interview-Worthy

- **Realistic problem** (inventory management) that every tech/e-commerce
  interviewer understands.
- **Two algorithms** (DQN for discrete, PPO for policy gradient) implemented
  from scratch, not copy-pasted from Stable-Baselines3.
- **Production concerns** covered: configs, logging, checkpoints, tests,
  Dockerfile, REST API serving.
- **Clear MDP formulation**: you can explain state/action/reward design, which
  is what interviewers actually probe.

See [`docs/06-interview-prep.md`](docs/06-interview-prep.md) for a list of
questions you will be asked and the best way to answer them.

---

## 7. License

MIT. Use it, fork it, put it on your resume.
