"""FastAPI server that wraps a trained PPO agent as a REST endpoint.

Run:
    uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

Set the checkpoint path via the RL_CHECKPOINT env variable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rl_project.agents import PPOAgent
from rl_project.environments.inventory_env import InventoryConfig


class InventoryState(BaseModel):
    stock: int = Field(..., ge=0, description="Current on-hand stock")
    on_order: int = Field(0, ge=0, description="Units already ordered, arriving next")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday ... 6=Sunday")
    recent_demand_mean: float = Field(..., ge=0.0, description="7-day rolling mean of demand")
    recent_demand_std: float = Field(..., ge=0.0, description="7-day rolling std of demand")


class OrderRecommendation(BaseModel):
    order_units: int
    raw_action: float
    notes: str


app = FastAPI(title="RL Inventory Advisor", version="0.1.0")
_agent: PPOAgent | None = None
_config = InventoryConfig()


def _load_agent() -> PPOAgent:
    global _agent
    if _agent is None:
        ckpt = os.environ.get("RL_CHECKPOINT")
        if not ckpt or not Path(ckpt).exists():
            raise HTTPException(
                status_code=503,
                detail="Checkpoint not found. Set RL_CHECKPOINT to a trained .pt file.",
            )
        agent = PPOAgent(obs_dim=5, action_dim=1, discrete=False)
        agent.load(ckpt)
        _agent = agent
    return _agent


def _encode(state: InventoryState) -> np.ndarray:
    cap = float(_config.capacity)
    mean_base = float(np.mean(_config.base_demand_by_weekday))
    return np.array(
        [
            state.stock / cap,
            state.on_order / cap,
            state.day_of_week / 6.0,
            (state.recent_demand_mean - mean_base) / mean_base,
            state.recent_demand_std / mean_base,
        ],
        dtype=np.float32,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=OrderRecommendation)
def predict(state: InventoryState) -> OrderRecommendation:
    agent = _load_agent()
    obs = _encode(state)
    raw = float(np.asarray(agent.greedy_action(obs)).reshape(-1)[0])
    raw = float(np.clip(raw, 0.0, 1.0))
    # Respect capacity in the response.
    max_order = max(0, _config.capacity - state.stock - state.on_order)
    units = int(min(max_order, round(raw * _config.capacity)))
    return OrderRecommendation(
        order_units=units,
        raw_action=raw,
        notes=(
            "Units clipped to warehouse capacity minus current pipeline. "
            "Integrate with your ERP for hard business rules before committing the PO."
        ),
    )
