# 5. Real-World Project — Smart Warehouse Inventory Management

> This is the story you will tell in your interview. Memorize the MDP.

## 5.1 Business Problem

You are a data scientist at an e-commerce company. One SKU (say, a popular
smartphone case) is managed by a warehouse that must decide **each morning how
many units to order** from the supplier.

- **Lead time**: the order arrives the next morning.
- **Demand**: random, depends on day of week and promotions.
- **Holding cost** `h` per unit per day (capital tied up, shelf space).
- **Stockout cost** `p` per unit of unmet demand (lost sale + brand damage).
- **Order cost** `c` per unit.
- **Selling price** `s` per unit.
- **Warehouse capacity** `C` units.

Goal: maximize expected profit over a long horizon.

Rule-based solutions (reorder point, EOQ) work only under strong assumptions
(stationary demand, known distribution). Real-world demand is non-stationary
-> RL.

## 5.2 The MDP Formulation

**State `s`** (observation the agent sees each morning):
- `stock_on_hand`: current inventory level (int, 0..C).
- `on_order`: units ordered yesterday, arriving today (int).
- `day_of_week`: 0..6 (one-hot or scalar).
- `recent_demand_mean`: rolling 7-day mean.
- `recent_demand_std`: rolling 7-day std.

We normalize these for neural net input.

**Action `a`**:
- For DQN (discrete): choose order quantity in `{0, 5, 10, 20, 30, 50}`.
- For PPO (continuous): a scalar in `[0, 1]` mapped to `[0, C]`.

**Reward `r`**:
```
sales        = min(stock_after_arrival, demand)
unmet_demand = max(0, demand - stock_after_arrival)
new_stock    = stock_after_arrival - sales
r = s * sales - c * order - h * new_stock - p * unmet_demand
```

**Transition**:
```
stock_t+1 = new_stock
on_order_t+1 = action
```

**Terminal**: episode ends after `T = 60` days (two months of business).

**Discount** `gamma = 0.99`.

## 5.3 Why This Is a Good RL Problem

- **Sequential**: today's decision affects tomorrow's state.
- **Stochastic**: demand is random -> can't solve with one LP.
- **Non-myopic**: greedy (max profit today) != max long-term profit.
- **Interpretable reward**: dollars. You can show a CFO the result.
- **Partial observability handled easily** by including rolling stats.

## 5.4 Baselines To Compare Against

When you show this in interviews, do **not** compare RL only to "random".
Compare to:

1. **Do nothing** (never order). Terrible, but a sanity floor.
2. **Newsvendor policy**: order `F^-1(p/(p+h))` of the demand CDF. Classic OR.
3. **(s, S) policy**: when stock falls below `s`, order up to `S`. Tune with grid search.
4. **DQN agent**.
5. **PPO agent**.

The point is to show that RL **matches or beats a well-tuned baseline under
non-stationary demand** — that's the business-relevant result.

## 5.5 Production Considerations (interview gold)

- **Simulator fidelity**: an RL policy is only as good as the simulator. In
  practice we'd fit demand from historical data and validate distributions.
- **Offline RL**: if you can't safely deploy an exploring policy in prod, use
  CQL / BCQ on logged data.
- **Safety constraints**: wrap the policy with hard rules (e.g. "never order
  more than capacity", "cap daily spend") — RL is soft guidance.
- **Evaluation**: use an independent **test period** and report profit,
  stockout rate, holding cost — not just mean reward.
- **Monitoring in production**: drift in demand distribution should trigger
  retraining. Log state/action/reward tuples continuously.
- **Human-in-the-loop**: in early rollout, RL suggests, human approves.
- **A/B testing**: split warehouses into control (rule) and treatment (RL),
  measure profit delta.

## 5.6 Extensions

- **Multiple SKUs**: shared capacity, substitution effects -> multi-agent RL.
- **Price as action**: dynamic pricing joint with ordering.
- **Supplier uncertainty**: stochastic lead time.
- **Lost-sales vs backlog**: changes reward structure.

These make great follow-up questions when the interviewer asks "how would you
extend this?".
