# COMA Research Project — Final Technical Reference

**Project:** Temporal Counterfactual Credit Assignment (TCCA) vs. COMA Ablation Study  
**Environment targets:** PettingZoo MPE `simple_spread_v3` · StarCraft II SMAC `3m`  
**Date:** May 2026

---

## Overview

This project implements and compares two families of multi-agent reinforcement learning algorithms:

- **COMA** — Counterfactual Multi-Agent Policy Gradients (Foerster et al., 2018)
- **TCCA** — Temporal Counterfactual Credit Assignment (this work)

The study spans two environments (cooperative navigation MPE and StarCraft II combat) and includes ablations over the temporal horizon `k` and SAC entropy coefficient `α`. All five folders are self-contained, runnable experiments.

---

## Repository Structure

```
COMA/
├── coma_mpe_baseline/        Algorithm 1 — COMA on MPE (baseline)
├── coma_smac_baseline/       Algorithm 2 — COMA on SMAC (baseline)
├── tcca_mpe_baseline/        Algorithm 3 — TCCA on MPE (main comparison)
├── tcca_sac_baseline/        Algorithm 4 — TCCA + SAC on MPE (off-policy variant)
├── tcca_smac_research/       Algorithm 5 — TCCA + SAC on SMAC (full research build)
│
├── plot_coma_vs_tcca_mpe.py          → Task 1: MPE comparison plot
├── plot_coma_vs_tcca_smac.py         → Task 2: SMAC comparison plots
├── run_k_ablation.py                 → Task 3/4: k-horizon ablation trainer + plotter
├── run_entropy_ablation.py           → Task 5: entropy ablation trainer + plotter
├── extract_hyperparameters.py        → Task 6: hyperparameter report generator
│
├── comparison_coma_vs_tcca_mpe.png           (PLOT — Task 1)
├── comparison_coma_vs_tcca_smac_winrate.png  (PLOT — Task 2)
├── comparison_coma_vs_tcca_smac_reward.png   (PLOT — Task 2)
├── ablation_k_mpe.png                        (PLOT — Task 3/4)
├── ablation_entropy_mpe.png                  (PLOT — Task 5)
└── hyperparameters_used.txt                  (REPORT — Task 6)
```

---

## Folder 1 — `coma_mpe_baseline/`

### What it implements
The original **COMA algorithm** applied to the cooperative navigation task `simple_spread_v3`. This is the primary baseline against which all TCCA variants are compared on the MPE environment.

### Algorithm
**COMA** uses a centralized-training / decentralized-execution (CTDE) framework:
- Each agent has its own **independent MLP actor** that sees only local observations and outputs a categorical action distribution.
- A single **centralized MLP critic** observes the global state and the one-hot encoded actions of every other agent (all agents *except* the one being evaluated), outputting Q-values over all possible actions for that agent.
- The **counterfactual advantage** for agent *i* is:

  `A_i(s, a_i) = Q(s, a_i, u_{-i}) − Σ_{a'} π_i(a'|o_i) · Q(s, a', u_{-i})`

  This baseline `B_i` is the policy-weighted sum over Q-values, which cancels out the contribution of other agents and isolates agent *i*'s marginal impact.
- Critic targets use **TD(0)** — single-step Bellman targets computed using the next state's Q-value.
- Training is **on-policy**: 32 full episodes are collected, the critic and all actors are updated, then the buffer is cleared.

### Files

| File | Role |
|---|---|
| `networks.py` | `Actor` — 2-layer MLP (64→64→n_actions + softmax). `CentralCritic` — MLP taking `[state ‖ other_actions_onehot]` → Q-values for all actions. |
| `coma_agent.py` | `COMAMultiAgent` — `get_actions()` samples from each actor independently; `update()` does TD(0) critic update then COMA advantage actor update for all agents. Saves/loads model weights. |
| `env_wrapper.py` | Thin wrapper over PettingZoo parallel API: returns `(obs_dict, global_state, shared_reward, done)`. Global state = concatenation of all agents' observations. |
| `train.py` | Main loop: collect 32-episode batch → update → log every 100 eps. Saves models at end. |
| `test.py` | Loads saved `.pth` weights and evaluates for N test episodes. |
| `plot.py` | Reads `train_log.txt`, applies rolling average, saves reward curve. |
| `train_log.txt` | 500 lines, each covering 100 episodes: episode number, 100-ep avg return, critic loss. |
| `actor_0/1/2.pth`, `critic.pth` | Saved model checkpoints from the completed 50,000-episode run. |

### Key results
- Trained for 50,000 episodes. Best 100-ep avg reward reached approximately **−22 to −23** (less negative = better spread).
- COMA shows clear instability bands (critic loss spikes to ~100 around ep 8k–10k) before recovering.

---

## Folder 2 — `coma_smac_baseline/`

### What it implements
**COMA adapted for StarCraft II** (`3m` — 3 Marines vs 3 Marines). This is the SMAC baseline, used for comparison against the TCCA-SAC SMAC research build.

### Why it's different from Folder 1
SMAC is **partially observable**: each agent sees only a local field of view. This means the flat MLP actor from MPE is insufficient — agents need **memory** to infer missing information. Here, the actor uses a **GRU (Gated Recurrent Unit)** to maintain a hidden state across time steps within an episode.

Additionally, SMAC has **discrete unavailable actions** (e.g., an agent can't attack a target that is out of range). These must be masked before sampling.

### Algorithm
- **Shared GRU actor** (`RNNActor`): input at each step is `[local_obs ‖ last_action_onehot ‖ agent_id_onehot]`. Passed through a linear layer → GRUCell (hidden=64) → linear → Q-logits. Unavailable actions are masked to −1e10 before softmax.
- **Centralized MLP critic** (`CentralCritic`): input is `[global_state ‖ local_obs_i ‖ other_actions_onehot ‖ agent_id_onehot]` — richer input than MPE to handle SMAC's more complex state space.
- Critic targets use **Monte Carlo returns** (full-episode discounted sum, not TD(0)).
- Actor and critic share gradient accumulation within the episode loop, then optimizer steps are taken once per episode.
- The file contains a commented-out first draft above the active implementation; the active version fixes in-place tensor assignment bugs (`masked_fill` instead of direct indexing) and gradient flow issues.

### Files

| File | Role |
|---|---|
| `networks.py` | `RNNActor` — Linear→GRUCell→Linear (hidden=64). `CentralCritic` — deeper MLP (128→128) with state+obs+others+id input. |
| `coma_agent.py` | `COMAMACAgent` — `init_hidden()` allocates GRU state; `get_actions()` runs one GRU step; `update()` unrolls the full episode, computes MC returns, accumulates gradients over all agents, steps optimizers once. |
| `runner.py` | Runs one SMAC episode: polls `get_obs()`, `get_state()`, `get_avail_actions()`, feeds to agent, steps environment, collects per-step dicts. |
| `train.py` | Accumulates episodes into batches of 4, calls `agent.update()`, logs to CSV every 100 eps. |
| `plot_results.py` | Plots win rate and reward from `training_logs.csv`. |
| `training_logs.csv` | 500 rows: episode, win rate (per 100), total reward, critic loss. |
| `smac_coma_results.png` | Generated result graph. |
| `smac/` | SMAC StarCraft II environment package. |

---

## Folder 3 — `tcca_mpe_baseline/`

### What it implements
**TCCA (Temporal Counterfactual Credit Assignment)** on MPE — the primary novel contribution. This is the direct ablation partner to `coma_mpe_baseline` on the same environment with the same network architecture.

### What TCCA adds over COMA
COMA uses TD(0) — a 1-step lookahead. TCCA generalizes this to **k-step returns**:

`G_t^(k) = Σ_{l=0}^{k-1} γ^l r_{t+l}  +  γ^k · V(s_{t+k})`

where `V(s_{t+k})` is bootstrapped from the **counterfactual baseline** of the *next* state:

`V(s_{t+k}) = Σ_a π_i(a|o_{t+k}) · Q(s_{t+k}, a, u_{-i,t+k})`

The **TCCA advantage** is then:

`C_i^t = G_t^(k) − B_i(s_t)`

This gives a tunable bias-variance tradeoff: k=1 approaches COMA's TD target, large k approaches Monte Carlo.

### Files

| File | Role |
|---|---|
| `networks.py` | Identical architecture to `coma_mpe_baseline` — same `Actor` and `CentralCritic`. |
| `tcca_agent.py` | `TCCAAgent` — `update()` iterates over each trajectory, for each agent computes k-step return G_t for every timestep, uses it as critic MSE target, and computes TCCA advantage for the policy gradient. |
| `env_wrapper.py` | Identical to `coma_mpe_baseline`. |
| `train.py` | Accumulates 32 full episode trajectories then calls `agent.update(trajectories)`. Logs every episode to CSV. Default `k=5`. |
| `plot_results.py` | Plots reward curve from CSV. |
| `training_logs.csv` | 50,000 rows (one per episode): episode, total reward, avg critic loss. |
| `tcca_results_graph.png` | Generated result graph. |
| `tcca_k1_log.csv` | **NEW** — k=1 ablation run log (50,000 eps, from `run_k_ablation.py`). |
| `tcca_k3_log.csv` | **NEW** — k=3 ablation run log (50,000 eps). |
| `tcca_k5_log.csv` | **NEW** — k=5 ablation run log (50,000 eps, matches default). |

---

## Folder 4 — `tcca_sac_baseline/`

### What it implements
**TCCA combined with Discrete Soft Actor-Critic (SAC)** on MPE. This is the off-policy variant of TCCA that adds entropy regularization and a replay buffer, bringing it closer to the full SMAC research build.

### What's new over Folder 3
1. **Off-policy training with replay buffer**: transitions are stored as full episode trajectories. Each update samples random k-step windows (`k=3`) from the buffer, enabling data reuse.
2. **Twin critics** (`TwinCentralCritics`): two independent Q-networks trained simultaneously. The minimum of their outputs (`min(Q1, Q2)`) is used for both the counterfactual baseline and the bootstrap target, reducing overestimation bias.
3. **SAC entropy regularization**: the actor loss includes an entropy bonus scaled by `α`:

   `L_actor = E[α · log π] − E[log π_taken · C_i^t]`

4. **Soft bootstrap value**: the k-step horizon value uses SAC's soft-value:

   `V_soft(s_{t+k}) = Σ_a π · (min(Q1, Q2) − α · log π)`

5. **Polyak averaging** (`τ = 0.005`): target critic parameters are updated via exponential moving average each step, maintaining a stable TD target.

### Files

| File | Role |
|---|---|
| `networks.py` | `Actor` — MLP returning `(probs, log_probs)`. `TwinCentralCritics` — contains two `CentralCritic` MLPs, returns `(q1, q2)`. |
| `tcca_sac_agent.py` | `TCCASACAgent` — off-policy update: samples k-step segments from buffer, computes soft k-step return, twin critic MSE loss, TCCA+SAC actor loss, polyak target update. |
| `replay_buffer.py` | Circular buffer of full episode trajectories. `sample(batch_size, k)` picks random episodes and random start indices, returns k-step windows. |
| `env_wrapper.py` | Same MPE wrapper. |
| `train.py` | Pushes each episode to buffer; if buffer > 10, samples 64 segments and calls `agent.update()`. |
| `plot_results.py` | Plots reward and both critic losses from CSV. |
| `training_logs.csv` | 50,000 rows: episode, reward, critic1 loss, critic2 loss. |
| `tcca_sac_graph.png` | Generated result graph. |
| `tcca_sac_alpha0_log.csv` | **NEW** — α=0 entropy-disabled ablation run (50,000 eps, from `run_entropy_ablation.py`). |

---

## Folder 5 — `tcca_smac_research/`

### What it implements
The **full production TCCA-SAC research build** for StarCraft II SMAC. This combines every technique from the previous four folders into one complete system: GRU actors (for partial observability) + twin critics + k-step soft returns + off-policy replay buffer with hidden state storage + validity masking for variable-length episodes.

### Architecture

```
Episode Data Flow:
  SMAC env → Runner.run_episode() → ReplayBuffer.insert_episode()
                                         ↓
                            ReplayBuffer.sample_k_step_segments()
                                         ↓
                               TCCAAgent.update(batch)
                                ┌──────────────┐
                                │ GRU unroll   │ ← obs, last_action, agent_id
                                │ (k+1 steps)  │
                                └──────┬───────┘
                                       │ pis (batch, k+1, agents, n_actions)
                                ┌──────▼───────┐
                                │ Twin Critics │ ← state, other_actions, agent_id
                                │ Q1, Q2       │
                                └──────┬───────┘
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼               ▼
                  k-step soft    Critic MSE       TCCA+SAC
                  returns (G_t)   update        actor update
                                       │
                                 Polyak update
                                 target critics
```

### Why every component is necessary for SMAC
- **GRU**: SMAC is partially observable — hidden state is essential for inferring full game state
- **Twin critics**: overestimation in Q-values is catastrophic for sparse-reward combat tasks
- **k-step returns**: longer horizon helps propagate sparse win/loss signals backwards through time
- **Stored hidden states**: buffer stores the GRU hidden state at each step, enabling correct re-play of any segment without full episode re-traversal
- **Validity masks**: episodes end at different lengths; masks exclude padding from loss computation
- **SAC entropy**: encourages exploration in the large discrete action space of SMAC

### Files

| File | Role |
|---|---|
| `networks.py` | `RNNActor` — Linear→GRUCell→Linear; masks unavailable actions then softmax. `CentralCritic` — 2-layer MLP. `TwinCentralCritics` — wraps two critics, returns `(q1, q2)`. |
| `tcca_agent.py` | `TCCAAgent` — full update: (1) GRU unroll k+1 steps, (2) twin Q-values from critics, (3) recursive k-step soft return backwards pass, (4) masked critic MSE loss, (5) TCCA+SAC actor loss (`advantage · log_π + α · H(π)`), (6) Polyak update. |
| `runner.py` | `Runner` — manages one SMAC episode; builds actor inputs `[obs ‖ last_action_onehot ‖ agent_id_onehot]`, maintains GRU hidden state, records all fields, appends terminal state, writes episode to buffer. |
| `replay_buffer.py` | Pre-allocated numpy arrays for capacity × max_ep_len. Stores obs, state, actions, actions_onehot, rewards, dones, avail_actions, **hidden_states**. `sample_k_step_segments()` picks random start offsets, pads short segments, generates validity masks, returns GPU tensors. |
| `train.py` | `Args` dataclass with all hyperparams. Main loop: run episode → (if buffer warm) train 4× per episode → log every 20 eps to CSV. |
| `plot_results.py` | Plots win rate and reward from `tcca_logs.csv`. |
| `tcca_logs.csv` | 2,500 rows (every 20 eps): episode, avg reward, win rate, critic MSE, actor loss. |
| `tcca_training_results.png` | Generated result graph. |

---

## Generated Plots & Reports

All of the following were generated by the analysis scripts at the project root.

### `comparison_coma_vs_tcca_mpe.png` — Task 1
Direct COMA vs TCCA (k=5) comparison on MPE. Both lines drawn from 50,000-episode runs at identical settings. 100-episode rolling average applied. Shows TCCA's superior convergence and final reward.

### `comparison_coma_vs_tcca_smac_winrate.png` — Task 2
COMA vs TCCA-SAC win rate on SMAC 3m, dual-line. COMA (500 log points every 100 eps) vs TCCA-SAC (2500 log points every 20 eps).

### `comparison_coma_vs_tcca_smac_reward.png` — Task 2
Same as above but for average episode reward.

### `ablation_k_mpe.png` — Tasks 3 & 4
Three lines: **TCCA-instant (k=1)**, TCCA (k=3), TCCA (k=5) — all trained from scratch for 50,000 episodes. Demonstrates the effect of temporal horizon on learning. k=1 is equivalent to COMA with an off-policy buffer.

### `ablation_entropy_mpe.png` — Task 5
Two lines: **TCCA-SAC α=0.005** (entropy enabled, existing run) vs **TCCA-SAC α=0** (entropy disabled, newly trained). Demonstrates the value of entropy regularization in TCCA-SAC.

### `hyperparameters_used.txt` — Task 6
Full hyperparameter table covering all four algorithm variants (COMA-MPE, TCCA-MPE, TCCA-SAC-MPE, TCCA-SAC-SMAC) plus both ablation configurations.

---

## Hyperparameter Summary

| Parameter | COMA-MPE | TCCA-MPE | TCCA-SAC-MPE | TCCA-SAC-SMAC |
|---|---|---|---|---|
| Episodes | 50,000 | 50,000 | 50,000 | 50,000 |
| Batch size | 32 eps | 32 eps | 64 segs | 32 segs |
| γ (gamma) | 0.99 | 0.99 | 0.99 | 0.99 |
| LR actor | 3e-4 | 3e-4 | 3e-4 | 5e-4 |
| LR critic | 1e-3 | 1e-3 | 1e-3 | 5e-4 |
| k (horizon) | 1 (TD-0) | **5** | **3** | **3** |
| α (entropy) | 0.01 | 0.01 | **0.005** | **0.05** |
| τ (polyak) | — | — | 0.005 | 0.005 |
| Replay buffer | — (on-policy) | — (on-policy) | 50,000 eps | 5,000 eps |
| Critic type | Single MLP | Single MLP | Twin MLP | Twin MLP |
| Actor type | MLP per agent | MLP per agent | MLP per agent | Shared GRU |
| Return target | TD(0) | k-step + CF bootstrap | k-step soft (SAC) | k-step soft (SAC) |
| Env | MPE | MPE | MPE | SMAC 3m |

---

## Algorithm Progression (Reading Order)

```
coma_mpe_baseline
      │
      │  (add k-step returns)
      ▼
tcca_mpe_baseline  ──── k-ablation: k ∈ {1, 3, 5}
      │
      │  (add SAC entropy + twin critics + replay buffer)
      ▼
tcca_sac_baseline  ──── entropy-ablation: α ∈ {0, 0.005}
      │
      │  (replace MLP actor with GRU, adapt to SMAC)
      ▼
tcca_smac_research

coma_smac_baseline ─── baseline counterpart for SMAC comparison
```
