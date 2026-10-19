# Temporal Counterfactual Credit Assignment (TCCA) for Multi-Agent Reinforcement Learning

> An ablation study comparing **COMA** (Counterfactual Multi-Agent Policy Gradients) with **TCCA** (Temporal Counterfactual Credit Assignment) across cooperative navigation and StarCraft II combat tasks.

---

## Overview

Standard COMA relies on single-step (TD(0)) value estimation for credit assignment. This project introduces **TCCA**, which generalises COMA's counterfactual baseline to **k-step returns**, providing a tunable bias–variance tradeoff:

$$G_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l\, r_{t+l} \;+\; \gamma^k \cdot V(s_{t+k})$$

where the bootstrap value uses the counterfactual baseline of the next state. The study progressively layers on **Soft Actor-Critic (SAC)** entropy regularisation, **twin critics**, **off-policy replay**, and **GRU-based actors** to scale from simple cooperative navigation to StarCraft II micromanagement.

### Environments

| Environment | Library | Task |
|---|---|---|
| **Cooperative Navigation** | PettingZoo `simple_spread_v3` | 3 agents must spread to cover 3 landmarks |
| **StarCraft II 3m** | SMAC | 3 Marines vs 3 Marines combat |

---

## Repository Structure

```
Temporal-MARL/
├── coma_mpe_baseline/          # Algorithm 1 — COMA on MPE (baseline)
├── coma_smac_baseline/         # Algorithm 2 — COMA on SMAC (baseline)
├── tcca_mpe_baseline/          # Algorithm 3 — TCCA on MPE (main comparison)
├── tcca_sac_baseline/          # Algorithm 4 — TCCA + SAC on MPE (off-policy)
├── tcca_smac_research/         # Algorithm 5 — TCCA + SAC on SMAC (full build)
│
├── plot_coma_vs_tcca_mpe.py    # MPE comparison plot
├── plot_coma_vs_tcca_smac.py   # SMAC comparison plots
├── run_k_ablation.py           # k-horizon ablation trainer + plotter
├── run_entropy_ablation.py     # Entropy ablation trainer + plotter
├── extract_hyperparameters.py  # Hyperparameter report generator
│
├── comparison_coma_vs_tcca_mpe.png
├── comparison_coma_vs_tcca_smac_winrate.png
├── comparison_coma_vs_tcca_smac_reward.png
├── ablation_k_mpe.png
├── ablation_entropy_mpe.png
└── hyperparameters_used.txt
```

---

## Algorithm Progression

```
coma_mpe_baseline
      │
      │  + k-step returns
      ▼
tcca_mpe_baseline  ─── k-ablation: k ∈ {1, 3, 5}
      │
      │  + SAC entropy + twin critics + replay buffer
      ▼
tcca_sac_baseline  ─── entropy-ablation: α ∈ {0, 0.005}
      │
      │  + GRU actor (partial observability) → SMAC
      ▼
tcca_smac_research

coma_smac_baseline ─── baseline counterpart for SMAC comparison
```

---

## Key Results

### MPE — COMA vs TCCA (k=5)
TCCA achieves **faster convergence** and **higher final reward** compared to the COMA baseline on cooperative navigation over 50,000 episodes.

### SMAC — COMA vs TCCA-SAC
TCCA-SAC reaches competitive win rates on the 3m scenario while benefiting from off-policy data reuse and entropy-driven exploration.

### Ablation: Temporal Horizon (k)
| k | Behaviour |
|---|---|
| k=1 | Equivalent to COMA with a buffer (instant credit) |
| k=3 | Moderate lookahead, balanced bias–variance |
| k=5 | Full temporal credit, best final performance on MPE |

### Ablation: Entropy Coefficient (α)
- **α = 0.005** (entropy enabled): improved exploration, more stable learning curves
- **α = 0.0** (entropy disabled): faster early progress but less robust convergence

---

## Hyperparameter Summary

| Parameter | COMA-MPE | TCCA-MPE | TCCA-SAC-MPE | TCCA-SAC-SMAC |
|---|---|---|---|---|
| Episodes | 50,000 | 50,000 | 50,000 | 50,000 |
| Batch size | 32 eps | 32 eps | 64 segments | 32 segments |
| γ (discount) | 0.99 | 0.99 | 0.99 | 0.99 |
| LR actor | 3e-4 | 3e-4 | 3e-4 | 5e-4 |
| LR critic | 1e-3 | 1e-3 | 1e-3 | 5e-4 |
| k (horizon) | 1 (TD-0) | **5** | **3** | **3** |
| α (entropy) | 0.01 | 0.01 | **0.005** | **0.05** |
| τ (polyak) | — | — | 0.005 | 0.005 |
| Critic | Single MLP | Single MLP | Twin MLP | Twin MLP |
| Actor | MLP/agent | MLP/agent | MLP/agent | Shared GRU |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sahiti3636/Temporal-MARL.git
cd Temporal-MARL

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (MPE experiments)
pip install torch pettingzoo[mpe] numpy matplotlib

# For SMAC experiments, additionally install StarCraft II and SMAC:
# See https://github.com/oxwhirl/smac for setup instructions
```

---

## Usage

### Training

Each subfolder is a self-contained experiment:

```bash
# Train COMA baseline on MPE
python coma_mpe_baseline/train.py

# Train TCCA on MPE (k=5 by default)
python tcca_mpe_baseline/train.py

# Train TCCA-SAC on MPE
python tcca_sac_baseline/train.py

# Train TCCA-SAC on SMAC 3m
python tcca_smac_research/train.py
```

### Ablation Studies

```bash
# k-horizon ablation (trains k=1, k=3, k=5 and plots)
python run_k_ablation.py

# Entropy ablation (trains α=0 and α=0.005 and plots)
python run_entropy_ablation.py
```

### Generating Plots & Reports

```bash
# COMA vs TCCA comparison on MPE
python plot_coma_vs_tcca_mpe.py

# COMA vs TCCA-SAC comparison on SMAC
python plot_coma_vs_tcca_smac.py

# Extract hyperparameter summary
python extract_hyperparameters.py
```

---

## Dependencies

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PettingZoo](https://pettingzoo.farama.org/) (with MPE extras)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SMAC](https://github.com/oxwhirl/smac) (for StarCraft II experiments)

---

## References

- Foerster, J. et al. (2018). *Counterfactual Multi-Agent Policy Gradients.* AAAI.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor.* ICML.
- Samvelyan, M. et al. (2019). *The StarCraft Multi-Agent Challenge.* AAMAS.

---

## License

This project is part of academic coursework. Please contact the author before reuse.
