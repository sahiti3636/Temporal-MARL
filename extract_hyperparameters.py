"""
Task 6: Extract and report all hyperparameters used across the three TCCA training scripts.
Writes hyperparameters_used.txt in the project root.
"""

lines = []
lines.append("=" * 65)
lines.append("  COMA PROJECT — HYPERPARAMETERS USED IN EXPERIMENTS")
lines.append("  (auto-extracted from train.py files)")
lines.append("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
lines.append("\n\n[1] COMA Baseline — MPE  (coma_mpe_baseline/train.py)")
lines.append("-" * 55)
lines += [
    "  Environment       : PettingZoo simple_spread_v3",
    "  Agents (N)        : 3",
    "  Max cycles/ep     : 25",
    "  Num episodes      : 50,000",
    "  Batch size        : 32 episodes (on-policy rollout batch)",
    "  Gamma (γ)         : 0.99",
    "  LR actor          : 3e-4",
    "  LR critic         : 1e-3",
    "  Entropy coef (ent): 0.01",
    "  Critic type       : Single centralized MLP",
    "  Actor type        : Independent MLP per agent",
    "  Return target     : TD(0)",
    "  Alpha (SAC)       : N/A",
    "  Tau (polyak)      : N/A",
    "  Replay buffer     : N/A (on-policy)",
    "  k (horizon)       : N/A (TD-0 only)",
]

# ─────────────────────────────────────────────────────────────────────────────
lines.append("\n\n[2] TCCA Baseline — MPE  (tcca_mpe_baseline/train.py)")
lines.append("-" * 55)
lines += [
    "  Environment       : PettingZoo simple_spread_v3",
    "  Agents (N)        : 3",
    "  Max cycles/ep     : 25",
    "  Num episodes      : 50,000",
    "  Batch size        : 32 episodes (on-policy trajectory batch)",
    "  Gamma (γ)         : 0.99",
    "  LR actor          : 3e-4",
    "  LR critic         : 1e-3",
    "  Entropy coef      : 0.01",
    "  k (return horizon): 5",
    "  Critic type       : Single centralized MLP",
    "  Actor type        : Independent MLP per agent",
    "  Return target     : k-step return + counterfactual bootstrap",
    "  Alpha (SAC)       : N/A",
    "  Tau (polyak)      : N/A",
    "  Replay buffer     : N/A (on-policy)",
]

# ─────────────────────────────────────────────────────────────────────────────
lines.append("\n\n[3] TCCA-SAC Baseline — MPE  (tcca_sac_baseline/train.py)")
lines.append("-" * 55)
lines += [
    "  Environment       : PettingZoo simple_spread_v3",
    "  Agents (N)        : 3",
    "  Max cycles/ep     : 25",
    "  Num episodes      : 50,000",
    "  Batch size        : 64 k-step segments",
    "  Gamma (γ)         : 0.99",
    "  LR actor          : 3e-4",
    "  LR critic         : 1e-3",
    "  Alpha (α, entropy): 0.005 (fixed, no auto-tuning)",
    "  Tau (τ, polyak)   : 0.005",
    "  k (return horizon): 3",
    "  Critic type       : Twin centralized MLP (min(Q1,Q2))",
    "  Actor type        : Independent MLP per agent",
    "  Return target     : k-step soft return (SAC-style)",
    "  Replay buffer     : 50,000 episodes (off-policy)",
    "  Warmup            : 10 episodes before first update",
]

# ─────────────────────────────────────────────────────────────────────────────
lines.append("\n\n[4] TCCA-SAC Research — SMAC  (tcca_smac_research/train.py)")
lines.append("-" * 55)
lines += [
    "  Environment       : StarCraft II SMAC 3m (3 Marines)",
    "  Agents            : 3 (set by env)",
    "  Max ep length     : env limit (default 100)",
    "  Num episodes      : 50,000",
    "  Batch size        : 32 k-step segments",
    "  Train interval    : every 1 episode",
    "  Updates per train : 4",
    "  Gamma (γ)         : 0.99",
    "  LR (shared)       : 5e-4",
    "  Alpha (α, entropy): 0.05",
    "  Tau (τ, polyak)   : 0.005",
    "  Grad norm clip    : 10.0",
    "  k (return horizon): 3",
    "  RNN hidden dim    : 64",
    "  Critic hidden dim : 64",
    "  Critic type       : Twin centralized MLP",
    "  Actor type        : Shared GRU (RNNActor)",
    "  Return target     : k-step soft return (SAC-style)",
    "  Replay buffer     : 5,000 episodes (off-policy)",
    "  Log interval      : every 20 episodes",
]

# ─────────────────────────────────────────────────────────────────────────────
lines.append("\n\n[5] k-Ablation Study — TCCA MPE  (run_k_ablation.py)")
lines.append("-" * 55)
lines += [
    "  Same config as [2] except k swept over: {1, 3, 5}",
    "  k=1 label: 'TCCA-instant (k=1)'  — reduces to COMA+buffer",
    "  k=3 label: 'TCCA (k=3)'",
    "  k=5 label: 'TCCA (k=5)'  — matches default [2] run",
]

lines.append("\n\n[6] Entropy Ablation — TCCA-SAC MPE  (run_entropy_ablation.py)")
lines.append("-" * 55)
lines += [
    "  Same config as [3] except alpha swept over: {0.0, 0.005}",
    "  alpha=0.0   : entropy completely disabled",
    "  alpha=0.005 : entropy-regularized (standard run from [3])",
]

lines.append("\n" + "=" * 65)

out = "hyperparameters_used.txt"
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Saved -> {out}")
print("\n".join(lines))
