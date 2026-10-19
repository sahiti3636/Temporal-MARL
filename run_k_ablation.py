"""
Tasks 3 & 4: k-Ablation Study — TCCA MPE
Trains TCCA with k=1 ("TCCA-instant"), k=3, and k=5 for num_episodes episodes each.
Saves logs to tcca_mpe_baseline/tcca_k1_log.csv, tcca_k3_log.csv, tcca_k5_log.csv
Then plots all three on ablation_k_mpe.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Add tcca_mpe_baseline to path so we can import its modules ─────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tcca_mpe_baseline"))
from env_wrapper import SimpleSpreadWrapper
from tcca_agent import TCCAAgent

NUM_EPISODES = 50000
BATCH_SIZE   = 32
K_VALUES     = [1, 3, 5]
OUTDIR       = "tcca_mpe_baseline"

# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────
def run_training(k_val):
    csv_path = os.path.join(OUTDIR, f"tcca_k{k_val}_log.csv")
    print(f"\n{'='*60}")
    print(f"  Training TCCA  k={k_val}  →  {csv_path}")
    print(f"{'='*60}")

    env   = SimpleSpreadWrapper(N=3, max_cycles=25)
    agent = TCCAAgent(env.n_agents, env.obs_dim, env.state_dim,
                      env.n_actions, k=k_val)

    df = pd.DataFrame(columns=["Episode", "Total Reward", "Average Critic Loss"])
    df.to_csv(csv_path, index=False)

    trajectories  = []
    ep_returns    = []
    last_cl       = 0.0

    for ep in range(1, NUM_EPISODES + 1):
        obs, state = env.reset()
        done       = False
        ep_return  = 0.0
        trajectory = []

        while not done:
            obs_list   = [obs[env.agents[i]] for i in range(env.n_agents)]
            actions    = agent.get_actions(obs_list)
            action_dict = {env.agents[i]: actions[i] for i in range(env.n_agents)}
            next_obs, next_state, reward, done = env.step(action_dict)
            trajectory.append((obs_list, state, actions, reward,
                                next_obs, next_state, done))
            obs, state  = next_obs, next_state
            ep_return  += reward

        trajectories.append(trajectory)
        ep_returns.append(ep_return)

        if len(trajectories) >= BATCH_SIZE:
            last_cl = agent.update(trajectories)
            trajectories = []

        pd.DataFrame([{
            "Episode": ep,
            "Total Reward": ep_return,
            "Average Critic Loss": last_cl
        }]).to_csv(csv_path, mode='a', header=False, index=False)

        if ep % 1000 == 0:
            avg = np.mean(ep_returns[-100:])
            print(f"  k={k_val} | Ep {ep:05d} | Avg(100)={avg:8.2f} | CL={last_cl:.4f}")

    env.close()
    print(f"  Done. Log → {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Train each k (skip if log already exists to allow re-use)
# ─────────────────────────────────────────────────────────────────────────────
log_paths = {}
for k in K_VALUES:
    path = os.path.join(OUTDIR, f"tcca_k{k}_log.csv")
    if os.path.exists(path):
        nrows = sum(1 for _ in open(path)) - 1
        print(f"Found existing {path} ({nrows} rows). Skipping training for k={k}.")
        log_paths[k] = path
    else:
        log_paths[k] = run_training(k)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
})

COLORS = {1: "#f78166", 3: "#d2a8ff", 5: "#3fb950"}
SMOOTH = 5   # over 100-ep samples

LABELS = {
    1: "TCCA-instant (k=1)",
    3: "TCCA (k=3)",
    5: "TCCA (k=5)",
}

fig, ax = plt.subplots(figsize=(11, 6))

for k in K_VALUES:
    df = pd.read_csv(log_paths[k])
    df.columns = df.columns.str.strip()
    # Sample every 100 episodes to reduce noise density for plotting
    ds = df[df["Episode"] % 100 == 0].copy()
    ds["smooth"] = ds["Total Reward"].rolling(SMOOTH, min_periods=1).mean()

    ax.plot(ds["Episode"], ds["Total Reward"],
            color=COLORS[k], alpha=0.18, linewidth=0.8)
    ax.plot(ds["Episode"], ds["smooth"],
            color=COLORS[k], linewidth=2.2, label=LABELS[k])

ax.set_xlabel("Episode", fontsize=13, labelpad=8)
ax.set_ylabel("Avg Episode Reward", fontsize=13, labelpad=8)
ax.set_title("TCCA k-Ablation Study — MPE simple_spread_v3", fontsize=15,
             fontweight="bold", pad=14)
ax.legend(fontsize=12, framealpha=0.35, facecolor="#21262d", edgecolor="#30363d")
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

note = f"Smoothing window: {SMOOTH} × 100-episode sampled points"
ax.text(0.99, 0.02, note, transform=ax.transAxes,
        fontsize=8, color="#8b949e", ha="right", va="bottom")

plt.tight_layout()
out = "ablation_k_mpe.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close()
