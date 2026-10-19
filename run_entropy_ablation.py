"""
Task 5: Entropy Ablation — TCCA-SAC MPE
Trains TCCA-SAC with alpha=0 (entropy disabled).
Compares against the existing alpha=0.005 run from tcca_sac_baseline/training_logs.csv.
Saves new log to tcca_sac_baseline/tcca_sac_alpha0_log.csv
Plots both reward curves as ablation_entropy_mpe.png
"""

import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tcca_sac_baseline"))
from env_wrapper import SimpleSpreadWrapper
from tcca_sac_agent import TCCASACAgent
from replay_buffer import ReplayBuffer

NUM_EPISODES   = 50000
BATCH_SIZE     = 64
K_VAL          = 3
OUTDIR         = "tcca_sac_baseline"
ALPHA0_CSV     = os.path.join(OUTDIR, "tcca_sac_alpha0_log.csv")
ALPHA_ORIG_CSV = os.path.join(OUTDIR, "training_logs.csv")


def run_alpha0():
    print(f"\n{'='*55}\n  Training TCCA-SAC alpha=0  ->  {ALPHA0_CSV}\n{'='*55}")
    env   = SimpleSpreadWrapper(N=3, max_cycles=25)
    agent = TCCASACAgent(env.n_agents, env.obs_dim, env.state_dim,
                         env.n_actions, k=K_VAL, alpha=0.0)
    buf   = ReplayBuffer(capacity=50000)
    pd.DataFrame(columns=["Episode","Reward","Critic1 Loss","Critic2 Loss"]
                 ).to_csv(ALPHA0_CSV, index=False)
    ep_returns = []
    for ep in range(1, NUM_EPISODES + 1):
        obs, state = env.reset()
        done = False; ep_return = 0.0; trajectory = []
        while not done:
            obs_list = [obs[env.agents[i]] for i in range(env.n_agents)]
            actions  = agent.get_actions(obs_list)
            action_dict = {env.agents[i]: actions[i] for i in range(env.n_agents)}
            next_obs, next_state, reward, done = env.step(action_dict)
            next_obs_list = [next_obs[env.agents[i]] for i in range(env.n_agents)]
            trajectory.append((obs_list, state, actions, reward,
                                next_obs_list, next_state, done))
            obs, state = next_obs, next_state
            ep_return += reward
        buf.push(trajectory); ep_returns.append(ep_return)
        c1 = c2 = 0.0
        if len(buf) > 10:
            c1, c2 = agent.update(buf.sample(BATCH_SIZE, K_VAL))
        pd.DataFrame([{"Episode": ep, "Reward": ep_return,
                        "Critic1 Loss": c1, "Critic2 Loss": c2}]
                     ).to_csv(ALPHA0_CSV, mode='a', header=False, index=False)
        if ep % 1000 == 0:
            print(f"  a=0 | Ep {ep:05d} | Avg(100)={np.mean(ep_returns[-100:]):8.2f}")
    env.close()
    print(f"  Done. -> {ALPHA0_CSV}")


if os.path.exists(ALPHA0_CSV):
    print(f"Existing {ALPHA0_CSV} found. Skipping alpha=0 training.")
else:
    run_alpha0()

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "xtick.color":"#8b949e","ytick.color":"#8b949e","text.color":"#e6edf3",
    "grid.color":"#21262d","grid.linestyle":"--","grid.linewidth":0.6,
    "font.family":"DejaVu Sans",
})
SMOOTH = 5

def load_sampled(path, col, every=100):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    ds = df[df["Episode"] % every == 0].copy()
    ds["smooth"] = ds[col].rolling(SMOOTH, min_periods=1).mean()
    return ds

ds_orig = load_sampled(ALPHA_ORIG_CSV, "Reward")
ds_a0   = load_sampled(ALPHA0_CSV, "Reward")

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(ds_orig["Episode"], ds_orig["Reward"], color="#3fb950", alpha=0.18, linewidth=0.8)
ax.plot(ds_a0["Episode"],   ds_a0["Reward"],   color="#f78166", alpha=0.18, linewidth=0.8)
ax.plot(ds_orig["Episode"], ds_orig["smooth"],  color="#3fb950", linewidth=2.2,
        label=r"TCCA-SAC  $\alpha$=0.005 (with entropy)")
ax.plot(ds_a0["Episode"],   ds_a0["smooth"],    color="#f78166", linewidth=2.2,
        label=r"TCCA-SAC  $\alpha$=0  (no entropy)")
ax.set_xlabel("Episode", fontsize=13, labelpad=8)
ax.set_ylabel("Avg Episode Reward", fontsize=13, labelpad=8)
ax.set_title(r"Entropy Ablation — TCCA-SAC (MPE simple_spread_v3)", fontsize=15,
             fontweight="bold", pad=14)
ax.legend(fontsize=12, framealpha=0.35, facecolor="#21262d", edgecolor="#30363d")
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.text(0.99, 0.02, f"Smoothing: {SMOOTH} × 100-ep points",
        transform=ax.transAxes, fontsize=8, color="#8b949e", ha="right", va="bottom")
plt.tight_layout()
out = "ablation_entropy_mpe.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()
