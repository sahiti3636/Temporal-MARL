"""
Task 2: COMA vs TCCA-SAC Comparison Plot (SMAC)
Reads coma_smac_baseline/training_logs.csv and tcca_smac_research/tcca_logs.csv
Produces two plots: win-rate comparison and reward comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Styling ────────────────────────────────────────────────────────────────────
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

SMOOTH = 5

# ── Load COMA SMAC ─────────────────────────────────────────────────────────────
coma = pd.read_csv("coma_smac_baseline/training_logs.csv")
coma.columns = coma.columns.str.strip()
# Columns: Episode, Win Rate, Total Reward, Average Critic Loss
coma["wr_smooth"]  = coma["Win Rate"].rolling(SMOOTH, min_periods=1).mean()
coma["rew_smooth"] = coma["Total Reward"].rolling(SMOOTH, min_periods=1).mean()

# ── Load TCCA SMAC ─────────────────────────────────────────────────────────────
tcca = pd.read_csv("tcca_smac_research/tcca_logs.csv")
tcca.columns = tcca.columns.str.strip()
# Strip any \r leftovers in column names
tcca.columns = [c.rstrip('\r') for c in tcca.columns]
# Columns: Episode, Average Reward, Win Rate, Critic MSE, Actor Loss
tcca["wr_smooth"]  = tcca["Win Rate"].rolling(SMOOTH, min_periods=1).mean()
tcca["rew_smooth"] = tcca["Average Reward"].rolling(SMOOTH, min_periods=1).mean()

print(f"COMA SMAC:  {len(coma)} log points  | max ep = {coma['Episode'].max()}")
print(f"TCCA SMAC:  {len(tcca)} log points  | max ep = {tcca['Episode'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Win Rate
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(coma["Episode"], coma["Win Rate"],
        color="#58a6ff", alpha=0.18, linewidth=0.8)
ax.plot(tcca["Episode"], tcca["Win Rate"],
        color="#f78166", alpha=0.18, linewidth=0.8)

ax.plot(coma["Episode"], coma["wr_smooth"],
        color="#58a6ff", linewidth=2.2, label="COMA (SMAC 3m)")
ax.plot(tcca["Episode"], tcca["wr_smooth"],
        color="#f78166", linewidth=2.2, label="TCCA-SAC (SMAC 3m)")

ax.set_xlabel("Episode", fontsize=13, labelpad=8)
ax.set_ylabel("Win Rate", fontsize=13, labelpad=8)
ax.set_ylim(-0.05, 1.05)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax.set_title("COMA vs TCCA-SAC — Win Rate (SMAC 3m)", fontsize=15, fontweight="bold", pad=14)
ax.legend(fontsize=12, framealpha=0.35, facecolor="#21262d", edgecolor="#30363d")
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

note = f"Smoothing window: {SMOOTH} log points"
ax.text(0.99, 0.02, note, transform=ax.transAxes,
        fontsize=8, color="#8b949e", ha="right", va="bottom")

plt.tight_layout()
out1 = "comparison_coma_vs_tcca_smac_winrate.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Reward
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(coma["Episode"], coma["Total Reward"],
        color="#58a6ff", alpha=0.18, linewidth=0.8)
ax.plot(tcca["Episode"], tcca["Average Reward"],
        color="#f78166", alpha=0.18, linewidth=0.8)

ax.plot(coma["Episode"], coma["rew_smooth"],
        color="#58a6ff", linewidth=2.2, label="COMA (SMAC 3m)")
ax.plot(tcca["Episode"], tcca["rew_smooth"],
        color="#f78166", linewidth=2.2, label="TCCA-SAC (SMAC 3m)")

ax.set_xlabel("Episode", fontsize=13, labelpad=8)
ax.set_ylabel("Avg Episode Reward", fontsize=13, labelpad=8)
ax.set_title("COMA vs TCCA-SAC — Episode Reward (SMAC 3m)", fontsize=15, fontweight="bold", pad=14)
ax.legend(fontsize=12, framealpha=0.35, facecolor="#21262d", edgecolor="#30363d")
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

ax.text(0.99, 0.02, note, transform=ax.transAxes,
        fontsize=8, color="#8b949e", ha="right", va="bottom")

plt.tight_layout()
out2 = "comparison_coma_vs_tcca_smac_reward.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close()
