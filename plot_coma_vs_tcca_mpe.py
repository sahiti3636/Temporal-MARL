"""
Task 1: Direct COMA vs TCCA Comparison Plot (MPE)
Reads coma_mpe_baseline/train_log.txt and tcca_mpe_baseline/training_logs.csv,
aligns by episode, applies 100-ep rolling average, saves comparison_coma_vs_tcca_mpe.png
"""

import re
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

SMOOTH = 5   # rolling window over logged points (each point = 100 eps in COMA log)

# ── Load COMA MPE (train_log.txt) ──────────────────────────────────────────────
coma_eps, coma_rew = [], []
pattern = re.compile(r"Episode\s+(\d+)\s+Avg Shared Return \(last 100\):\s+([-\d.]+)")
with open("coma_mpe_baseline/train_log.txt") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            coma_eps.append(int(m.group(1)))
            coma_rew.append(float(m.group(2)))

coma_df = pd.DataFrame({"episode": coma_eps, "reward": coma_rew})
coma_df["smooth"] = coma_df["reward"].rolling(SMOOTH, min_periods=1).mean()

# ── Load TCCA MPE (training_logs.csv) ─────────────────────────────────────────
tcca_df = pd.read_csv("tcca_mpe_baseline/training_logs.csv")
tcca_df.columns = tcca_df.columns.str.strip()
# Downsample to every 100 episodes to match COMA's log density
tcca_sampled = tcca_df[tcca_df["Episode"] % 100 == 0].copy()
tcca_sampled["smooth"] = tcca_sampled["Total Reward"].rolling(SMOOTH, min_periods=1).mean()

print(f"COMA MPE: {len(coma_df)} logged points  |  max ep = {coma_df['episode'].max()}")
print(f"TCCA MPE: {len(tcca_sampled)} sampled points  |  max ep = {tcca_sampled['Episode'].max()}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

# Raw faint traces
ax.plot(coma_df["episode"],    coma_df["reward"],
        color="#58a6ff", alpha=0.18, linewidth=0.8)
ax.plot(tcca_sampled["Episode"], tcca_sampled["Total Reward"],
        color="#3fb950", alpha=0.18, linewidth=0.8)

# Smoothed bold lines
ax.plot(coma_df["episode"],    coma_df["smooth"],
        color="#58a6ff", linewidth=2.2, label="COMA (MPE)")
ax.plot(tcca_sampled["Episode"], tcca_sampled["smooth"],
        color="#3fb950", linewidth=2.2, label=f"TCCA k=5 (MPE)")

ax.set_xlabel("Episode", fontsize=13, labelpad=8)
ax.set_ylabel("Avg Episode Reward", fontsize=13, labelpad=8)
ax.set_title("COMA vs TCCA — MPE simple_spread_v3", fontsize=15, fontweight="bold", pad=14)
ax.legend(fontsize=12, framealpha=0.35, facecolor="#21262d", edgecolor="#30363d")
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

note = f"Smoothing window: {SMOOTH} × 100-episode log points"
ax.text(0.99, 0.02, note, transform=ax.transAxes,
        fontsize=8, color="#8b949e", ha="right", va="bottom")

plt.tight_layout()
out = "comparison_coma_vs_tcca_mpe.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
