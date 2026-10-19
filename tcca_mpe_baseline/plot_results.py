import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(csv_file="training_logs.csv", output_file="tcca_results_graph.png"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Could not find {csv_file}. Make sure you train the agent natively first.")
        return
        
    episodes = df["Episode"]
    rewards = df["Total Reward"]
    losses = df["Average Critic Loss"]
    
    smoothed_rewards = rewards.rolling(window=100, min_periods=1).mean()
    
    plt.figure(figsize=(14, 5))
    
    # Render Left Subplot (Reward Curve Validation)
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label="Raw Total Reward", color='#3b82f6', alpha=0.15, linewidth=1)
    plt.plot(episodes, smoothed_rewards, label="100-Episode Rolling Avg", color='#1e3a8a', linewidth=2)
    plt.title("Reward Performance over Time", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Shared Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    
    # Render Right Subplot (Bellman MSE Optimization)
    plt.subplot(1, 2, 2)
    valid_mask = losses > 0
    valid_eps = episodes[valid_mask]
    valid_losses = losses[valid_mask]
    
    plt.plot(valid_eps, valid_losses, label="Critic MSE (TCCA targets)", color='#ef4444', alpha=0.85, linewidth=1.5)
    plt.title("Critic Convergence Profile", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("MSE Loss Format", fontsize=12)
    plt.yscale("log") # Explicit validation tracking via metric constraints
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph visualizations saved physically as local artifact: '{output_file}'")

if __name__ == "__main__":
    plot_graph()
