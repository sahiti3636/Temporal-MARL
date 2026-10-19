import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(csv_file="training_logs.csv", output_file="tcca_sac_graph.png"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Log file '{csv_file}' inherently not found. Execute standard training procedures firstly.")
        return
        
    episodes = df["Episode"]
    rewards = df["Reward"]
    avg_critic_loss = (df["Critic1 Loss"] + df["Critic2 Loss"]) / 2.0
    
    smoothed_rewards = rewards.rolling(window=100, min_periods=1).mean()
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label="Raw Total Reward", color='#2563eb', alpha=0.15, linewidth=1)
    plt.plot(episodes, smoothed_rewards, label="100-Episode Rolling Avg", color='#1e3a8a', linewidth=2)
    plt.title("Reward Performance over Time (SAC)", fontsize=14)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Reward Architecture", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    valid_mask = avg_critic_loss > 0
    
    plt.plot(episodes[valid_mask], avg_critic_loss[valid_mask], label="Avg Twin Critic MSE", color='#dc2626', alpha=0.85, linewidth=1.5)
    plt.title("Critic Convergence Profile", fontsize=14)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Twin Target MSE Loss (Log)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph visualizations physically written internally via validation to: '{output_file}'")

if __name__ == "__main__":
    plot_graph()
