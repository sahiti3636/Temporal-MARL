import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(csv_file="training_logs.csv", output_file="smac_coma_results.png"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Target log '{csv_file}' undiscovered natively. Initialize execution tracking originally.")
        return
        
    episodes = df["Episode"]
    win_rates = df["Win Rate"]
    rewards = df["Total Reward"]
    losses = df["Average Critic Loss"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1st Component Structure: Twin Metrics mapping Win-Rate prominently
    color1 = '#2563eb' # Prominently styled colors dynamically
    ax1.set_xlabel('Episodes Trained')
    ax1.set_ylabel('Execution Win Rate (Per 100)', color=color1)
    ax1.plot(episodes, win_rates, color=color1, linewidth=2, label='Win Rate Percentage')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    ax1_twin = ax1.twinx()
    color2 = '#15803d'
    ax1_twin.set_ylabel('Base Objective Return', color=color2)
    ax1_twin.plot(episodes, rewards, color=color2, alpha=0.5, linewidth=1.5, label='Raw Return')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('SMAC Policy Integration Outcomes', fontsize=14)
    
    # 2nd Validation Structure: Explicit evaluation representation
    ax2.set_xlabel('Episodes Trained')
    ax2.set_ylabel('Sequential Target Error (Log Baseline)')
    
    valid_mask = losses > 0
    ax2.plot(episodes[valid_mask], losses[valid_mask], color='#dc2626', linewidth=1.5, label='Central Critic MSE')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    ax2.set_title('Central Target Convergence Profile', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Double-structured representation generated reliably -> {output_file}")

if __name__ == '__main__':
    plot_graph()
