import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path="tcca_logs.csv", save_path="tcca_training_results.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure training has started and logged data.")
        return
        
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("Error: CSV is empty.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Reward and Win Rate
    # We plot Win Rate on primary axis and Average Reward on secondary axis, or both on same with proper scaling
    ax1.plot(df["Episode"], df["Win Rate"], color="blue", label="Win Rate")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Win Rate", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_ylim(-0.05, 1.05)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["Episode"], df["Average Reward"], color="green", alpha=0.6, label="Average Reward")
    ax1_twin.set_ylabel("Average Reward", color="green")
    ax1_twin.tick_params(axis='y', labelcolor="green")
    
    ax1.set_title("Performance vs Episodes")
    
    # Panel 2: Critic MSE
    ax2.plot(df["Episode"], df["Critic MSE"], color="red", label="Critic MSE")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title("Critic Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")
    
if __name__ == "__main__":
    plot_results()
