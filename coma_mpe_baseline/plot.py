import re
import matplotlib.pyplot as plt
import argparse
import sys

def parse_and_plot(log_file_path):
    episodes = []
    returns = []
    critic_losses = []
    
    # This explicit regex accurately extracts details from your untouched train.py stdout.
    # Example format matching: "Episode 45000    Avg Shared Return (last 100):   -26.42          Critic Loss (Bellman): 5.2332"
    pattern = re.compile(
        r"Episode\s+(\d+)\s+Avg Shared Return \(last 100\):\s+([-\d\.]+)\s+Critic Loss \(Bellman\):\s+([\d\.]+)"
    )
    
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    episodes.append(int(match.group(1)))
                    returns.append(float(match.group(2)))
                    critic_losses.append(float(match.group(3)))
    except FileNotFoundError:
        print(f"Error: Could not find log file at '{log_file_path}'.")
        print("Tip: Next time you run training, you can explicitly save logs by routing the output to a text file.")
        print("Command:   python train.py > train_log.txt")
        sys.exit(1)
        
    if not episodes:
        print(f"Error: Found no strictly formatted training metrics in '{log_file_path}'.")
        print("Warning: Did you ensure standard output routing was properly formulated?")
        sys.exit(1)
        
    print(f"Extracted {len(episodes)} data points. Generating graph...")
        
    # Standard structured plot configuration visualizing both key metrics
    plt.figure(figsize=(14, 5))
    
    # 1. Return Curve
    plt.subplot(1, 2, 1)
    plt.plot(episodes, returns, label="Avg Shared Return (Smoothed 100 eps)", color="#2563eb", linewidth=1.8)
    plt.title("Reward Performance over Time", fontsize=13)
    plt.xlabel("Training Episode", fontsize=11)
    plt.ylabel("Reward", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    # 2. Critic Convergence Curve
    plt.subplot(1, 2, 2)
    plt.plot(episodes, critic_losses, label="Critic MSE (TD-0 Bellman)", color="#dc2626", linewidth=1.5, alpha=0.8)
    plt.title("Critic Convergence Profile", fontsize=13)
    plt.xlabel("Training Episode", fontsize=11)
    plt.ylabel("MSE Loss", fontsize=11)
    plt.yscale("log") # Loss is generally better visualized on logarithmic scales for stability
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot COMA learning curves reliably without altering training configurations.")
    parser.add_argument("--file", type=str, default="train_log.txt", help="Path to the training terminal log file.")
    args = parser.parse_args()
    
    parse_and_plot(args.file)
