"""Plot training curves from an experiment's episodes.csv.

Usage:
    uv run plot_results.py --member Francis --exp 1
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DQN training results")
    parser.add_argument("--member", required=True, help="Member folder, e.g. Francis")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.join(args.member, f"exp{args.exp:02d}", "episodes.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No log found at {csv_path}. Run train.py first.")

    episodes, lengths, steps, rewards, epsilons = [], [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            lengths.append(int(row["episode_length"]))
            steps.append(int(row["total_steps"]))
            rewards.append(float(row["reward"]))
            epsilons.append(float(row["epsilon"]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    axes[0].plot(episodes, rewards, color="steelblue")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title(f"{args.member} exp{args.exp:02d} — training curves")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, lengths, color="darkorange")
    axes[1].set_ylabel("Episode length (steps)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, epsilons, color="seagreen")
    axes[2].set_xlabel("Total agent steps")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(args.member, f"exp{args.exp:02d}", "training_plot.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
