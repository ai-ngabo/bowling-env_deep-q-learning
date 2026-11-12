"""Plot DQN training results from episodes.csv.

Usage:
    uv run plot_results.py --member Alain            # every experiment compared with the baseline (Exp 1)
    uv run plot_results.py --member Alain --exp 4    # rewards, episode length and steps of one experiment
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt

RANDOM_PLAY = 23  # average score of random play in Bowling


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DQN training results")
    parser.add_argument("--member", required=True, help="Member folder, e.g. Alain")
    parser.add_argument("--exp", type=int, help="Experiment number (leave out to compare all experiments)")
    return parser.parse_args()


def load_log(member, exp):
    """Read one experiment's episodes.csv into lists."""
    csv_path = os.path.join(member, f"exp{exp:02d}", "episodes.csv")
    episodes, lengths, steps, rewards = [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            lengths.append(int(row["episode_length"]))
            steps.append(int(row["total_steps"]))
            rewards.append(float(row["reward"]))
    return episodes, lengths, steps, rewards


def moving_average(values, window=10):
    """Average of the last 10 games at each point, so the trend is easier to see."""
    averages = []
    for i in range(len(values)):
        recent = values[max(0, i - window + 1): i + 1]
        averages.append(sum(recent) / len(recent))
    return averages


def plot_experiment(member, exp):
    """Three charts for one experiment: rewards, episode length, and rewards against steps."""
    episodes, lengths, steps, rewards = load_log(member, exp)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(episodes, rewards, alpha=0.3, label="Each game")
    axes[0].plot(episodes, moving_average(rewards), label="10-game average")
    axes[0].axhline(RANDOM_PLAY, color="gray", linestyle="--", label="Random play")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(episodes, lengths)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode length (steps)")

    axes[2].plot(steps, moving_average(rewards))
    axes[2].axhline(RANDOM_PLAY, color="gray", linestyle="--")
    axes[2].set_xlabel("Total steps")
    axes[2].set_ylabel("Reward (10-game average)")

    fig.suptitle(f"{member} exp{exp:02d}: training curves")
    fig.tight_layout()
    out_path = os.path.join(member, f"exp{exp:02d}", "training_plot.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def plot_comparison(member):
    """One small chart per experiment (Exp 1 to 10), each with the baseline (Exp 1) in gray."""
    _, _, _, baseline = load_log(member, 1)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)

    for exp, ax in zip(range(1, 11), axes.flat):
        if not os.path.isfile(os.path.join(member, f"exp{exp:02d}", "episodes.csv")):
            ax.set_title(f"Exp {exp}: not run yet")
            continue
        _, _, _, rewards = load_log(member, exp)
        if exp != 1:
            ax.plot(moving_average(baseline), color="gray", label="Baseline (Exp 1)")
        ax.plot(moving_average(rewards), color="tab:blue", label=f"Exp {exp}")
        ax.axhline(RANDOM_PLAY, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Exp {exp}")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel("Reward (10-game average)")
    axes[1, 0].set_ylabel("Reward (10-game average)")
    fig.suptitle(f"{member}: each experiment compared with the baseline")
    fig.tight_layout()
    out_path = os.path.join(member, "reward_curves.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    if args.exp is None:
        plot_comparison(args.member)
    else:
        plot_experiment(args.member, args.exp)


if __name__ == "__main__":
    main()