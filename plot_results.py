"""Plot DQN training results from episodes.csv files.

Usage:
    uv run plot_results.py --member Francis            # all experiments vs the baseline (Exp 1)
    uv run plot_results.py --member Francis --exp 4    # training curves of one experiment
"""

import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RANDOM_PLAY = 23  # average score of random play in Bowling
BLUE, GRAY, GRID, INK = "#2a78d6", "#86857f", "#e4e3df", "#52514e"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DQN training results")
    parser.add_argument("--member", required=True, help="Member folder, e.g. Francis")
    parser.add_argument("--exp", type=int, help="Experiment number (leave out to compare all experiments)")
    return parser.parse_args()


def load_log(member, exp):
    """Read one experiment's episodes.csv into lists, or return None if it doesn't exist."""
    csv_path = os.path.join(member, f"exp{exp:02d}", "episodes.csv")
    if not os.path.isfile(csv_path):
        return None

    log = {"episodes": [], "lengths": [], "steps": [], "rewards": [], "epsilons": []}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            log["episodes"].append(int(row["episode"]))
            log["lengths"].append(int(row["episode_length"]))
            log["steps"].append(int(row["total_steps"]))
            log["rewards"].append(float(row["reward"]))
            log["epsilons"].append(float(row["epsilon"]))
    return log


def rolling_average(values, window=10):
    """Average of the last `window` games at each point (smooths the noisy rewards)."""
    averages = []
    for i in range(len(values)):
        recent = values[max(0, i - window + 1): i + 1]
        averages.append(sum(recent) / len(recent))
    return averages


def style(ax):
    """Light grid, no top/right border."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def plot_comparison(member):
    """One small chart per experiment, each drawn against the baseline (Exp 1)."""
    experiments = sorted(int(name[3:]) for name in os.listdir(member)
                         if name.startswith("exp") and name[3:].isdigit()
                         and os.path.isfile(os.path.join(member, name, "episodes.csv")))
    if not experiments:
        raise FileNotFoundError(f"No episodes.csv found under {member}/. Run train.py first.")
    baseline = load_log(member, 1)

    rows = math.ceil(len(experiments) / 5)
    fig, axes = plt.subplots(rows, 5, figsize=(16, 3.2 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, exp in zip(axes.flat, experiments):
        rewards = load_log(member, exp)["rewards"]
        last_20 = sum(rewards[-20:]) / len(rewards[-20:])
        ax.axhline(RANDOM_PLAY, color=GRAY, linestyle="--", linewidth=1)
        if baseline and exp != 1:
            ax.plot(rolling_average(baseline["rewards"]), color=GRAY, linewidth=2)
        ax.plot(rolling_average(rewards), color=BLUE, linewidth=2)
        ax.set_title(f"Exp {exp}  (last 20 games: {last_20:.1f})", fontsize=10, loc="left", color=INK)
        style(ax)
    for ax in list(axes.flat)[len(experiments):]:
        ax.axis("off")

    legend = [Line2D([], [], color=BLUE, linewidth=2, label="This experiment"),
              Line2D([], [], color=GRAY, linewidth=2, label="Baseline (Exp 1)"),
              Line2D([], [], color=GRAY, linewidth=1, linestyle="--", label="Random play (about 23)")]
    fig.legend(handles=legend, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{member}: reward per game, 10-game average", x=0.01, ha="left", fontsize=12, color=INK)
    fig.supxlabel("Game", fontsize=9, color=INK)
    fig.supylabel("Reward", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(member, "reward_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def plot_experiment(member, exp):
    """Rewards, episode length and number of steps for one experiment."""
    log = load_log(member, exp)
    if log is None:
        raise FileNotFoundError(f"No log found for {member} exp{exp:02d}. Run train.py first.")
    smooth = rolling_average(log["rewards"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].scatter(log["episodes"], log["rewards"], color=BLUE, alpha=0.3, s=12)
    axes[0].plot(log["episodes"], smooth, color=BLUE, linewidth=2)
    axes[0].axhline(RANDOM_PLAY, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_title("Reward per game (dots) and 10-game average", fontsize=10, loc="left", color=INK)
    axes[0].set_xlabel("Game", fontsize=9, color=INK)

    axes[1].plot(log["episodes"], log["lengths"], color=BLUE, linewidth=1.5)
    axes[1].set_title("Episode length (steps per game)", fontsize=10, loc="left", color=INK)
    axes[1].set_xlabel("Game", fontsize=9, color=INK)

    # Mark when epsilon stops falling: from there on the agent mostly uses what it learned
    min_epsilon_step = log["steps"][log["epsilons"].index(min(log["epsilons"]))]
    axes[2].plot(log["steps"], smooth, color=BLUE, linewidth=2)
    axes[2].axhline(RANDOM_PLAY, color=GRAY, linestyle="--", linewidth=1)
    axes[2].axvline(min_epsilon_step, color=GRAY, linestyle=":", linewidth=1)
    axes[2].annotate("epsilon at minimum", (min_epsilon_step, 0.97), xycoords=("data", "axes fraction"),
                     xytext=(-3, 0), textcoords="offset points", rotation=90, ha="right", va="top",
                     fontsize=8, color=INK)
    axes[2].xaxis.set_major_formatter(lambda value, _: f"{value / 1000:.0f}k")
    axes[2].set_title("10-game average reward vs number of steps", fontsize=10, loc="left", color=INK)
    axes[2].set_xlabel("Total agent steps", fontsize=9, color=INK)

    for ax in axes:
        style(ax)
    fig.suptitle(f"{member} exp{exp:02d}: training curves", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(member, f"exp{exp:02d}", "training_plot.png")
    fig.savefig(out_path, dpi=150)
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