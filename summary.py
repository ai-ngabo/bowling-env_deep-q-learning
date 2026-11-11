"""Show how one experiment did, right after running it.

    uv run summary.py --member Principie --exp 1

Prints the average reward for that experiment. The "greedy phase" average ignores
the early episodes where the agent was still mostly exploring at random, so it
shows how the trained agent actually played. The random baseline to beat is 24.
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", required=True)
    parser.add_argument("--exp", type=int, required=True)
    args = parser.parse_args()

    log = os.path.join(args.member, f"exp{args.exp:02d}", "episodes.csv")
    if not os.path.isfile(log):
        print(f"No log at {log}. Run train.py for this experiment first.")
        return

    rewards, epsilons = [], []
    with open(log, newline="") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            epsilons.append(float(row["epsilon"]))

    lowest_eps = min(epsilons)
    greedy = [r for r, e in zip(rewards, epsilons) if e == lowest_eps]

    print(f"Experiment {args.exp:02d} ({args.member})")
    print(f"  episodes played:        {len(rewards)}")
    print(f"  average reward (all):   {sum(rewards) / len(rewards):.1f}")
    print(f"  average reward (greedy): {sum(greedy) / len(greedy):.1f}")
    print(f"  best single game:       {max(rewards):.0f}")
    print(f"  random baseline to beat: 24")


if __name__ == "__main__":
    main()
