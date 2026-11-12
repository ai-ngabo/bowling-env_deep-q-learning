"""Pick the best experiment and copy its model for play.py.

    uv run select_best.py --member Principie   # rank one person's experiments
    uv run select_best.py --all                # compare everyone, pick the overall best

An experiment is scored by its average reward in the greedy phase: the episodes
after epsilon has dropped to its lowest value, when the agent plays with what it
learned instead of random moves. That is fairer than looking at one lucky game.
"""

import argparse
import csv
import os
import shutil

MEMBERS = ["Alain", "Francis", "Principie"]


def greedy_average(csv_path):
    """Average reward over the episodes where epsilon is at its floor."""
    rewards, epsilons = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            epsilons.append(float(row["epsilon"]))

    lowest_eps = min(epsilons)
    greedy = [r for r, e in zip(rewards, epsilons) if e == lowest_eps]
    return sum(greedy) / len(greedy)


def best_experiment(member):
    """Print the member's experiments ranked, and return (name, score) of the best."""
    scores = {}
    for name in sorted(os.listdir(member)):
        log = os.path.join(member, name, "episodes.csv")
        model = os.path.join(member, name, "dqn_model.pth")
        if name.startswith("exp") and os.path.isfile(log) and os.path.isfile(model):
            scores[name] = greedy_average(log)

    if not scores:
        return None

    for name in sorted(scores, key=scores.get, reverse=True):
        print(f"  {name}: average reward {scores[name]:.1f}")

    best = max(scores, key=scores.get)
    return best, scores[best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", help="one person's folder, e.g. Principie")
    parser.add_argument("--all", action="store_true", help="compare all members")
    args = parser.parse_args()

    if args.member:
        result = best_experiment(args.member)
        if not result:
            print(f"No finished experiments in {args.member}/ yet")
            return
        name, score = result
        best_folder = os.path.join(args.member, "best_model")
        os.makedirs(best_folder, exist_ok=True)
        shutil.copy(os.path.join(args.member, name, "dqn_model.pth"),
                    os.path.join(best_folder, "dqn_model.pth"))
        print(f"\nBest for {args.member}: {name} (average reward {score:.1f})")
        print(f"Model copied to {best_folder}/dqn_model.pth")
        return

    if args.all:
        winners = {}
        for member in MEMBERS:
            if os.path.isdir(member):
                print(f"{member}:")
                result = best_experiment(member)
                if result:
                    winners[member] = result
        if not winners:
            print("No experiments found yet")
            return
        champion = max(winners, key=lambda m: winners[m][1])
        name, score = winners[champion]
        shutil.copy(os.path.join(champion, name, "dqn_model.pth"), "dqn_model.pth")
        print(f"\nGroup best: {champion} {name} (average reward {score:.1f})")
        print("Copied to dqn_model.pth for play.py")
        return

    print("Use --member <name> or --all")


if __name__ == "__main__":
    main()
