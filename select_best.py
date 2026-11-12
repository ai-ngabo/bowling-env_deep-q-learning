"""Pick the best experiment and copy its model for play.py.

    uv run select_best.py --member Principie   # rank one person's experiments
    uv run select_best.py --all                # compare everyone, pick the overall best

Each saved model plays a few games with no random moves, exactly like play.py,
and the experiments are ranked by that score. We test the model itself because
the training log can look good while the saved model stalls or throws into the
gutter. Ties (for example several models scoring 30) are broken by the training
average once epsilon reached its lowest value.
"""

import argparse
import csv
import os
import shutil

import torch

from train import QNetwork, make_env

MEMBERS = ["Alain", "Francis", "Principie"]
MAX_STEPS = 5_000  # a normal game is about 2,100 to 2,600 steps; a model that never throws is stopped here


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


def play_average(model_path, games):
    """Play the saved model with no random moves (like play.py) and return its average score."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    model = QNetwork(env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_reward = 0.0
    for game in range(1, games + 1):
        state, _ = env.reset(seed=1000 + game)  # same games as greedy_check.py
        steps, done = 0, False
        while not done and steps < MAX_STEPS:
            with torch.no_grad():
                q_values = model(torch.as_tensor(state, device=device).unsqueeze(0))
            state, reward, terminated, truncated, _ = env.step(int(q_values.argmax()))
            total_reward += reward
            steps += 1
            done = terminated or truncated
    env.close()
    return total_reward / games


def best_experiment(member, games):
    """Print the member's experiments ranked, and return (name, score) of the best."""
    scores = {}
    for name in sorted(os.listdir(member)):
        log = os.path.join(member, name, "episodes.csv")
        model = os.path.join(member, name, "dqn_model.pth")
        if name.startswith("exp") and os.path.isfile(log) and os.path.isfile(model):
            scores[name] = (play_average(model, games), greedy_average(log))  # play score first, training average breaks ties

    if not scores:
        return None

    for name in sorted(scores, key=scores.get, reverse=True):
        play, training = scores[name]
        print(f"  {name}: plays {play:.1f} | training average {training:.1f}")

    best = max(scores, key=scores.get)
    return best, scores[best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", help="one person's folder, e.g. Principie")
    parser.add_argument("--all", action="store_true", help="compare all members")
    parser.add_argument("--games", type=int, default=3, help="games each model plays (default 3)")
    args = parser.parse_args()

    if args.member:
        result = best_experiment(args.member, args.games)
        if not result:
            print(f"No finished experiments in {args.member}/ yet")
            return
        name, (play, training) = result
        best_folder = os.path.join(args.member, "best_model")
        os.makedirs(best_folder, exist_ok=True)
        shutil.copy(os.path.join(args.member, name, "dqn_model.pth"),
                    os.path.join(best_folder, "dqn_model.pth"))
        print(f"\nBest for {args.member}: {name} (plays {play:.1f}, training average {training:.1f})")
        print(f"Model copied to {best_folder}/dqn_model.pth")
        return

    if args.all:
        winners = {}
        for member in MEMBERS:
            if os.path.isdir(member):
                print(f"{member}:")
                result = best_experiment(member, args.games)
                if result:
                    winners[member] = result
        if not winners:
            print("No experiments found yet")
            return
        champion = max(winners, key=lambda m: winners[m][1])
        name, (play, training) = winners[champion]
        shutil.copy(os.path.join(champion, name, "dqn_model.pth"), "dqn_model.pth")
        print(f"\nGroup best: {champion} {name} (plays {play:.1f}, training average {training:.1f})")
        print("Copied to dqn_model.pth for play.py")
        return

    print("Use --member <name> or --all")


if __name__ == "__main__":
    main()