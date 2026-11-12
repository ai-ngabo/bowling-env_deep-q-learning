"""Play a trained model greedily, like play.py but without the game window, and count the buttons it presses.

Usage:
    uv run greedy_check.py --member Alain --exp 4 --games 3
"""

import argparse
import os
from collections import Counter

import torch

from train import QNetwork, make_env

MAX_STEPS = 5_000  # a normal game is about 2,100 to 2,600 steps; more means the agent stopped throwing


def parse_args():
    parser = argparse.ArgumentParser(description="Check how a trained model plays without random moves")
    parser.add_argument("--member", required=True, help="Member folder, e.g. Alain")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number")
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    button_names = env.unwrapped.get_action_meanings()

    model = QNetwork(env.action_space.n).to(device)
    model_path = os.path.join(args.member, f"exp{args.exp:02d}", "dqn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    buttons = Counter()
    for game in range(1, args.games + 1):
        state, _ = env.reset(seed=1000 + game)
        game_reward, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            with torch.no_grad():
                q_values = model(torch.as_tensor(state, device=device).unsqueeze(0))
            action = int(q_values.argmax())  # greedy: always the button with the highest Q-value
            buttons[button_names[action]] += 1
            state, reward, terminated, truncated, _ = env.step(action)
            game_reward += reward
            steps += 1
            done = terminated or truncated
        note = "" if done else "  (stopped: never finished the game)"
        print(f"Game {game}: reward {game_reward:.0f} in {steps} steps{note}")

    total = sum(buttons.values())
    print("Buttons pressed:", ", ".join(f"{name} {100 * count / total:.0f}%" for name, count in buttons.most_common(3)))
    env.close()


if __name__ == "__main__":
    main()