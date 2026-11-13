"""Watch a trained DQN agent play Bowling (Task 4).

    uv run play.py                                       # play the best model
    uv run play.py --model Principie/exp06/dqn_model.pth  # play a specific one
    uv run play.py --games 3                              # play several games
"""

import argparse

import numpy as np
import torch

from train import make_env, QNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained agent play Bowling")
    parser.add_argument("--model", default="Champion_model/dqn_model.pth",
                        help="Path to the saved model (defaults to the best one)")
    parser.add_argument("--games", type=int, default=1, help="How many games to play")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize the same Atari environment used during training.
    #    render_mode="human" opens a window so we can watch the game live.
    env = make_env(render_mode="human")
    num_actions = env.action_space.n

    # 2. Load the saved model.  3. Rebuild the same network.  4. Put the weights in it.
    saved_weights = torch.load(args.model, map_location="cpu")
    q_network = QNetwork(num_actions)
    q_network.load_state_dict(saved_weights)

    # 5. Run the agent in evaluation mode (we only play, no learning).
    q_network.eval()

    for game in range(1, args.games + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # turn the 4 stacked frames into a batch-of-one tensor for the network
            state_tensor = torch.as_tensor(np.asarray(state)).unsqueeze(0)

            # 6. Select the action with the highest Q-value (greedy, no randomness).
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = q_values.argmax().item()

            # 7. Step the game; the window updates in real time.
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Game {game}: score {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    main()