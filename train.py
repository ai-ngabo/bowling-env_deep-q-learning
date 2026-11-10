"""Training a DQN agent to play Atari Bowling (ALE/Bowling-v5).

    1. experiment run example:
            uv run train.py --member "name" --exp 1

    2. hyperparamters change from the command line example:
            uv run train.py --member "name" --exp 2 --lr 1e-3
"""

import argparse
import csv
import os
import random
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)


# 1. Fixed settings agreed for each and every experiment

ENV_ID = "ALE/Bowling-v5"
TOTAL_STEPS = 300_000        # agent steps per experiment (1 step = 4 game frames)
BUFFER_SIZE = 50_000         # transitions kept in the replay buffer
LEARNING_STARTS = 10_000     # steps collected before the first update
TRAIN_EVERY = 4              # one gradient update every 4 agent steps
TARGET_UPDATE_EVERY = 1_000  # copy online network into target network every N steps
SEED = 42


def parse_args(argv=None):
    """Read the experiment's hyperparameters. Baseline Exp. / Default (Exp 1)."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Bowling")
    parser.add_argument("--member", required=True, help="Your folder name, e.g. Francis")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number, 1 to 10")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=100_000,
                        help="Steps for epsilon to fall linearly from start to end")
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    return parser.parse_args(argv)


# Setting environment, replay buffer, logging

def make_env(render_mode=None):
    """Create Bowling env with standard DQN preprocessing.

      -Observations are uint8 arrays of shape (4, 84, 84): the last 4 frames,
       grayscale and resized to 84x84.
    """
    # frameskip=1: frame skipping by AtariPreprocessing
    bowling_env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    bowling_env = gym.wrappers.AtariPreprocessing(bowling_env, frame_skip=4, screen_size=84, grayscale_obs=True)
    bowling_env = gym.wrappers.FrameStackObservation(bowling_env, stack_size=4)
    return bowling_env


class ReplayBuffer:
    """Stores past transitions and returns random mini-batches of them."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # drops oldest trans.. when it's full

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),           # (B, 4, 84, 84) uint8
            np.array(actions, dtype=np.int64),    # (B,)
            np.array(rewards, dtype=np.float32),   # (B,)
            np.array(next_states),         # (B, 4, 84, 84) uint8
            np.array(dones, dtype=np.float32),    # (B,) 1.0 if the episode ended
        )

    def __len__(self):
        return len(self.memory)


def log_episode(csv_path, episode, episode_length, total_steps, reward, epsilon):
    """ Writes one row per finished ep. & episode 1 starts a new file.."""
    mode = "w" if episode == 1 else "a"
    with open(csv_path, mode, newline="") as file:
        editor = csv.writer(file)
        if episode == 1:
            editor.writerow(["episode", "episode_length", "total_steps", "reward", "epsilon"])
        editor.writerow([episode, episode_length, total_steps, reward, round(epsilon, 4)])



# Q-network and epsilon-greedy action selection

class QNetwork(nn.Module):
    """CNN that maps 4 stacked 84x84 frames to one Q-value per action.

    Input: tensor (B, 4, 84, 84), uint8 or float, pixel values 0-255
    (scale to 0-1 inside forward). Output: tensor (B, num_actions).
    """

    def __init__(self, num_actions):
        super().__init__()
        raise NotImplementedError("TODO: QNetwork layers")

    def forward(self, x):
        raise NotImplementedError("TODO: QNetwork.forward")


def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Linear decay: eps_start at step 0, eps_end at step eps_decay, then stay at eps_end."""
    raise NotImplementedError("TODO: get_epsilon")


def select_action(q_network, state, epsilon, num_actions, device):
    """Epsilon-greedy. With probability epsilon return a random action (int),
    otherwise the action with the highest Q-value for `state` (4, 84, 84).
    """
    raise NotImplementedError("TODO: select_action")


# DQN update and training loop

def train_step(online_net, target_net, optimizer, buffer, batch_size, gamma, device):
    """One DQN update on a random mini-batch from `buffer`. Returns the loss (float).

    1. Q(s, a) from online_net for the actions taken
    2. Bellman target from target_net: r + gamma * max_a' Q_target(s', a') * (1 - done)
    3. Huber loss, then zero_grad, backward, optimizer.step
    Must not change target_net.
    """
    raise NotImplementedError("TODO: train_step")


def main():
    """Run one experiment and save the model to <member>/expNN/dqn_model.pth.

    Outline:
    1. args = parse_args(); set seeds; pick device (cuda if available)
    2. create the output folder <member>/expNN/
    3. env = make_env(); online and target QNetwork (target = copy of online);
       Adam optimizer on the online network; ReplayBuffer(BUFFER_SIZE)
    4. for each step up to args.total_steps:
         epsilon -> action -> env.step -> buffer.push (done = terminated only)
         after LEARNING_STARTS, every TRAIN_EVERY steps: train_step
         every TARGET_UPDATE_EVERY steps: copy online weights into target
         when the episode ends: log_episode, print a line, reset the env
    5. save online_net.state_dict() to dqn_model.pth
    """
    raise NotImplementedError("TODO: main training loop")


if __name__ == "__main__":
    main()