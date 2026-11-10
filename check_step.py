"""Step 2 checks for train.py. Run from the repo folder:

    uv run check_step.py

Each line shows PASS, NOT DONE, WAITING (needs a teammate's part first) or FAIL.
Step 2 is finished when every line says PASS.
"""

import os
import tempfile

import numpy as np
import torch

import train

NUM_ACTIONS = 6


def fake_state():
    """A random 4-frame observation, the same shape and type as make_env() returns."""
    return np.random.randint(0, 256, size=(4, 84, 84), dtype=np.uint8)



# Alain

def check_parse_args():
    args = train.parse_args(["--member", "Test", "--exp", "1"])
    baseline = {"lr": 1e-4, "gamma": 0.99, "batch_size": 32,
                "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 100_000}
    for name, value in baseline.items():
        assert getattr(args, name) == value, f"default {name} is {getattr(args, name)}, baseline is {value}"


def check_make_env():
    env = train.make_env()
    state, _ = env.reset(seed=0)
    assert state.shape == (4, 84, 84), f"state shape is {state.shape}, expected (4, 84, 84)"
    assert state.dtype == np.uint8, f"state type is {state.dtype}, expected uint8"
    assert env.action_space.n == NUM_ACTIONS, f"{env.action_space.n} actions, expected {NUM_ACTIONS}"
    next_state, reward, terminated, truncated, _ = env.step(1)  # 1 = FIRE
    assert next_state.shape == (4, 84, 84), "env.step() returned the wrong shape"
    env.close()


def check_replay_buffer():
    buffer = train.ReplayBuffer(capacity=100)
    for i in range(150):
        buffer.push(fake_state(), i % NUM_ACTIONS, 1.0, fake_state(), i % 2 == 0)
    assert len(buffer) == 100, f"len is {len(buffer)} after 150 pushes into capacity 100, expected 100"

    states, actions, rewards, next_states, dones = buffer.sample(32)
    assert states.shape == (32, 4, 84, 84) and states.dtype == np.uint8, "states: expected (32, 4, 84, 84) uint8"
    assert next_states.shape == (32, 4, 84, 84), "next_states: expected (32, 4, 84, 84)"
    assert actions.shape == (32,) and actions.dtype == np.int64, "actions: expected (32,) int64"
    assert rewards.shape == (32,) and rewards.dtype == np.float32, "rewards: expected (32,) float32"
    assert dones.shape == (32,) and dones.dtype == np.float32, "dones: expected (32,) float32"
    assert np.isin(dones, [0.0, 1.0]).all(), "dones should only be 0.0 or 1.0"


def check_log_episode():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "log.csv")
        train.log_episode(path, 1, 2100, 2100, 25.0, 0.98)
        train.log_episode(path, 2, 2050, 4150, 31.0, 0.96)
        with open(path) as f:
            lines = f.read().strip().splitlines()
    assert lines[0] == "episode,episode_length,total_steps,reward,epsilon", f"header is '{lines[0]}'"
    assert len(lines) == 3, f"expected a header and 2 rows, got {len(lines)} lines"


# Principie

def check_q_network():
    net = train.QNetwork(NUM_ACTIONS)
    batch = torch.randint(0, 256, (32, 4, 84, 84), dtype=torch.uint8)
    q_values = net(batch)
    assert q_values.shape == (32, NUM_ACTIONS), f"output shape is {tuple(q_values.shape)}, expected (32, 6)"
    assert torch.isfinite(q_values).all(), "output contains NaN or inf"
    num_params = sum(p.numel() for p in net.parameters())
    return f"{num_params:,} parameters"


def check_get_epsilon():
    expected = {0: 1.0, 500: 0.525, 1000: 0.05, 5000: 0.05}  # start 1.0, end 0.05, decay 1000
    for step, value in expected.items():
        epsilon = train.get_epsilon(step, 1.0, 0.05, 1000)
        assert abs(epsilon - value) < 1e-6, f"step {step} gives {epsilon}, expected {value}"


def check_select_action():
    net = train.QNetwork(NUM_ACTIONS)
    state = fake_state()

    action = train.select_action(net, state, 0.0, NUM_ACTIONS, "cpu")
    assert isinstance(action, int), f"returned {type(action).__name__}, expected int"

    random_actions = {train.select_action(net, state, 1.0, NUM_ACTIONS, "cpu") for _ in range(200)}
    assert len(random_actions) >= 4, f"epsilon 1.0 only picked {random_actions}, expected many actions"

    with torch.no_grad():
        best_action = int(net(torch.as_tensor(state).unsqueeze(0)).argmax())
    greedy_actions = {train.select_action(net, state, 0.0, NUM_ACTIONS, "cpu") for _ in range(20)}
    assert greedy_actions == {best_action}, f"epsilon 0.0 picked {greedy_actions}, expected only {best_action}"



# Francis

def run_updates(reward, done, gamma, updates=400):
    """Train on a tiny buffer whose target network predicts about 10 for every action.

    Returns what the online network learned for Q(s, a), the Bellman target it
    should have learned, the last loss, and whether the target network stayed unchanged.
    """
    torch.manual_seed(0)
    online_net = train.QNetwork(NUM_ACTIONS)
    target_net = train.QNetwork(NUM_ACTIONS)
    target_net.load_state_dict(online_net.state_dict())
    with torch.no_grad():
        list(target_net.parameters())[-1].fill_(10.0)  # output-layer bias: target Q-values near 10
    target_before = [p.clone() for p in target_net.parameters()]

    buffer = train.ReplayBuffer(capacity=64)
    states = [fake_state() for _ in range(8)]
    for i in range(64):
        buffer.push(states[i % 8], i % NUM_ACTIONS, reward, states[(i + 1) % 8], done)

    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)
    for _ in range(updates):
        loss = train.train_step(online_net, target_net, optimizer, buffer, 16, gamma, "cpu")

    batch_states, batch_actions, _, batch_next_states, _ = buffer.sample(64)
    with torch.no_grad():
        q_taken = online_net(torch.as_tensor(batch_states))
        learned = q_taken.gather(1, torch.as_tensor(batch_actions).unsqueeze(1)).mean().item()
        next_max = target_net(torch.as_tensor(batch_next_states)).max(dim=1).values.mean().item()
    should_learn = reward if done else reward + gamma * next_max
    target_unchanged = all(torch.equal(a, b) for a, b in zip(target_before, target_net.parameters()))
    return learned, should_learn, loss, target_unchanged


def check_train_step_basics():
    _, _, loss, target_unchanged = run_updates(reward=2.0, done=True, gamma=0.99, updates=5)
    assert isinstance(loss, float), f"returned {type(loss).__name__}, expected a float (use loss.item())"
    assert np.isfinite(loss), "loss is NaN or inf"
    assert target_unchanged, "train_step changed the target network; only the online network should change"


def check_bellman_done():
    learned, should_learn, _, _ = run_updates(reward=2.0, done=True, gamma=0.99)
    assert abs(learned - should_learn) < 0.75, (
        f"learned Q = {learned:.2f}, expected {should_learn:.2f}: when done, the target should be just r")


def check_bellman_not_done():
    learned, should_learn, _, _ = run_updates(reward=2.0, done=False, gamma=0.5)
    assert abs(learned - should_learn) < 0.75, (
        f"learned Q = {learned:.2f}, expected {should_learn:.2f}: "
        "target should be r + gamma * max Q_target(s'), using the target network")



# Runner
CHECKS = [
    # (owner, what is checked, check function, checks that must pass first)
    ("Alain", "parse_args: defaults match the baseline", check_parse_args, []),
    ("Alain", "make_env: (4, 84, 84) uint8 frames, 6 actions", check_make_env, []),
    ("Alain", "ReplayBuffer: capacity, push, sample shapes", check_replay_buffer, []),
    ("Alain", "log_episode: header and one row per episode", check_log_episode, []),
    ("Principie", "QNetwork: (B, 4, 84, 84) in, (B, 6) out", check_q_network, []),
    ("Principie", "get_epsilon: linear decay, then constant", check_get_epsilon, []),
    ("Principie", "select_action: random at 1.0, greedy at 0.0", check_select_action, [check_q_network]),
    ("Francis", "train_step: float loss, target net untouched", check_train_step_basics,
     [check_replay_buffer, check_q_network]),
    ("Francis", "Bellman target when the episode ended: r", check_bellman_done,
     [check_replay_buffer, check_q_network]),
    ("Francis", "Bellman target otherwise: r + gamma * max Q_target", check_bellman_not_done,
     [check_replay_buffer, check_q_network]),
]


def main():
    results = {}
    print(f"{'Owner':<11}{'Result':<10}Check")
    for owner, description, check, needs in CHECKS:
        detail = ""
        if any(results.get(need) != "PASS" for need in needs):
            result = "WAITING"
            missing = [need.__name__.replace("check_", "") for need in needs if results.get(need) != "PASS"]
            detail = f"  (waiting for: {', '.join(missing)})"
        else:
            try:
                info = check()
                result = "PASS"
                detail = f"  ({info})" if info else ""
            except NotImplementedError:
                result = "NOT DONE"
            except AssertionError as error:
                result = "FAIL"
                detail = f"\n{'':<21}-> {error}"
            except Exception as error:
                result = "FAIL"
                detail = f"\n{'':<21}-> {type(error).__name__}: {error}"
        results[check] = result
        print(f"{owner:<11}{result:<10}{description}{detail}")

    passed = sum(result == "PASS" for result in results.values())
    print(f"\n{passed}/{len(CHECKS)} checks pass.")
    if passed == len(CHECKS):
        print("Step 2 done. Next, Step 3 smoke test:")
        print("  uv run train.py --member Test --exp 1 --total-steps 15000")


if __name__ == "__main__":
    main()