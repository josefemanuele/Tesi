''' Solve GridWorldEnv using DQN algorithm. '''
import random
import collections
import argparse
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv
from NeuralRewardMachines.LTL_tasks import formulas
from utils.DirectoryManager import DirectoryManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dm = DirectoryManager()
RUN = 0
max_steps = 50

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, np.array(actions), np.array(rewards, dtype=np.float32), next_states, np.array(dones, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, env: GridWorldEnv, hidden=128):
        super().__init__()
        self.state_type = env.state_type
        self.use_dfa = env.use_dfa_state

        if self.state_type == "image":
            # small CNN to extract features from 3x64x64 images
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            # compute cnn output size with a dummy pass
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                cnn_out = self.cnn(dummy).shape[1]
            if self.use_dfa:
                dfa_dim = env.automaton.num_of_states
            else:
                dfa_dim = 0
            self.fc = nn.Sequential(
                nn.Linear(cnn_out + dfa_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n),
            )
        else:
            # symbolic: observation is a vector (e.g. x,y,dfa_state) or (x,y) + dfa
            obs_dim = env.state_space_size if isinstance(env.state_space_size, int) else int(np.prod(env.state_space_size))
            if self.use_dfa:
                obs_dim += env.automaton.num_of_states
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n),
            )

    def forward(self, state):
        # state can be:
        #  - symbolic: single tensor vector
        #  - image: tuple (one_hot_dfa_tensor, image_tensor) or just image_tensor if not using dfa
        if self.state_type == "image":
            if isinstance(state, tuple) or isinstance(state, list):
                dfa, img = state
                img = img.to(device).float().unsqueeze(0)
                cnn_feat = self.cnn(img)
                if dfa is not None:
                    dfa = dfa.to(device).float().unsqueeze(0)
                    x = torch.cat([cnn_feat, dfa], dim=1)
                else:
                    x = cnn_feat
            else:
                # just image tensor
                img = state.to(device).float().unsqueeze(0)
                cnn_feat = self.cnn(img)
                x = cnn_feat
            return self.fc(x)
        else:
            x = state.to(device).float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.fc(x)

def obs_to_state(obs, env: GridWorldEnv):
    # normalize/convert observation to tensors in the shape expected by DQN.forward
    if env.state_type == "symbolic":
        # If using DFA state we expect obs to contain the symbolic state and a one-hot/dfa part,
        # or a single concatenated vector. Ensure the return has shape (obs_dim,)
        if env.use_dfa_state:
            # obs might be (state_vec, one_hot) or already concatenated array
            arr = np.array(obs)
            pos_dim = env.state_space_size
            pos = arr[:pos_dim].astype(np.float32)
            idx = int(arr[pos_dim])
            one_hot = np.zeros(env.automaton.num_of_states, dtype=np.float32)
            one_hot[idx] = 1.0
            state_vec = np.concatenate([pos, one_hot]).astype(np.float32)
            return torch.as_tensor(state_vec, dtype=torch.float32, device=device)
        else:
            # simple numeric/vector observation
            if isinstance(obs, np.ndarray):
                return torch.from_numpy(obs.astype(np.float32))
            else:
                return torch.tensor(np.array(obs).astype(np.float32))
    else:
        # image mode: reset/step return either image only or [one_hot, image_tensor]
        if env.use_dfa_state:
            one_hot = obs[0]
            img = obs[1]
            one_hot_t = torch.tensor(np.array(one_hot).astype(np.float32))
            if not torch.is_tensor(img):
                img_t = torch.tensor(np.array(img).astype(np.float32))
            else:
                img_t = img.float()
            return (one_hot_t, img_t)
        else:
            img = obs
            if not torch.is_tensor(img):
                img_t = torch.tensor(np.array(img).astype(np.float32))
            else:
                img_t = img.float()
            return img_t

def train(env: GridWorldEnv, episodes=10000, batch_size=64, gamma=0.99, lr=1e-4,
          buffer_capacity=20000, target_update=1000, start_train=1000):
    ''' Train DQN agent in the given environment.'''
    online = DQN(env).to(device)
    target = DQN(env).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    steps_done = 0
    eps_start, eps_end, eps_decay = 1.0, 0.05, 30000
    reward_sum = 0.0
    model_name = f"DQN_{RUN}.pth"
    model_file = dm.get_model_folder() + model_name
    log_name = f"training_DQN_{RUN}.csv"
    log_file = dm.get_log_folder() + log_name
    early_stop = episodes / 5  # set early stop to 20% of episodes
    count = 0
    logging_rate = int(episodes / 20) # log every 5% of episodes
    with Path(log_file).open("w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["episode", "steps", "reward", "epsilon", "buffer_size"])

        for ep in range(1, episodes + 1):
            obs, _, _ = env.reset()
            state = obs_to_state(obs, env)
            total_reward = 0.0
            done = False
            truncated = False
            for t in range(max_steps):
                eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
                steps_done += 1
                # select action
                if random.random() < eps_threshold:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        qvals = online(state)
                        action = int(torch.argmax(qvals, dim=1).cpu().numpy()[0])
                # take step and store in buffer
                next_obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                next_state = obs_to_state(next_obs, env)
                terminal = bool(done or truncated)
                buffer.push(state, action, reward, next_state, terminal)
                state = next_state
                # optimize
                if len(buffer) >= max(64, start_train):
                    states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)
                    # prepare tensors
                    def stack_states(s_list):
                        # convert list of states (some are tuples for images) into batched tensors for forward
                        if env.state_type == "image":
                            if env.use_dfa_state:
                                dfa_batch = torch.stack([s[0] for s in s_list]).to(device)
                                img_batch = torch.stack([s[1] for s in s_list]).to(device)
                                return (dfa_batch, img_batch)
                            else:
                                return torch.stack([s for s in s_list]).to(device)
                        else:
                            return torch.stack([s for s in s_list]).to(device)
                    states_t = stack_states(states_b)
                    next_states_t = stack_states(next_states_b)
                    actions_t = torch.tensor(actions_b, device=device, dtype=torch.long).unsqueeze(1)
                    rewards_t = torch.tensor(rewards_b, device=device, dtype=torch.float32).unsqueeze(1)
                    dones_t = torch.tensor(dones_b, device=device, dtype=torch.float32).unsqueeze(1)
                    q_values = online(states_t).gather(1, actions_t)
                    with torch.no_grad():
                        next_q = target(next_states_t)
                        next_q_max = next_q.max(1)[0].unsqueeze(1)
                        target_q = rewards_t + (1.0 - dones_t) * gamma * next_q_max # dones?? TODO: check.
                    loss = F.mse_loss(q_values, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if steps_done % target_update == 0:
                    target.load_state_dict(online.state_dict())
                if terminal:
                    break
            reward_sum += total_reward
            # simple logging
            if ep % logging_rate == 0:
                line = f"Episode {ep:4d} | steps {steps_done:6d} | steps/episode {steps_done/ep:4.2f} | reward {total_reward:5.2f} | reward/episode {reward_sum/ep:5.2f} | epsilon {eps_threshold:.3f} | buffer {len(buffer)}"
                print(line)
            writer.writerow([ep, steps_done, f"{total_reward:.2f}", f"{eps_threshold:.3f}", len(buffer)])
            # Early stopping if solved consistently
            count = count + 1 if done else 0
            if count >= early_stop:
                print(f"Early stopping as agent consistently solved the environment in the last {count} episodes.")
                break
    print(f"Saved training log to {log_file}")
    # Save model
    torch.save(online.state_dict(), model_file)
    print(f"Saved trained model to {model_file}")
    return online

def print_sample_execution(env: GridWorldEnv, model: DQN):
    obs, _, _ = env.reset()
    state_tensor = obs_to_state(obs, env)
    total_reward = 0.0
    done = False
    truncated = False
    info = {}
    steps = 0
    for t in range(max_steps):
        steps += 1
        with torch.no_grad():
            qvals = model(state_tensor)
            # handle batched or single-sample outputs
            if qvals.dim() == 2:
                action = int(qvals.argmax(dim=1)[0].cpu().item())
            else:
                action = int(qvals.argmax().cpu().item())
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        past_state = state_tensor
        state_tensor = obs_to_state(next_obs, env)
        print(f"Step {steps}: state = {past_state}, action={action}, next_state={state_tensor}, reward={reward:.2f}, total_reward={total_reward:.4f}, info={info}")
        if done or truncated:
            print("Done:", done, "Truncated:", truncated)
            break
    print(f"Sample execution: steps={steps}, total_reward={total_reward:.2f}, success={done}")

def evaluate(model: DQN, env: GridWorldEnv, runs: int, episodes: int):
    max_steps = 50
    model.eval()
    out_csv = dm.get_eval_folder() + f"eval_DQN_{RUN}.csv"
    with Path(out_csv).open("w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["run", "episode", "steps", "total_reward", "success"])
        for run in range(1, runs + 1):
            for ep in range(1, episodes + 1):
                obs, _, _ = env.reset()
                state_tensor = obs_to_state(obs, env)
                total_reward = 0.0
                done = False
                truncated = False
                info = {}
                steps = 0
                for t in range(max_steps):
                    steps += 1
                    with torch.no_grad():
                        qvals = model(state_tensor)
                        # handle batched or single-sample outputs
                        if qvals.dim() == 2:
                            action = int(qvals.argmax(dim=1)[0].cpu().item())
                        else:
                            action = int(qvals.argmax().cpu().item())
                    next_obs, reward, done, truncated, info = env.step(action)
                    total_reward += float(reward)
                    state_tensor = obs_to_state(next_obs, env)
                    if done or truncated:
                        break
                writer.writerow([run, ep, steps, f"{total_reward:.4f}", 1 if done else 0])
    # compute simple summary
    data = np.genfromtxt(str(out_csv), delimiter=",", names=True, dtype=None, encoding=None)
    rewards = np.array([float(x["total_reward"]) for x in data])
    successes = np.array([int(x["success"]) for x in data])
    steps_array = np.array([int(x["steps"]) for x in data])
    print(f"Saved post-training evaluations to {out_csv}")
    print(f"Performances: episodes {len(rewards)} | steps/episode {steps_array.mean():.2f} | mean_reward: {rewards.mean():.2f} | success_rate: {successes.mean():.2f}")
    print_sample_execution(env, model)

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--state_type", choices=["symbolic", "image"], default="symbolic")
    parser.add_argument("--use_dfa", action="store_true", default=True)
    args = parser.parse_args()
    print(f"Using {device}.")
    # Train for all formulas in LTL_tasks.py
    for formula in formulas:
        dm.set_formula_name(formula[2])
        env = GridWorldEnv(formula=formula,
                           render_mode="human", state_type=args.state_type, use_dfa_state=args.use_dfa, train=True)
        # Train multiple runs for each formula.
        for run in range(1, args.runs + 1):
            RUN = run
            print(f"\n=== Training formula: {formula[2]} | Run {RUN} ===")
            model = train(env, episodes=args.episodes)
            evaluate(model, env, runs=1, episodes=10)
    