''' Implement DDQN algorithm for training agents in GridWorld environment. '''
import random
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv

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
        automaton = env.external_automaton if env.external_automaton else env.automaton
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
                dfa_dim = automaton.num_of_states
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
                obs_dim += automaton.num_of_states
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
                img = img.float()
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                cnn_feat = self.cnn(img)
                if dfa is not None:
                    dfa = dfa.float()
                    if dfa.dim() == 1:
                        dfa = dfa.unsqueeze(0)
                    x = torch.cat([cnn_feat, dfa], dim=1)
                else:
                    x = cnn_feat
            else:
                # just image tensor
                img = state.float().unsqueeze(0)
                cnn_feat = self.cnn(img)
                x = cnn_feat
            return self.fc(x)
        else:
            x = state.float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.fc(x)

def obs_to_state(obs, env: GridWorldEnv, device):
    automaton = env.external_automaton if env.external_automaton else env.automaton
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
            one_hot = np.zeros(automaton.num_of_states, dtype=np.float32)
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

def stack_states(states_list, env: GridWorldEnv, device):
    """Stack list of states into batched tensors."""
    if env.state_type == "image":
        if env.use_dfa_state:
            dfa_batch = torch.stack([s[0] for s in states_list]).to(device)
            img_batch = torch.stack([s[1] for s in states_list]).to(device)
            return (dfa_batch, img_batch)
        else:
            return torch.stack([s for s in states_list]).to(device)
    else:
        return torch.stack([s for s in states_list]).to(device)
    
def train_ddqn(device, env: GridWorldEnv, episodes=10_000, max_steps=256, 
          batch_size=64, buffer_capacity=20_000, gamma=0.99, lr=1e-4,
          start_train=1_000, target_update=1_000):
    ''' Train DQN agent in the given environment.'''
    online = DQN(env).to(device)
    target = DQN(env).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    data = list()
    # Start training loop
    total_steps = 0
    eps_start, eps_end, eps_decay = 1.0, 0.05, 30000
    # TODO: Check early stop strategy
    early_stop = episodes / 5  # set early stop to 20% of episodes
    count = 0
    # logging_rate = int(episodes / 20) # log every 5% of episodes
    for ep in range(1, episodes + 1):
        obs, _, _ = env.reset()
        state = obs_to_state(obs, env, device)
        episode_reward = 0.0
        done = False
        truncated = False
        for step in range(max_steps):
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay)
            total_steps += 1
            # select action
            if random.random() < eps_threshold:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = online(state)
                    action = int(torch.argmax(qvals, dim=1).cpu().numpy()[0])
            # take step and store in buffer
            next_obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            next_state = obs_to_state(next_obs, env, device)
            terminal = bool(done or truncated)
            buffer.push(state, action, reward, next_state, terminal)
            state = next_state
            # optimize
            if len(buffer) >= max(64, start_train):
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)
                states_t = stack_states(states_b, env, device)
                next_states_t = stack_states(next_states_b, env, device)
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
            if total_steps % target_update == 0:
                target.load_state_dict(online.state_dict())
            if terminal:
                break
        # simple logging
        # if ep % logging_rate == 0:
        #     line = f"Episode {ep:4d} | steps {total_steps:6d} | steps/episode {total_steps/ep:4.2f} | reward {total_reward:5.2f} | reward/episode {reward_sum/ep:5.2f} | epsilon {eps_threshold:.3f} | buffer {len(buffer)}"
        #     print(line)
        # Early stopping if solved consistently
        data.append((ep, episode_reward, step + 1, done, truncated, total_steps))
        # Early stopping check
        count = count + 1 if done else 0
        if count >= early_stop:
            print(f"Early stopping as agent consistently solved the environment in the last {count} episodes.")
            break
    return online, data