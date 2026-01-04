''' Implement PPO algorithm.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv
from utils.DirectoryManager import DirectoryManager
from utils.SlidingWindow import SlidingWindow

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    def __init__(self, env: GridWorldEnv, hidden=128):
        super().__init__()
        self.state_type = env.state_type
        self.use_dfa = env.use_dfa_state
        if self.state_type == "image":
            # CNN to extract features from 3x64x64 images
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
            # Actor network
            self.actor = nn.Sequential(
                nn.Linear(cnn_out + dfa_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n)
            )
            # Critic network
            self.critic = nn.Sequential(
                nn.Linear(cnn_out + dfa_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        else:
            # symbolic: observation is a vector (e.g. x,y,dfa_state) or (x,y)
            obs_dim = env.state_space_size
            if self.use_dfa:
                obs_dim += env.automaton.num_of_states
            # Actor network
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n)
            )
            # Critic network
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

    def forward(self, state):
        """Forward pass through the network.
        Returns:
            logits: action logits from actor
            value: state value from critic
        """
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
                img = state.float()
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                cnn_feat = self.cnn(img)
                x = cnn_feat
        else:
            x = state.float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, state):
        """Sample action from policy and get value estimate."""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        # print(f"State: {state} | probs: {probs} | action: {action.item()} | value: {value.item():.2f} | log_prob: {log_prob.item():.2f} | entropy: {entropy.item():.2f}")
        return action.item(), log_prob.item(), value.item(), entropy.item()

def obs_to_state(obs, env: GridWorldEnv, device):
    """Convert environment observation to tensor format."""
    if env.state_type == "symbolic":
        if env.use_dfa_state:
            arr = np.array(obs)
            pos_dim = env.state_space_size
            pos = arr[:pos_dim].astype(np.float32)
            idx = int(arr[pos_dim])
            one_hot = np.zeros(env.automaton.num_of_states, dtype=np.float32)
            one_hot[idx] = 1.0
            state_vec = np.concatenate([pos, one_hot]).astype(np.float32)
            return torch.as_tensor(state_vec, dtype=torch.float32, device=device)
        else:
            if isinstance(obs, np.ndarray):
                return torch.from_numpy(obs.astype(np.float32))
            else:
                return torch.tensor(np.array(obs).astype(np.float32))
    else:
        # image mode
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

class RolloutBuffer:
    """Buffer for storing rollout data during on-policy collection."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.truncateds = []

    def push(self, state, action, reward, log_prob, value, done, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.truncateds = []

    def __len__(self):
        return len(self.rewards)

def compute_gae(rewards, values, dones, truncateds, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0
    # Add next_value to values list for computation
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - (dones[t] or truncateds[t])
        # next_non_terminal = 1.0 - (dones[t])
        next_value_t = values[t + 1]
        # TD error: delta = r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
        # GAE: A = delta + gamma * lambda * A'
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
    # Returns are advantages + values
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

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

def train_ppo(device, env: GridWorldEnv, hidden=128,
              episodes=10_000, steps=256, minibatch_size=64, epochs=4, 
              gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, lr=3e-4, 
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    """Train PPO agent in the given environment."""
    # TODO: Implement a better early stop strategy.
    # TODO: Handle data with Pandas dataframe.
    model = ActorCritic(env, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer()
    data = list()
    # Start training loop
    episode = 0
    total_steps = 0
    # TODO: Switch to Seaborn sliding window implementation
    sw = SlidingWindow(size=100) # To calculate sliding window averaged value
    while episode < episodes:
        episode_length = 0
        episode_reward = 0.0
        episode_rewards = [] # To calculate average reward per training iteration
        done = False
        truncated = False
        obs, _, _ = env.reset()
        state = obs_to_state(obs, env, device)
        # Collect rollout
        for step in range(steps):
            if done or truncated:
                # Restart episode
                episode_length = 0
                episode_reward = 0.0
                obs, _, _ = env.reset()
                state = obs_to_state(obs, env, device)
            # Evaluate action, log_prob, and value
            with torch.no_grad():
                action, log_prob, value, _ = model.get_action_and_value(state)
            # Take action in environment
            next_obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            # Store transition
            buffer.push(state, action, reward, log_prob, value, done, truncated)
            # Update state
            next_state = obs_to_state(next_obs, env, device)
            state = next_state
            # Check if episode ended
            if done or truncated:
                episode += 1
                episode_rewards.append(episode_reward)
                sw.add(episode_reward)
                # Store episode data
                data.append((episode, episode_reward, episode_length, done, truncated, total_steps, sw.average()))
                # # If we've collected enough steps or reached episode limit, break to train
                # if len(buffer) >= minibatch_size or episode >= episodes:
                # #     print(f"Collected {len(buffer)} steps, proceeding to update.")
                #     break 
        # If buffer is empty or we've finished all episodes, continue
        if len(buffer) == 0 or episode >= episodes:
            break
        # Compute advantages and returns
        with torch.no_grad():
            _, next_value = model.forward(state)
            next_value = next_value.item()
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones, buffer.truncateds,
            next_value, gamma, gae_lambda
        )
        # Convert to tensors
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        old_log_probs_t = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        # Prepare batch data
        states_batch = buffer.states
        actions_batch = torch.tensor(buffer.actions, dtype=torch.long, device=device)
        # Indicators for loss logging
        # total_policy_loss = 0
        # total_value_loss = 0
        # total_entropy = 0
        # n_updates = 0
        for epoch in range(epochs):
            # Shuffle indices
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            # Mini-batch updates
            for start in range(0, len(buffer), minibatch_size):
                end = min(start + minibatch_size, len(buffer))
                batch_indices = indices[start:end]
                # Get batch data
                batch_states = [states_batch[i] for i in batch_indices]
                batch_actions = actions_batch[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                # Forward pass
                states_stacked = stack_states(batch_states, env, device)
                logits, values = model.forward(states_stacked)
                # Compute policy loss
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                # Ratio for PPO
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                # KL divergence: k3
                # k1 = (-log_ratio).mean()
                # k2 = (log_ratio ** 2 / 2).mean()
                # k3 = (ratio - 1 - log_ratio).mean()
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                # Value loss
                values = values.squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)
                # Total loss
                # policy loss 0.5
                # value loss 1
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                # total_policy_loss += policy_loss.item()
                # total_value_loss += value_loss.item()
                # total_entropy += entropy.item()
                # n_updates += 1
        # # Log metrics
        # avg_policy_loss = total_policy_loss / n_updates if n_updates > 0 else 0
        # avg_value_loss = total_value_loss / n_updates if n_updates > 0 else 0
        # avg_entropy = total_entropy / n_updates if n_updates > 0 else 0
        # avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        # print(f"{episode},{total_steps},{episode_reward:.2f},{episode_length},{done},{truncated},"
        #                     f"{avg_policy_loss:.4f},{avg_value_loss:.4f},{avg_entropy:.4f},{avg_reward:.2f},"
        #                     f"{k1.item():.4f},{k2.item():.4f},{k3.item():.4f}\n")
        # Clear buffer for next rollout
        buffer.clear()
    # TODO: Remove model saving (?)
    # Save model
    # torch.save(model.state_dict(), model_file)
    # print(f"\nTraining completed! Model saved to {model_file}")
    return model, data