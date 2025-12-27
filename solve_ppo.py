'''
Proximal Policy Optimization (PPO) implementation for training agents in GridWorld environments.
'''
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from collections import deque
import pandas
import seaborn
from pathlib import Path
import time
from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv
from utils.DirectoryManager import DirectoryManager
from utils.SlidingWindow import SlidingWindow
from NeuralRewardMachines.LTL_tasks import formulas, ltls
from GridWorldEnvWrapper import GridWorldEnvWrapper

dm = DirectoryManager()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = 1  # Global variable for current run number

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
            obs_dim = env.state_space_size if isinstance(env.state_space_size, int) else int(np.prod(env.state_space_size))
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
                img = img.to(device).float()
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                cnn_feat = self.cnn(img)
                if dfa is not None:
                    dfa = dfa.to(device).float()
                    if dfa.dim() == 1:
                        dfa = dfa.unsqueeze(0)
                    x = torch.cat([cnn_feat, dfa], dim=1)
                else:
                    x = cnn_feat
            else:
                # just image tensor
                img = state.to(device).float()
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                cnn_feat = self.cnn(img)
                x = cnn_feat
        else:
            x = state.to(device).float()
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

def obs_to_state(obs, env: GridWorldEnv):
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

def stack_states(states_list, env: GridWorldEnv):
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

def train_ppo(env: GridWorldEnv, hidden=128,
              episodes=10_000, steps=256, minibatch_size=64, epochs=4, 
              gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, lr=3e-4, 
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    """Train PPO agent in the given environment."""
    # TODO: Implement a better early stop strategy
    model = ActorCritic(env, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer()
    model_folder = dm.get_model_folder()
    model_name = f"PPO_{RUN}.pth"
    model_file = model_folder + model_name
    log_name = f"training_PPO.csv"
    log_folder = dm.get_log_folder()
    log_file = log_folder + log_name
    loss_log_name = f"loss_PPO_{RUN}.csv"
    loss_log_file_path = log_folder + loss_log_name
    loss_log_file = open(loss_log_file_path, "w")
    loss_log_file.write("episode,total_steps,episode_reward,episode_steps,done,truncated,"
                        "avg_policy_loss,avg_value_loss,avg_entropy,avg_reward,k1,k2,k3\n")
    if Path(log_file).exists():
        f_log= open(log_file, "a")
    else:
        f_log= open(log_file, "w")
        f_log.write("episode,total_steps,episode_reward,episode_steps,done,truncated,average_reward,run\n")
    # Training loop
    episode = 0
    total_steps = 0
    sw = SlidingWindow(size=100)
    plot_n = 0
    print(f"Training PPO on {device}")
    print(f"Episodes: {episodes}, Steps per rollout: {steps}")
    while episode < episodes:
        obs, _, _ = env.reset()
        state = obs_to_state(obs, env)
        episode_reward = 0.0
        episode_rewards = []
        episode_length = 0
        done = False
        truncated = False
        # Collect rollout
        for step in range(steps):
            if done or truncated:
                # Start new episode
                obs, _, _ = env.reset()
                state = obs_to_state(obs, env)
                episode_reward = 0.0
                episode_length = 0
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
            next_state = obs_to_state(next_obs, env)
            state = next_state
            # Check if episode ended
            if done or truncated:
                episode += 1
                episode_rewards.append(episode_reward)
                # Log episode 
                sw.add(episode_reward)
                f_log.write(f"{episode},{total_steps},{episode_reward:.2f},{episode_length},"
                            f"{1 if done else 0},{1 if truncated else 0},{sw.average():.2f},{RUN}\n")
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
        # PPO update for n_epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
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
                states_stacked = stack_states(batch_states, env)
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
                k1 = (-log_ratio).mean()
                k2 = (log_ratio ** 2 / 2).mean()
                k3 = (ratio - 1 - log_ratio).mean()
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
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        # Log metrics
        avg_policy_loss = total_policy_loss / n_updates if n_updates > 0 else 0
        avg_value_loss = total_value_loss / n_updates if n_updates > 0 else 0
        avg_entropy = total_entropy / n_updates if n_updates > 0 else 0
        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        loss_log_file.write(f"{episode},{total_steps},{episode_reward:.2f},{episode_length},{done},{truncated},"
                            f"{avg_policy_loss:.4f},{avg_value_loss:.4f},{avg_entropy:.4f},{avg_reward:.2f},"
                            f"{k1.item():.4f},{k2.item():.4f},{k3.item():.4f}\n")
        # Log episode 
        # with open(log_file, "a") as logf:
        #     logf.write(f"{episode},{total_steps},{episode_reward:.2f},{episode_length},"
        #                f"{1 if done else 0},{1 if truncated else 0},"
        #               f"{avg_value_loss:.4f},{avg_policy_loss:.4f},{avg_entropy:.4f}\n")
        # Clear buffer for next rollout
        buffer.clear()
    # Save model
    loss_log_file.close()
    f_log.close()
    torch.save(model.state_dict(), model_file)
    print(f"\nTraining completed! Model saved to {model_file}")

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on GridWorld environment")
    parser.add_argument("--image-state", action='store_true', help="Set state type to image")
    parser.add_argument("--formulas", type=int, default=10, help="Number of formulas to consider from LTL_tasks")
    parser.add_argument("--no-automaton", action='store_true', help="If set, do not use automaton states")
    parser.add_argument("--external-automaton", action='store_true', help="If set, use external automaton")
    parser.add_argument("--runs", type=int, default=3, help="Experiment runs per formula")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size for the model")
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps per rollout")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Minibatch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per rollout")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for optimizer")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    state_type = "image" if args.image_state else "symbolic"
    n_formulas = args.formulas
    use_automaton = not args.no_automaton
    external_automaton = args.external_automaton
    if external_automaton:
        use_automaton = True
    runs = args.runs
    hidden = args.hidden
    episodes = args.episodes
    steps = args.steps
    minibatch_size = args.minibatch_size
    epochs = args.epochs
    clip_epsilon = args.clip_epsilon
    lr = args.lr
    vf_coef = args.vf_coef
    ent_coef = args.ent_coef
    max_grad_norm = args.max_grad_norm
    seed = args.seed
    # Log experiment parameters
    with open(dm.get_experiment_folder() + "experiment_parameters.txt", "w") as f:
        f.write(f"# Experiment parameters:\n")
        f.write(f"# State type: {state_type}\n")
        f.write(f"# Number of formulas: {n_formulas}\n")
        f.write(f"# Use automaton states: {use_automaton}\n")
        f.write(f"# Use external automaton: {external_automaton}\n")
        f.write(f"# Runs per formula: {runs}\n")
        f.write(f"# hidden layer size: {hidden}\n")
        f.write(f"# training episodes: {episodes}, steps per rollout: {steps}, minibatch size: {minibatch_size}, epochs: {epochs}\n")
        f.write(f"# clip_epsilon: {clip_epsilon}, lr: {lr}\n") 
        f.write(f"# vf_coef: {vf_coef}, ent_coef: {ent_coef}, max_grad_norm: {max_grad_norm}\n")
        f.write(f"# seed: {seed}\n")
    set_seed(seed)
    formulas = formulas[:n_formulas]
    for i in range(len(formulas)):
        formula = formulas[i]
        ltl = ltls[i]
        print(f"Running experiments for: {formula[2]}")
        dm.set_formula_name(formula[2].replace(" ", "_"))
        # Create environment
        env = GridWorldEnvWrapper(formula=formula, state_type=state_type, use_dfa_state=use_automaton, external_automaton=external_automaton, ltl=ltl)
        for r in range(runs):
            print(f" Experiment {r+1} / {runs}")
            RUN = r + 1
            # Train PPO
            train_ppo(env, hidden=hidden, episodes=episodes, steps=steps, minibatch_size=minibatch_size, 
                      epochs=epochs, clip_epsilon=clip_epsilon, lr=lr, vf_coef=vf_coef, 
                      ent_coef=ent_coef, max_grad_norm=max_grad_norm)
        # Plot learning curves
        print(f"Plotting learning curves for: {formula[2]}")
        df = pandas.read_csv(dm.get_log_folder() + f"training_PPO.csv")
        plt.title(f"PPO Learning Curve - {formula[2]}")
        seaborn.relplot(data=df, kind="line", x="episode", y="average_reward")
        plt.savefig(dm.get_plot_folder() + f"PPO_Learning_Curve.png")
        plt.clf()
        plt.title(f"PPO Learning Curve per run - {formula[2]}")
        seaborn.relplot(data=df, kind="line", x="episode", y="average_reward", col="run", hue="run")
        plt.savefig(dm.get_plot_folder() + f"PPO_Learning_Curve_per_run.png")
        plt.clf()