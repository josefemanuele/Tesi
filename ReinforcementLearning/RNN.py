import torch
import torch.nn as nn
from torch.utils.data import Dataset

class RewardTrajectory:

    def __init__(self, window=10):
        self.window = window
        self.reset()

    def add(self, reward):
        self.trajectory.append(reward)

    def reset(self):
        self.trajectory = [0 for i in range(self.window)]  # Start with a dummy trajectory to avoid empty buffer issues

    def get_trajectory(self):
        return self.trajectory.copy()

class RewardTrajectoryDataset(Dataset):

    def __init__(self, trajectories, window=10, device='cpu'):
        self.X = []
        self.y = []
        for traj in trajectories:
            if len(traj) <= window:
                continue
            for i in range(0, len(traj) - window):
                seq = traj[i:i+window]
                target = traj[i+window]
                self.X.append(seq)
                self.y.append(target)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=device).unsqueeze(-1)
        self.y = torch.tensor(self.y, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class RNN(nn.Module):

    def __init__(self, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, h = self.gru(x)
        latent_state = h[-1]
        pred_reward = self.head(latent_state)
        if len(pred_reward) == 1:
            return pred_reward, latent_state

        return pred_reward.squeeze(), latent_state
    
