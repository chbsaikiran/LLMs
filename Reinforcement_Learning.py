import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from collections import deque

# Hyperparameters
gamma = 0.99
lr = 1e-3
tau = 0.005  # for soft update
batch_size = 64
memory_size = 10000
min_memory = 1000
episodes = 500

# Define Q-Network (used for both fast and slow learner)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)

# Soft update function
def soft_update(target_net, online_net, tau):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

# Initialize environment and networks
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

online_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())  # Sync initially

optimizer = optim.Adam(online_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(memory_size)

# Training loop
for episode in range(episodes):
    state = env.reset()
    state = np.array(state)
    total_reward = 0

    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = online_net(state_tensor)
        action = q_values.argmax().item() if random.random() > 0.1 else env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Train
        if len(replay_buffer) > min_memory:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Fast learner chooses next action
            next_actions = online_net(next_states).argmax(dim=1, keepdim=True)

            # Slow learner evaluates it
            next_q_values = target_net(next_states).gather(1, next_actions)

            target_q = rewards + gamma * next_q_values * (1 - dones)

            q_values = online_net(states).gather(1, actions)
            loss = nn.MSELoss()(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Soft update target network (slow learner)
            soft_update(target_net, online_net, tau)

    print(f"Episode {episode} | Total Reward: {total_reward}")

env.close()
