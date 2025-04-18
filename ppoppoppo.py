import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Target Function ---
def target_function(x):
    """근사하려는 목표 5차 다항 함수"""
    return np.power(x, 5) + 1.5 * np.power(x, 4) - 5.0 * np.power(x, 3) - 7.5 * np.power(x, 2) + 4.0 * x + 6.0

# --- 2. Actor Network ---
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_log_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = torch.exp(log_std)
        return mu, std

# --- 3. Critic Network ---
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

# --- 4. PPO Agent ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, lambda_gae=0.95, epsilon_clip=0.2, K_epochs=10, hidden_dim=64):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs

        self.actor = Actor(state_dim, action_dim, hidden_dim).float()
        self.critic = Critic(state_dim, hidden_dim).float()
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        self.actor_old = Actor(state_dim, action_dim, hidden_dim).float()
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mse_loss = nn.MSELoss()
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mu, std = self.actor_old(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def store_transition(self, state, action, reward, action_log_prob, value):
        self.buffer.append((state, action, reward, action_log_prob, value))

    def calculate_gae(self, rewards, values, next_value):
        advantages = []
        gae = 0
        last_value = next_value
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * last_value - values[i]
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages.insert(0, gae)
            last_value = values[i]
        returns = [a + v for a, v in zip(advantages, values)]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def update(self):
        if not self.buffer: return

        states = torch.FloatTensor([t[0] for t in self.buffer]).unsqueeze(1)
        actions = torch.FloatTensor([t[1] for t in self.buffer]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in self.buffer]).unsqueeze(1)
        old_log_probs = torch.FloatTensor([t[3] for t in self.buffer]).unsqueeze(1)
        old_values = torch.FloatTensor([t[4] for t in self.buffer]).unsqueeze(1)

        with torch.no_grad():
            next_value_state = torch.FloatTensor([self.buffer[-1][0]]).unsqueeze(1)
            next_value = self.critic(next_value_state).item()

        advantages, returns = self.calculate_gae(rewards.squeeze().tolist(), old_values.squeeze().tolist(), next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.unsqueeze(1)
        returns = returns.unsqueeze(1)

        for _ in range(self.K_epochs):
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            new_values = self.critic(states)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(new_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer = []

# --- 5. Training Loop ---
if __name__ == '__main__':
    state_dim = 1
    action_dim = 1
    lr = 3e-4
    gamma = 0.9
    lambda_gae = 0.9
    epsilon_clip = 0.2
    K_epochs = 5
    hidden_dim = 128
    max_steps = 50000
    update_interval = 200

    x_min = -3.0
    x_max = 3.0

    agent = PPOAgent(state_dim, action_dim, lr, gamma, lambda_gae, epsilon_clip, K_epochs, hidden_dim)

    plot_interval = 1000

    # --- 시각화 설정 ---
    plt.ion()
    # fig, ax = plt.subplots(figsize=(10, 5)) # 단일 subplot 생성, 가로 길이를 조금 늘림
    fig, ax = plt.subplots(figsize=(10, 5)) # Use a single subplot

    x_test = np.linspace(x_min, x_max, 300)
    y_true_test = target_function(x_test)

    # 그래프 초기 설정
    line_target, = ax.plot(x_test, y_true_test, 'r-', label='Target Graph', linewidth=1)
    scatter_agent, = ax.plot([], [], 'b.', markersize=5, label='RL Agent Output')
    ax.set_title('RL Ideal Output with Target Graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    # Y축 범위 고정!
    ax.set_ylim(-15, 15)
    ax.legend()

    plt.tight_layout() # 레이아웃 자동 조정
    plt.show()
    # --- 시각화 설정 끝 ---

    collected_x = []
    collected_y_pred = []

    for step in range(max_steps):
        current_x = np.random.uniform(x_min, x_max)
        action_y_pred, action_log_prob = agent.select_action([current_x])
        with torch.no_grad():
            value = agent.critic(torch.FloatTensor([current_x]).unsqueeze(0)).item()

        true_y = target_function(current_x)
        reward = np.exp(-(action_y_pred - true_y)**2 / 2.0)

        agent.store_transition(current_x, action_y_pred, reward, action_log_prob, value)

        collected_x.append(current_x)
        collected_y_pred.append(action_y_pred)

        if len(agent.buffer) >= update_interval:
            agent.update()

        if (step + 1) % plot_interval == 0:
            print(f"Step: {step+1}/{max_steps}")

            # 점(scatter) 데이터 업데이트
            scatter_agent.set_data(collected_x, collected_y_pred)
            ax.set_title(f'RL Ideal Output with Target Graph (Step {step+1})')

            collected_x = []
            collected_y_pred = []

            # 그림 다시 그리기
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    plt.ioff()
    print("Training finished.")
    plt.show()