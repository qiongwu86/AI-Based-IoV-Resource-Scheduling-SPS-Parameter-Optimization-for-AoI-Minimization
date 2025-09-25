import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from optimal import AoIEnvironment
import logging
from collections import deque
import random

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 创建文件处理器
file_handler = logging.FileHandler('ddpg_log.txt')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 确保日志不被其他处理器干扰
logger.propagate = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境
env = AoIEnvironment()
constraints = env.get_constraints()

# 动作空间范围
v_min, v_max = constraints['v']
rri_min, rri_max = constraints['RRI']
rho_l_min, rho_l_max = constraints['rho_l']
action_bounds = np.array([[rri_min, rri_max], [v_min, v_max]], dtype=np.float32)
state_dim = env.get_state_dim()  # 5: [v, rho_l, RRI, p_d, PRR]
action_dim = env.get_action_dim()  # 2: [RRI, v]

# 状态归一化边界
state_bounds = np.array([
    [v_min, v_max],
    [rho_l_min, rho_l_max],
    [rri_min, rri_max],
    [0, 1],  # p_d
    [0, 1]  # PRR
], dtype=np.float32)

# Ornstein-Uhlenbeck 噪声
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, sigma_min=0.05, sigma_decay=0.9995):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def decay(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

# 神经网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.mean = nn.Linear(32, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.tanh(self.mean(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value

# DDPG代理
class DDPG:
    def __init__(self, state_dim, action_dim, action_bounds, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=1000000, batch_size=64):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.action_bounds = torch.tensor(action_bounds, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim, sigma=0.2, sigma_min=0.05, sigma_decay=0.9995)

    def select_action(self, state, episode=None):
        # State normalization
        state_np = np.array(state, dtype=np.float32)
        state_normalized = (state_np - state_bounds[:, 0]) / (state_bounds[:, 1] - state_bounds[:, 0])
        state_normalized = np.clip(state_normalized, 0, 1)
        state_tensor = torch.tensor(state_normalized, dtype=torch.float32).to(device)

        # Get deterministic action from actor
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
        self.actor.train()

        # Scale to action bounds
        action = (action + 1) / 2 * (self.action_bounds[:, 1] - self.action_bounds[:, 0]) + self.action_bounds[:, 0]
        action = action.cpu().numpy()

        # Add exploration noise
        noise = self.noise.noise()
        action = action + noise
        action = np.clip(action, self.action_bounds[:, 0].cpu().numpy(), self.action_bounds[:, 1].cpu().numpy())

        # Decay noise
        if episode is not None:
            self.noise.decay()
            logger.info(f"Action with noise (sigma={self.noise.sigma:.3f}): {action}")

        return action, 0  # DDPG is deterministic, log_prob is not used but returned for compatibility

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(-1)

        # Normalize states
        states_normalized = (states - torch.tensor(state_bounds[:, 0], dtype=torch.float32).to(device)) / (
            torch.tensor(state_bounds[:, 1] - state_bounds[:, 0], dtype=torch.float32).to(device))
        next_states_normalized = (next_states - torch.tensor(state_bounds[:, 0], dtype=torch.float32).to(device)) / (
            torch.tensor(state_bounds[:, 1] - state_bounds[:, 0], dtype=torch.float32).to(device))

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states_normalized)
            next_actions = (next_actions + 1) / 2 * (self.action_bounds[:, 1] - self.action_bounds[:, 0]) + self.action_bounds[:, 0]
            target_Q = self.critic_target(next_states_normalized, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(states_normalized, actions)
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        predicted_actions = self.actor(states_normalized)
        predicted_actions = (predicted_actions + 1) / 2 * (self.action_bounds[:, 1] - self.action_bounds[:, 0]) + self.action_bounds[:, 0]
        actor_loss = -self.critic(states_normalized, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

# 训练函数
def train_ddpg(episodes=150, max_steps=100):
    ddpg = DDPG(state_dim, action_dim, action_bounds)
    episode_rewards = []
    min_aoi_history = []
    p_d_history = []
    prr_history = []
    actor_loss_history = []
    critic_loss_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_aoi = []  # List to store AoI values for the episode
        states, actions, rewards, dones = [], [], [], []

        for step in range(max_steps):
            action, _ = ddpg.select_action(state, episode=episode)  # Pass episode for noise decay
            rri, v = action
            rho_l = 6000 / v
            rho_l = np.clip(rho_l, rho_l_min, rho_l_max)
            v = 6000 / rho_l  # Ensure v * rho_l = 6000

            # Environment interaction
            result = env.step(rho_l, rri)
            aoi = result['AoI']
            
            # 屏蔽异常值：AoI < 20 或 AoI > 10000，设为无效
            if aoi < 20 or aoi > 10000:
                result['valid'] = False
                aoi = float('inf')  # 设为无穷大以便后续处理
            
            # New reward calculation with condition
            if result['valid']:
                if aoi < 70:
                    reward = (-70 / 10 + 10) + (70 - aoi)  # = 1.7 + (63 - aoi)
                elif aoi < 55:
                    reward = (-55 / 10 + 10) + 15 + 3 * (55 - aoi)  # = 1.7 + (63 - aoi)
                else:
                    reward = -aoi / 10 + 10
            else:
                reward = -30
            next_state = np.array([v, rho_l, rri, result['p_d'], result['PRR']], dtype=np.float32)
            done = False

            # Log step information
            logger.info(f"Episode {episode}, Step {step + 1}: "
                        f"Action [RRI: {rri:.2f}, v: {v:.2f}], "
                        f"State [v: {state[0]:.2f}, rho_l: {state[1]:.2f}, RRI: {state[2]:.2f}, "
                        f"p_d: {state[3]:.4f}, PRR: {state[4]:.4f}], "
                        f"Reward: {reward:.4f}, AoI: {aoi:.2f} ms, "
                        f"T_q: {result['T_q']:.2f} ms, T_t: {result['T_t']:.2f} ms, "
                        f"p_d: {result['p_d']:.4f}, PRR: {result['PRR']:.4f}")

            # Store experience - 只有有效的经验才放入replay buffer
            if result['valid']:
                ddpg.replay_buffer.push(state, action, reward, next_state, done)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                episode_aoi.append(aoi)  # Store AoI for this step

            # Update networks
            if len(ddpg.replay_buffer) >= ddpg.batch_size:
                actor_loss, critic_loss = ddpg.update()
                if actor_loss is not None and critic_loss is not None:
                    actor_loss_history.append(actor_loss)
                    critic_loss_history.append(critic_loss)
                    logger.info(f"Episode {episode}, Step {step + 1}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            state = next_state
            episode_reward += reward

            if done or not result['valid']:
                break

        ddpg.noise.reset()  # Reset noise at the end of each episode
        episode_rewards.append(episode_reward / (step + 1))
        # 只统计有效AoI值（在合理范围内）
        valid_aoi = [aoi for aoi in episode_aoi if 20 <= aoi <= 10000]
        min_aoi_history.append(np.mean(valid_aoi) if valid_aoi else float('inf'))
        p_d_history.append(result['p_d'])
        prr_history.append(result['PRR'])

        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}, Avg Reward: {episode_rewards[-1]:.2f}, Min AoI: {min_aoi_history[-1]:.2f} ms, "
                f"p_d: {p_d_history[-1]:.4f}, PRR: {prr_history[-1]:.4f}, Noise Sigma: {ddpg.noise.sigma:.3f}")

    return episode_rewards, min_aoi_history, p_d_history, prr_history, actor_loss_history, critic_loss_history

# 运行训练
episode_rewards, min_aoi_history, p_d_history, prr_history, actor_loss_history, critic_loss_history = train_ddpg()

# 保存统计数据
with open("ddpg_stats.txt", "w") as f:
    f.write("Episode,Min_AoI,p_d,PRR,Actor_Loss,Critic_Loss\n")
    for ep in range(len(min_aoi_history)):
        actor_loss = actor_loss_history[ep] if ep < len(actor_loss_history) else 0
        critic_loss = critic_loss_history[ep] if ep < len(critic_loss_history) else 0
        f.write(f"{ep},{min_aoi_history[ep]:.4f},{p_d_history[ep]:.4f},{prr_history[ep]:.4f},{actor_loss:.4f},{critic_loss:.4f}\n")

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(min_aoi_history, label='Minimum AoI', color='blue')
plt.xlabel('Episode')
plt.ylabel('AoI (ms)')
plt.title('DDPG: Minimum AoI Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ddpg_min_aoi_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label='Average Reward', color='green')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('DDPG: Average Reward Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ddpg_avg_reward_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(p_d_history, label='Packet Drop Probability (p_d)', color='red')
plt.xlabel('Episode')
plt.ylabel('p_d')
plt.title('DDPG: Packet Drop Probability Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ddpg_p_d_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(prr_history, label='Packet Reception Ratio (PRR)', color='purple')
plt.xlabel('Episode')
plt.ylabel('PRR')
plt.title('DDPG: PRR Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ddpg_prr_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(actor_loss_history, label='Actor Loss', color='orange')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('DDPG: Actor Loss Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ddpg_actor_loss_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(critic_loss_history, label='Critic Loss', color='brown')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('DDPG: Critic Loss Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ddpg_critic_loss_evolution.png')
plt.show()
