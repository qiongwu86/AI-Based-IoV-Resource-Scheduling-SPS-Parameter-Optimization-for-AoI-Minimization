import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from optimal import AoIEnvironment
import logging

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
file_handler = logging.FileHandler('ppo_log.txt')
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


# 神经网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)  # 增强初始探索
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.tanh(self.mean(x))
        log_std = self.log_std.clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value


# PPO代理
class PPO:
    def __init__(self, state_dim, action_dim, action_bounds, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, clip_ratio=0.2,
                 gae_lambda=0.95, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.action_bounds = torch.tensor(action_bounds, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon_start  # Initial epsilon value
        self.epsilon_min = epsilon_min  # Minimum epsilon value
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon

    def select_action(self, state, episode=None):
        # State normalization
        state_np = np.array(state, dtype=np.float32)
        state_normalized = (state_np - state_bounds[:, 0]) / (state_bounds[:, 1] - state_bounds[:, 0])
        state_normalized = np.clip(state_normalized, 0, 1)  # Prevent numerical overflow
        state_tensor = torch.tensor(state_normalized, dtype=torch.float32).to(device)

        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon and episode is not None:
            # Select random action within bounds
            action = np.random.uniform(self.action_bounds[:, 0].cpu().numpy(),
                                       self.action_bounds[:, 1].cpu().numpy())
            action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
            # Compute log probability for the random action (approximate for compatibility)
            mean, std = self.actor(state_tensor)
            norm_action = (action_tensor - self.action_bounds[:, 0]) / (
                    self.action_bounds[:, 1] - self.action_bounds[:, 0])
            norm_action = 2 * norm_action - 1
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(torch.atanh(norm_action.clamp(-0.999, 0.999)))
            log_prob = log_prob.sum(dim=-1, keepdim=True) - torch.log(1 - norm_action.pow(2) + 1e-6).sum(dim=-1,
                                                                                                         keepdim=True)
            log_prob = log_prob.item()  # Convert to scalar
            logger.info(f"Random action selected (epsilon={self.epsilon:.3f}): {action}")
        else:
            # Use actor's policy
            action_tensor, log_prob_tensor = self.actor.sample(state_tensor)
            # Scale to action bounds
            action = (action_tensor + 1) / 2 * (
                        self.action_bounds[:, 1] - self.action_bounds[:, 0]) + self.action_bounds[:, 0]
            action = action.detach().cpu().numpy()
            log_prob = log_prob_tensor.item()  # Convert to scalar

        # Decay epsilon
        if episode is not None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action, log_prob

    def compute_gae(self, rewards, values, next_value, dones):
        # [Unchanged, same as your original code]
        advantages = []
        returns = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, states, actions, log_probs, returns, advantages, epochs=50, batch_size=64):
        # [Unchanged, same as your original code]
        dataset_size = len(states)
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states_np = states[batch_indices]
                batch_states_normalized = (batch_states_np - state_bounds[:, 0]) / (
                            state_bounds[:, 1] - state_bounds[:, 0])
                batch_states_normalized = np.clip(batch_states_normalized, 0, 1)
                batch_states = torch.tensor(batch_states_normalized, dtype=torch.float32).to(device)
                batch_actions = torch.tensor(actions[batch_indices], dtype=torch.float32).to(device)
                batch_log_probs = torch.tensor(log_probs[batch_indices], dtype=torch.float32).to(device)
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Actor update
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                norm_actions = (batch_actions - self.action_bounds[:, 0]) / (
                            self.action_bounds[:, 1] - self.action_bounds[:, 0])
                norm_actions = 2 * norm_actions - 1
                new_log_probs = dist.log_prob(torch.atanh(norm_actions.clamp(-0.999, 0.999)))
                new_log_probs = new_log_probs.sum(dim=-1, keepdim=True) - torch.log(1 - norm_actions.pow(2) + 1e-6).sum(
                    dim=-1, keepdim=True)
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(batch_states).squeeze()
                critic_loss = torch.nn.functional.mse_loss(values, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()



# 训练函数
def train_ppo(episodes=3000, max_steps=50):
    ppo = PPO(state_dim, action_dim, action_bounds, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9995)
    episode_rewards = []
    min_aoi_history = []
    p_d_history = []
    prr_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        for step in range(max_steps):
            action, log_prob = ppo.select_action(state, episode=episode)  # Pass episode number
            rri, v = action
            rho_l = 3600 / v
            rho_l = np.clip(rho_l, rho_l_min, rho_l_max)
            v = 3600 / rho_l  # Ensure v * rho_l = 3600

            # Environment interaction
            result = env.step(rho_l, rri)
            aoi = result['AoI']
            reward = -aoi / 10 + 8 if result['valid'] else -10  # Reward scaling
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

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(ppo.critic(torch.tensor(state, dtype=torch.float32).to(device)).item())
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done or not result['valid']:
                break

        next_value = ppo.critic(torch.tensor(state, dtype=torch.float32).to(device)).item()
        advantages, returns = ppo.compute_gae(rewards, values, next_value, dones)
        ppo.update(np.array(states), np.array(actions), np.array(log_probs), returns, advantages)

        episode_rewards.append(episode_reward / (step + 1))
        min_aoi_history.append((-min(rewards)+8) * 10 if rewards else float('inf'))  # Reverse scaling
        p_d_history.append(result['p_d'])
        prr_history.append(result['PRR'])

        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}, Avg Reward: {episode_rewards[-1]:.2f}, Min AoI: {min_aoi_history[-1]:.2f} ms, "
                f"p_d: {p_d_history[-1]:.4f}, PRR: {prr_history[-1]:.4f}, Epsilon: {ppo.epsilon:.3f}")

    return episode_rewards, min_aoi_history, p_d_history, prr_history



# 运行训练
episode_rewards, min_aoi_history, p_d_history, prr_history = train_ppo()

# 保存统计数据
with open("ppo_stats.txt", "w") as f:
    f.write("Episode,Min_AoI,p_d,PRR\n")
    for ep, (min_aoi, p_d, prr) in enumerate(zip(min_aoi_history, p_d_history, prr_history)):
        f.write(f"{ep},{min_aoi:.4f},{p_d:.4f},{prr:.4f}\n")

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(min_aoi_history, label='Minimum AoI', color='blue')
plt.xlabel('Episode')
plt.ylabel('AoI (ms)')
plt.title('PPO: Minimum AoI Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ppo_min_aoi_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label='Average Reward', color='green')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('PPO: Average Reward Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ppo_avg_reward_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(p_d_history, label='Packet Drop Probability (p_d)', color='red')
plt.xlabel('Episode')
plt.ylabel('p_d')
plt.title('PPO: Packet Drop Probability Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ppo_p_d_evolution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(prr_history, label='Packet Reception Ratio (PRR)', color='purple')
plt.xlabel('Episode')
plt.ylabel('PRR')
plt.title('PPO: PRR Evolution')
plt.legend()
plt.grid(True)
plt.savefig('ppo_prr_evolution.png')
plt.show()
