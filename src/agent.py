# src/agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.networks import Actor, Critic
from src.replay_buffer import ReplayBuffer
from src.config import *

class DDPGAgent:
    """DDPG智能体，与环境交互并从经验中学习"""
    def __init__(self):
        # 初始化Actor和Critic网络及其目标网络
        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def select_action(self, state, add_noise=True):
        """根据当前策略选择动作"""
        weather_data, model_mses = state['weather_data'], state['model_mses']
        weather_data = weather_data.unsqueeze(0).to(DEVICE)
        model_mses = torch.from_numpy(model_mses).float().unsqueeze(0).to(DEVICE)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(weather_data, model_mses).cpu().data.numpy().flatten()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, EXPLORATION_NOISE, size=NUM_BASE_MODELS)
            action = (action + noise).clip(0, 1)
            # 重新归一化以确保和为1
            action /= np.sum(action)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """将经验存入回放池，并在需要时进行学习"""
        weather_data, model_mses = state['weather_data'], state['model_mses']
        weather_data = weather_data.unsqueeze(0).to(DEVICE)
        model_mses = torch.from_numpy(model_mses).float().unsqueeze(0).to(DEVICE)

        next_weather_data, next_model_mses = next_state['weather_data'], next_state['model_mses']
        next_weather_data = next_weather_data.unsqueeze(0).to(DEVICE)
        next_model_mses = torch.from_numpy(next_model_mses).float().unsqueeze(0).to(DEVICE)

        self.replay_buffer.add(weather_data, model_mses, action, reward, next_weather_data, next_model_mses, done)
        
        if len(self.replay_buffer) > BATCH_SIZE:
            self.update_model()

    def update_model(self):
        """从回放池中采样数据并更新网络"""
        weather, mses, actions, rewards, next_weather, next_mses, dones = self.replay_buffer.sample()

        # --- 更新Critic ---
        with torch.no_grad():
            
            next_actions = self.actor_target(next_weather, next_mses)
            target_Q = self.critic_target(next_weather, next_mses, next_actions)
            target_Q = rewards + (GAMMA * target_Q * (1 - dones))

        current_Q = self.critic(weather, mses, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 更新Actor ---
        actor_loss = -self.critic(weather, mses, self.actor(weather, mses)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 软更新目标网络 ---
        self._soft_update(self.critic, self.critic_target, TAU)
        self._soft_update(self.actor, self.actor_target, TAU)

    def _soft_update(self, local_model, target_model, tau):
        """软更新模型参数: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, directory="./models", ROUND=0):
        torch.save(self.actor.state_dict(), f'{directory}/actor_{str(ROUND)}.pth')
        torch.save(self.critic.state_dict(), f'{directory}/critic_{str(ROUND)}.pth')

    def load(self, directory="./models", ROUND=0):
        self.actor.load_state_dict(torch.load(f'{directory}/actor_{str(ROUND)}.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/critic_{str(ROUND)}.pth'))