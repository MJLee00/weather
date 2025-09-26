# src/replay_buffer.py
import numpy as np
import torch
from collections import deque
import random
from src.config import DEVICE

class ReplayBuffer:
    """用于存储和采样经验元组的固定大小的缓冲区"""
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, weather_data, model_mses, action, reward, next_weather_data, next_model_mses, done):
        """将一次经验存入内存"""
        experience = (weather_data, model_mses, action, reward, next_weather_data, next_model_mses, done)
        self.memory.append(experience)

    def sample(self):
        """从内存中随机采样一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        weather_datas, model_mses_list, actions, rewards, next_weather_datas, next_model_mses_list, dones = zip(*experiences)

        # 转换为Torch张量
        weather_datas = torch.vstack(weather_datas)
        model_mses = torch.vstack(model_mses_list)
        actions = torch.from_numpy(np.vstack(actions)).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_weather_datas = torch.vstack(next_weather_datas)
        next_model_mses = torch.vstack(next_model_mses_list)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)

        return (weather_datas, model_mses, actions, rewards, next_weather_datas, next_model_mses, dones)

    def __len__(self):
        return len(self.memory)