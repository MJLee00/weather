# src/training/train.py
import numpy as np
from collections import deque
from src.environment import WeatherEnv
from src.agent import DDPGAgent
from src.config import *
import os

def train():
    """主训练函数"""
    print(f"Starting training on device: {DEVICE}")

    # 创建环境和智能体
    env = WeatherEnv()
    agent = DDPGAgent()

    scores = []
    scores_window = deque(maxlen=100) # 用于计算最近100个回合的平均分

    for i_episode in range(1, NUM_TRAINING_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            # 智能体选择动作
            action = agent.select_action(state)
            
            # 环境执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 智能体存储经验并学习
            agent.step(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        scores_window.append(episode_reward)
        scores.append(episode_reward)
        
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\t"
              f"Ensemble MSE: {info.get('ensemble_mse', 0):.4f}", end="")
        
        if i_episode % 100 == 0:
            agent.update_model()
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
            # 保存模型
            if not os.path.exists("./models"):
                os.makedirs("./models")
            agent.save()
            
    env.close()
    print("\nTraining complete.")
    return scores

if __name__ == '__main__':
    train()