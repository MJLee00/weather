# src/training/train.py
import numpy as np
from collections import deque
from src.environment import WeatherEnv
from src.agent import DDPGAgent
from src.config import *
import os, re, glob, importlib

def _find_latest_episode(models_dir: str, prefix="actor_", suffix=".pth"):
    """扫描 models_dir 下的 actor_{ep}.pth，返回(max_ep 或 None)"""
    if not os.path.isdir(models_dir):
        return None
    pattern = os.path.join(models_dir, f"{prefix}*{suffix}")
    files = glob.glob(pattern)
    max_ep = None
    for f in files:
        # 匹配 actor_数字.pth
        m = re.search(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}$", os.path.basename(f))
        if m:
            ep = int(m.group(1))
            if max_ep is None or ep > max_ep:
                max_ep = ep
    return max_ep

def train():
    """主训练函数（修正：从 ./models 目录按最大 episode 续训）"""
    print(f"Starting training on device: {DEVICE}")

    # 创建环境和智能体
    env = WeatherEnv()
    agent = DDPGAgent()

    # ===== 断点续训（仅使用目录 + 设置 agent 模块里的 ROUND）=====
    models_dir = "./models"
    start_episode = 1
    latest_ep = _find_latest_episode(models_dir, prefix="actor_", suffix=".pth")
    if latest_ep is not None:
        try:
            # 只传目录，交由 load 内部按 ROUND 读取 actor_{ROUND}.pth / critic_{ROUND}.pth
            agent.load(directory=models_dir, ROUND=latest_ep)
            start_episode = latest_ep + 1
            print(f"[Resume] Loaded from '{models_dir}' with ROUND={latest_ep}. Resume at episode {start_episode}.")
        except Exception as e:
            print(f"[Resume][Warn] Found checkpoints but failed to load: {e}")
            print("[Resume][Hint] Ensure agent.load(directory) uses global ROUND and files exist: "
                  f"actor_{latest_ep}.pth / critic_{latest_ep}.pth")

    scores, scores_window = [], deque(maxlen=100)

    for i_episode in range(start_episode, NUM_TRAINING_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, terminated or truncated)
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break

        scores_window.append(episode_reward)
        scores.append(episode_reward)

        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\t"
              f"Ensemble MSE: {info.get('ensemble_mse', 0):.4f}", end="")

        if i_episode % UPDATE_INTERVAL == 0:
            for _ in range(UPDATE_EPOCHS):
                agent.update_model()
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        # 确保目录存在后保存
        os.makedirs(models_dir, exist_ok=True)
        agent.save(models_dir, i_episode)  # 期望写出 actor_{i_episode}.pth / critic_{i_episode}.pth

    env.close()
    print("\nTraining complete.")
    return scores
    
if __name__ == '__main__':
    train()