# src/config.py
import torch

# -- Environment Configurations --
# 从 earth2studio.data 中选择变量
VARIABLES = ["t850","z500","v925","u850"]
# 基础天气模型列表 (从 earth2studio.models 加载)
BASE_MODELS = ["Pangu6", "FuXi", "FengWu"] # 您可以添加更多模型，例如 "Aurora"
NUM_BASE_MODELS = len(BASE_MODELS)

PREDICT_TIME = 4 # 此数据集是6小时一次，我们预测48小时（8个时间步）            
# 使用的数据源

FORECAST_LEAD_TIME_HOURS = 6 # 预报的未来时间

# -- RL Agent Configurations --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTOR_LR = 1e-4 # Actor学习率
CRITIC_LR = 1e-3 # Critic学习率
GAMMA = 0.99  # 折扣因子
TAU = 0.005 # 目标网络软更新系数
BATCH_SIZE = 4 # 训练批次大小
BUFFER_SIZE = int(1e5) # 经验回放池大小

# -- Training Configurations --
NUM_TRAINING_EPISODES = 1000 # 总训练回合数
MAX_STEPS_PER_EPISODE = 7 # 每个回合的最大步数 (例如，预测一周)
EXPLORATION_NOISE = 0.1 # 探索噪声的标准差
UPDATE_INTERVAL = 1

# -- Network Architecture Configurations --
# 用于处理天气数据的CNN特征提取器的参数
CNN_FEATURE_CHANNELS = 16 # CNN输出的特征通道数
LATENT_DIM = 128 # 隐藏层维度