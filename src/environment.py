# src/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import earth2studio.data as data
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import importlib
import earth2studio.run as run
from earth2studio.io import ZarrBackend

# 假设 src.config 是一个配置文件，其中定义了所有大写常量
try:
    from src.config import (
        DEVICE, VARIABLES, BASE_MODELS, NUM_BASE_MODELS, 
        FORECAST_LEAD_TIME_HOURS, PREDICT_TIME, MAX_STEPS_PER_EPISODE
    )
except ImportError:
    # 临时占位符，以便代码可以在没有外部 config 文件时运行，但生产环境必须定义这些常量
    print("WARNING: Using placeholder configuration. Please ensure src/config.py exists.")
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    VARIABLES = ['t2m', 'z500'] 
    BASE_MODELS = ['FCN', 'Pangu'] # 示例模型
    NUM_BASE_MODELS = len(BASE_MODELS)
    FORECAST_LEAD_TIME_HOURS = 6
    PREDICT_TIME = 8 # 48小时
    MAX_STEPS_PER_EPISODE = 100 
    
# =========================================================================
# 辅助函数: 安全加载模型到指定设备
# =========================================================================

def safe_load_model(model_class, target_gpu_index):
    """
    尝试将模型加载到指定的 GPU 索引上。

    参数:
        model_class: 要加载的模型类。
        target_gpu_index (int): 目标 GPU 的编号 (例如 0, 1, 2, ...)。

    返回:
        model_instance: 成功加载的模型实例。
    
    抛出:
        RuntimeError: 如果加载失败。
    """
    num_gpus = torch.cuda.device_count()
    
    if torch.cuda.is_available() and target_gpu_index < num_gpus:
        # 目标是 GPU
        device = torch.device(f'cuda:{target_gpu_index}')
    else:
        # 目标 GPU 不可用或 CUDA 不可用，回退到 CPU
        device = torch.device('cpu')
        
        if torch.cuda.is_available() and target_gpu_index >= num_gpus:
            print(f"⚠️ Target GPU index {target_gpu_index} out of range (max {num_gpus-1}). Loading model to CPU.")
        elif not torch.cuda.is_available():
            print("⚠️ CUDA not available. Loading model to CPU.")
    
    print(f"Attempting to load model to {device}...")
    try:
        # 使用 earth2studio 的默认包加载模型，并将其移动到目标设备
        model_package = model_class.load_default_package()
        model_instance = model_class.load_model(model_package).to(device)
        print(f"✅ Successfully loaded model to {device}")
        return model_instance
    except Exception as e:
        error_msg = f"❌ Failed to load model on {device}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

# =========================================================================
# 强化学习环境
# =========================================================================

class WeatherEnv(gym.Env):
    """一个为RL智能体设计的自定义天气预报环境"""
    
    # 元数据，与 gymnasium.Env 兼容
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, start_date=datetime(2020, 1, 1), end_date=datetime(2020, 6, 1)):
        super(WeatherEnv, self).__init__()
        
        # 加载数据源和基础模型
        self.data_source = data.WB2ERA5(cache=True)
        self.base_models = {}
        
        # 获取可用 GPU 数量，并设置模型分配计数器
        num_gpus = torch.cuda.device_count()
        current_gpu_index = 0
        
        # 迭代加载模型，并使用轮询策略分配 GPU
        for model_name in BASE_MODELS:
            try:
                module = importlib.import_module("earth2studio.models.px")
                model_class = getattr(module, model_name, None)
                if model_class is None:
                    print(f"Model {model_name} not found in earth2studio.models.px")
                    continue
                
                # --- 设备分配逻辑 ---
                if num_gpus > 0:
                    # 分配给当前的 GPU 索引，并轮询到下一个
                    target_index = current_gpu_index
                    model_instance = safe_load_model(model_class, target_index)
                    current_gpu_index = (current_gpu_index + 1) % num_gpus
                else:
                    # 如果没有 GPU，加载到 CPU (target_gpu_index 0 会被 safe_load_model 视为 CPU)
                    model_instance = safe_load_model(model_class, 0) 
                
                self.base_models[model_name] = model_instance
                
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}. Skipping.")
        
        print(f"Loaded base models: {list(self.base_models.keys())}")
        
        self.lead_time = timedelta(hours=FORECAST_LEAD_TIME_HOURS)
        self.start_date = start_date
        self.end_date = end_date
        self.current_time = self.start_date
        self.step_count = 0 
        
  
        # 确定 observation space 的形状
        self.action_space = spaces.Box(low=0, high=1, shape=(NUM_BASE_MODELS,), dtype=np.float32)
        data_x, data_coords = self.get_cur_weather(start_date)
        
        lat, lon = int(data_coords['lat'][0]), int(data_coords['lon'][0])
        
        self.observation_space = spaces.Dict({
            "weather_data": spaces.Box(low=-np.inf, high=np.inf, shape=(len(VARIABLES), lat, lon), dtype=np.float32),
            "model_mses": spaces.Box(low=0, high=np.inf, shape=(NUM_BASE_MODELS,), dtype=np.float32)
        })


    def get_cur_weather(self, time):
        weather, coords = data.fetch_data(
            self.data_source,
            np.array([time], dtype="datetime64[ns]"),
            VARIABLES,
            device=DEVICE,
        )
        return weather, coords
    
    def calculate_detailed_metrics(self, predicted, actual):
        """计算详细的评估指标"""
        predicted_flat = predicted.flatten()
        actual_flat = actual.flatten()
        
        # 基本指标
        mse = mean_squared_error(actual_flat, predicted_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_flat, predicted_flat)
        
        # 相关系数
        correlation, _ = pearsonr(actual_flat, predicted_flat)
        
        # 平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs((actual_flat - predicted_flat) / (actual_flat + 1e-8))) * 100
        
        # 标准化RMSE (NRMSE)
        nrmse = rmse / (np.max(actual_flat) - np.min(actual_flat))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'correlation': correlation,
            'nrmse': nrmse
        }


    def _get_observation(self, time, mses):

        weather_data, _ = self.get_cur_weather(time)
        return {"weather_data": weather_data, "model_mses": mses}

    def reset(self, seed=None, options=None):
        """重置环境到起始点"""
        super().reset(seed=seed)
        
        self.current_time = self.start_date 
        self.step_count = 0
        
        # 初始MSEs为0
        initial_mses = np.zeros(NUM_BASE_MODELS, dtype=np.float32)
        
        observation = self._get_observation(self.current_time, initial_mses)
        info = {}
        
        return observation, info

    def step(self, action):
        """执行一个时间步（并行推理版：ThreadPoolExecutor）"""
        import numpy as np
        import torch
        from concurrent.futures import ThreadPoolExecutor, as_completed

        forecasts = {}
        individual_metrics = {}
        ground_truth_data_list = []

        # ===== 1) GT 拼接：与你原逻辑一致 =====
        for i in range(PREDICT_TIME):
            ground_truth_time = self.current_time + self.lead_time * (i + 1)
            ground_truth_data, _ = self.get_cur_weather(ground_truth_time)
            ground_truth_data = ground_truth_data.cpu().numpy()
            ground_truth_data_list.append(ground_truth_data)
        ground_truth_data = np.concatenate(ground_truth_data_list, axis=1)
        ground_truth_data = ground_truth_data[:, :, :, :-1, :]

        # ===== 2) 并行推理（线程池，每个线程绑定一个GPU）=====
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available for parallel inference.")

        # 为了保持和你原代码一致的行为，individual_mses 的顺序仍然按 BASE_MODELS（若存在）
        model_items = list(self.base_models.items())
        model_order = [name for name, _ in model_items]  # 用于后续 ensemble 与 individual_mses 的顺序

        def _infer_thread(index, name, model):
            """
            单模型推理线程：绑定GPU -> to(device) -> run.deterministic -> 组装 forecast -> 计算指标 -> 挪回CPU
            返回: (name, forecast, metrics)
            """
            device_id = index % num_gpus
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")

            print(f'current time is {self.current_time}, model index is {index}, inference model name is {name}')
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                io = ZarrBackend()
                _io = run.deterministic([self.current_time], PREDICT_TIME, model, self.data_source, io, device=device)

                arrays = [_io[VARIABLES[i]][:, 1:, :, :] for i in range(len(VARIABLES))]
                forecast = np.stack(arrays, axis=2)
                if name != 'Aurora':
                    forecast = forecast[:, :, :, :-1, :]

                # 每个模型的详细指标（线程内直接算，减少主线程遍历）
                metrics = self.calculate_detailed_metrics(forecast, ground_truth_data)

            # 释放/回收
            model = model.to('cpu')
            torch.cuda.empty_cache()
            return name, forecast, metrics

        # 开线程
        max_workers = min(len(model_items), num_gpus)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_infer_thread, idx, name, model) for idx, (name, model) in enumerate(model_items)]
            for fut in as_completed(futs):
                name, forecast, metrics = fut.result()
                forecasts[name] = forecast
                individual_metrics[name] = metrics

        # ===== 3) 加权集成 =====
        ensemble_forecast = np.zeros_like(ground_truth_data)
        for i, name in enumerate(model_order):
            ensemble_forecast += action[i] * forecasts[name]

        ensemble_metrics = self.calculate_detailed_metrics(ensemble_forecast, ground_truth_data)

        # 奖励（RMSE 越小越好）
        reward = -ensemble_metrics['rmse']

        # 更新时间（一次性推进 PREDICT_TIME * lead_time）
        for _ in range(PREDICT_TIME):
            self.current_time += self.lead_time

        self.step_count += 1

        # 兼容：individual_mses 顺序遵循 BASE_MODELS（若提供），否则用当前 model_order
        order_for_mses = BASE_MODELS if 'BASE_MODELS' in globals() else model_order
        individual_mses = np.array([individual_metrics[name]['rmse'] for name in order_for_mses], dtype=np.float32)

        next_observation = self._get_observation(self.current_time, individual_mses)
        terminated = self.step_count >= MAX_STEPS_PER_EPISODE
        truncated = False

        info = {
            "ensemble_metrics": ensemble_metrics,
            "individual_metrics": individual_metrics,
            "ensemble_mse": ensemble_metrics['mse'],
            "individual_mses": individual_mses,
            "forecasts": forecasts,
            "ground_truth": ground_truth_data
        }

        return next_observation, reward, terminated, truncated, info


    def close(self):
        print("Closing WeatherEnv.")