# src/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import earth2studio.data as data
from datetime import datetime, timedelta
from earth2studio.models.auto import Package
from sklearn.metrics import mean_squared_error
from src.config import *
import importlib
from earth2studio.data import DataArrayFile
import xarray as xr
import earth2studio.run as run
from earth2studio.io import ZarrBackend

class WeatherEnv(gym.Env):
    """一个为RL智能体设计的自定义天气预报环境"""
    def __init__(self, start_date=datetime(2020, 1, 1), end_date=datetime(2020, 1, 2)):
        super(WeatherEnv, self).__init__()
        
        # 加载数据源和基础模型
    
        self.data_source = data.WB2ERA5(cache=True)
        self.base_models = {}
        for model_name in BASE_MODELS:
            try:
                module = importlib.import_module("earth2studio.models.px")
                model_class = getattr(module, model_name, None)
                if model_class is None:
                    print(f"Model {model_name} not found in earth2studio.models.px")
                    continue
                #本地加载
                #pkg = Package(MODEL_PATH[model_name], cache=False)
                
                #model_instance = model_class.load_model(pkg).to(DEVICE)

                # 云端加载，会存到~/.cache/earth2studio文件夹
                model_instance = model_class.load_model(model_class.load_default_package()).to(DEVICE)
                self.base_models[model_name] = model_instance
                print(f"Loaded model {model_name} to {DEVICE}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
        
        print(f"Loaded base models: {list(self.base_models.keys())}")
        
        self.lead_time = timedelta(hours=FORECAST_LEAD_TIME_HOURS)
        self.start_date = start_date
        self.end_date = end_date # 确保有足够的数据
        
        self.current_time = self.start_date
        
        # 定义动作和观察空间
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
        """执行一个时间步"""
   
        forecasts = {}
        individual_mses = np.zeros(NUM_BASE_MODELS, dtype=np.float32)
        
        ground_truth_time = self.current_time + self.lead_time
        ground_truth_data, _ = self.get_cur_weather(ground_truth_time)
        ground_truth_data = ground_truth_data.cpu().numpy()
 
        with torch.no_grad():
            for i, (name, model) in enumerate(self.base_models.items()):
                io = ZarrBackend()
                io = run.deterministic([self.current_time], PREDICT_TIME, model, self.data_source, io)
                arrays = [io[VARIABLES[i]] for i in range(len(VARIABLES))]
                forecast = np.stack(arrays, axis=2) #io[VARIABLES][0, PREDICT_TIME]
     
                forecasts[name] = forecast
                
                # 计算每个模型的MSE
                mse = mean_squared_error(ground_truth_data.flatten(), forecast.flatten())
                rmse = np.sqrt(mse)
                individual_mses[i] = rmse

        # 2. 计算加权集成预报
        ensemble_forecast = np.zeros_like(ground_truth_data)
        for i, name in enumerate(self.base_models.keys()):
            ensemble_forecast += action[i] * forecasts[name]
        
        # 3. 计算集成预报的MSE和奖励
        ensemble_mse = mean_squared_error(ground_truth_data.flatten(), ensemble_forecast.flatten())
        ensemble_rmse = np.sqrt(ensemble_mse)
        reward = -ensemble_rmse
      
        self.current_time += self.lead_time
        self.step_count += 1
        
        next_observation = self._get_observation(self.current_time, individual_mses)
        
        terminated = self.step_count >= MAX_STEPS_PER_EPISODE
        truncated = False 
        
        info = {"ensemble_mse": ensemble_mse, "individual_mses": individual_mses}
        
        return next_observation, reward, terminated, truncated, info

    def close(self):
        print("Closing WeatherEnv.")