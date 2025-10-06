# src/test_model.py
"""
测试脚本：加载已经训练好的DDPG模型并评估其性能
"""
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os
from src.environment import WeatherEnv
from src.agent import DDPGAgent
from src.config import *
import argparse

class ModelTester:
    """模型测试器，用于加载和评估训练好的DDPG模型"""
    
    def __init__(self, model_path="./models", start_time=datetime.datetime(2020, 1, 8)):
        """
        初始化测试器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.device = DEVICE
        print(f"使用设备: {self.device}")
        
        # 创建环境和智能体
        self.env = WeatherEnv()
        self.agent = DDPGAgent()
        
        # 检查模型文件是否存在
        self.check_model_files()
        
    def check_model_files(self):
        """检查模型文件是否存在"""
        actor_path = os.path.join(self.model_path, "actor.pth")
        critic_path = os.path.join(self.model_path, "critic.pth")
        
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"找不到Actor模型文件: {actor_path}")
        if not os.path.exists(critic_path):
            raise FileNotFoundError(f"找不到Critic模型文件: {critic_path}")
            
        print(f"找到模型文件:")
        print(f"  - Actor: {actor_path}")
        print(f"  - Critic: {critic_path}")
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            # 加载模型权重
            self.agent.load(self.model_path)
            print("✅ 模型加载成功!")
            
            # 将模型设置为评估模式
            self.agent.actor.eval()
            self.agent.critic.eval()
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    
    def evaluate_model(self, num_episodes=10, render=False):
        """
        评估模型性能 - 48小时预测
        
        Args:
            num_episodes: 评估回合数
            render: 是否显示详细过程
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        print(f"\n开始评估模型，共{num_episodes}个回合，预测未来48小时...")
        
        episode_rewards = []
        episode_metrics = []
        individual_model_metrics = {name: [] for name in BASE_MODELS}
        ensemble_metrics_history = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            if render:
                print(f"\n--- 回合 {episode + 1} ---")
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # 智能体选择动作（不添加噪声）
                action = self.agent.select_action(state, add_noise=False)
                
                if render:
                    print(f"步骤 {step + 1}:")
                    print(f"  动作 (模型权重): {dict(zip(BASE_MODELS, action))}")
                
                # 环境执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # 记录详细指标
                if 'ensemble_metrics' in info:
                    ensemble_metrics_history.append(info['ensemble_metrics'])
                    
                    # 记录各基础模型指标
                    for model_name in BASE_MODELS:
                        if model_name in info['individual_metrics']:
                            individual_model_metrics[model_name].append(info['individual_metrics'][model_name])
                
                if render:
                    print(f"  奖励: {reward:.4f}")
                    if 'ensemble_metrics' in info:
                        metrics = info['ensemble_metrics']
                        print(f"  集成RMSE: {metrics['rmse']:.4f}")
                        print(f"  集成MAE: {metrics['mae']:.4f}")
                        print(f"  集成相关系数: {metrics['correlation']:.4f}")
                        print(f"  集成MAPE: {metrics['mape']:.2f}%")
                
                state = next_state
                
                if terminated or truncated:
                    break
            
            # 记录回合结果
            episode_rewards.append(episode_reward)
            
            if render:
                print(f"回合 {episode + 1} 完成:")
                print(f"  总奖励: {episode_reward:.4f}")
        
        # 计算平均性能指标
        def calculate_average_metrics(metrics_list):
            if not metrics_list:
                return {}
            
            avg_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    avg_metrics[f'avg_{key}'] = np.mean(values)
                    avg_metrics[f'std_{key}'] = np.std(values)
                    avg_metrics[f'min_{key}'] = np.min(values)
                    avg_metrics[f'max_{key}'] = np.max(values)
            return avg_metrics
        
        # 计算各基础模型平均性能
        individual_model_performance = []
        for model_name in BASE_MODELS:
            if individual_model_metrics[model_name]:
                avg_metrics = calculate_average_metrics(individual_model_metrics[model_name])
                individual_model_performance.append({
                    'model': model_name,
                    **avg_metrics
                })
        
        # 计算集成模型平均性能
        ensemble_performance = calculate_average_metrics(ensemble_metrics_history)
        ensemble_performance['avg_reward'] = np.mean(episode_rewards)
        ensemble_performance['std_reward'] = np.std(episode_rewards)
        
        results = {
            'episode_rewards': episode_rewards,
            'individual_model_performance': individual_model_performance,
            'ensemble_performance': ensemble_performance,
            'individual_model_metrics': individual_model_metrics,
            'ensemble_metrics_history': ensemble_metrics_history
        }
        
        return results
    
    def print_results(self, results):
        """打印评估结果 - 48小时预测"""
        print("\n" + "="*60)
        print("📊 48小时天气预报模型评估结果")
        print("="*60)
        
        ensemble_perf = results['ensemble_performance']
        print(f"\n🎯 DDPG集成模型性能 (48小时预测):")
        print(f"  平均奖励: {ensemble_perf['avg_reward']:.4f} ± {ensemble_perf['std_reward']:.4f}")
        print(f"  平均RMSE: {ensemble_perf['avg_rmse']:.4f} ± {ensemble_perf['std_rmse']:.4f}")
        print(f"  平均MAE:  {ensemble_perf['avg_mae']:.4f} ± {ensemble_perf['std_mae']:.4f}")
        print(f"  平均相关系数: {ensemble_perf['avg_correlation']:.4f} ± {ensemble_perf['std_correlation']:.4f}")
        print(f"  平均MAPE: {ensemble_perf['avg_mape']:.2f}% ± {ensemble_perf['std_mape']:.2f}%")
        print(f"  平均NRMSE: {ensemble_perf['avg_nrmse']:.4f} ± {ensemble_perf['std_nrmse']:.4f}")
        
        print(f"\n🔍 各基础模型性能 (48小时预测):")
        print(f"{'模型':<10} {'RMSE':<12} {'MAE':<12} {'相关系数':<10} {'MAPE(%)':<10} {'NRMSE':<10}")
        print("-" * 70)
        
        for model_perf in results['individual_model_performance']:
            print(f"{model_perf['model']:<10} "
                  f"{model_perf['avg_rmse']:.4f}±{model_perf['std_rmse']:.4f}  "
                  f"{model_perf['avg_mae']:.4f}±{model_perf['std_mae']:.4f}  "
                  f"{model_perf['avg_correlation']:.4f}±{model_perf['std_correlation']:.4f}  "
                  f"{model_perf['avg_mape']:.2f}±{model_perf['std_mape']:.2f}  "
                  f"{model_perf['avg_nrmse']:.4f}±{model_perf['std_nrmse']:.4f}")
        
        # 计算性能提升
        individual_rmse = [model['avg_rmse'] for model in results['individual_model_performance']]
        best_individual_rmse = min(individual_rmse)
        rmse_improvement = (best_individual_rmse - ensemble_perf['avg_rmse']) / best_individual_rmse * 100
        
        individual_mae = [model['avg_mae'] for model in results['individual_model_performance']]
        best_individual_mae = min(individual_mae)
        mae_improvement = (best_individual_mae - ensemble_perf['avg_mae']) / best_individual_mae * 100
        
        individual_corr = [model['avg_correlation'] for model in results['individual_model_performance']]
        best_individual_corr = max(individual_corr)
        corr_improvement = (ensemble_perf['avg_correlation'] - best_individual_corr) / best_individual_corr * 100
        
        print(f"\n📈 DDPG集成模型性能提升:")
        print(f"  RMSE提升: {rmse_improvement:.2f}% (相比最佳单一模型)")
        print(f"  MAE提升:  {mae_improvement:.2f}% (相比最佳单一模型)")
        print(f"  相关系数提升: {corr_improvement:.2f}% (相比最佳单一模型)")
        
        # 显示预测时间信息
        print(f"\n⏰ 预测时间范围: 未来48小时 (每6小时一个时间步)")
        print(f"📊 评估回合数: {len(results['episode_rewards'])}")
    
    def plot_results(self, results, save_path=None):
        """绘制评估结果图表 - 48小时预测对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 回合奖励趋势
        ax1.plot(results['episode_rewards'], 'b-', alpha=0.7, label='回合奖励')
        ax1.axhline(y=np.mean(results['episode_rewards']), color='r', linestyle='--', 
                   label=f'平均奖励: {np.mean(results["episode_rewards"]):.4f}')
        ax1.set_xlabel('回合')
        ax1.set_ylabel('奖励')
        ax1.set_title('DDPG集成模型 - 回合奖励趋势 (48小时预测)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RMSE对比 - 基础模型 vs DDPG集成
        model_names = [model['model'] for model in results['individual_model_performance']]
        model_rmse = [model['avg_rmse'] for model in results['individual_model_performance']]
        
        # 添加DDPG集成模型
        all_names = model_names + ['DDPG集成']
        all_rmse = model_rmse + [results['ensemble_performance']['avg_rmse']]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'red']
        
        bars = ax2.bar(all_names, all_rmse, alpha=0.7, color=colors)
        ax2.set_ylabel('RMSE')
        ax2.set_title('各模型RMSE对比 (48小时预测)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rmse in zip(bars, all_rmse):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 多指标对比雷达图
        metrics = ['RMSE', 'MAE', '相关系数', 'MAPE', 'NRMSE']
        
        # 归一化指标用于雷达图 (越小越好的指标取倒数)
        ensemble_metrics = [
            1/results['ensemble_performance']['avg_rmse'],  # RMSE (越小越好)
            1/results['ensemble_performance']['avg_mae'],   # MAE (越小越好)
            results['ensemble_performance']['avg_correlation'],  # 相关系数 (越大越好)
            1/results['ensemble_performance']['avg_mape'],  # MAPE (越小越好)
            1/results['ensemble_performance']['avg_nrmse']  # NRMSE (越小越好)
        ]
        
        # 计算最佳基础模型的指标
        best_model = min(results['individual_model_performance'], key=lambda x: x['avg_rmse'])
        best_metrics = [
            1/best_model['avg_rmse'],
            1/best_model['avg_mae'],
            best_model['avg_correlation'],
            1/best_model['avg_mape'],
            1/best_model['avg_nrmse']
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ensemble_metrics += ensemble_metrics[:1]
        best_metrics += best_metrics[:1]
        
        ax3.plot(angles, ensemble_metrics, 'o-', linewidth=2, label='DDPG集成', color='red')
        ax3.fill(angles, ensemble_metrics, alpha=0.25, color='red')
        ax3.plot(angles, best_metrics, 'o-', linewidth=2, label=f'最佳基础模型({best_model["model"]})', color='blue')
        ax3.fill(angles, best_metrics, alpha=0.25, color='blue')
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics)
        ax3.set_title('多指标性能对比 (归一化)')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 相关系数对比
        model_corr = [model['avg_correlation'] for model in results['individual_model_performance']]
        all_corr = model_corr + [results['ensemble_performance']['avg_correlation']]
        
        bars = ax4.bar(all_names, all_corr, alpha=0.7, color=colors)
        ax4.set_ylabel('相关系数')
        ax4.set_title('各模型相关系数对比 (48小时预测)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, corr in zip(bars, all_corr):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('48小时天气预报模型性能评估', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存到: {save_path}")
        
        plt.show()
    
    def run_single_episode_demo(self):
        """运行单个回合演示，显示详细过程 - 48小时预测"""
        print("\n🎬 开始单回合演示 - 48小时天气预报...")
        
        state, _ = self.env.reset()
        total_reward = 0
        step_details = []
        
        print(f"初始状态:")
        print(f"  天气数据形状: {state['weather_data'].shape}")
        print(f"  初始模型RMSE: {dict(zip(BASE_MODELS, state['model_mses']))}")
        print(f"  预测时间范围: 未来48小时 (每6小时一个时间步)")
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # 选择动作
            action = self.agent.select_action(state, add_noise=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            step_detail = {
                'step': step + 1,
                'action': action,
                'reward': reward,
                'ensemble_metrics': info.get('ensemble_metrics', {}),
                'individual_metrics': info.get('individual_metrics', {})
            }
            step_details.append(step_detail)
            
            total_reward += reward
            
            print(f"\n步骤 {step + 1} (预测时间: {step * 6 + 6} - {step * 6 + 48}小时):")
            print(f"  选择的动作 (模型权重):")
            for i, model_name in enumerate(BASE_MODELS):
                print(f"    {model_name}: {action[i]:.4f}")
            
            print(f"  奖励: {reward:.4f}")
            
            if 'ensemble_metrics' in info:
                metrics = info['ensemble_metrics']
                print(f"  DDPG集成模型性能:")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    MAE:  {metrics['mae']:.4f}")
                print(f"    相关系数: {metrics['correlation']:.4f}")
                print(f"    MAPE: {metrics['mape']:.2f}%")
                print(f"    NRMSE: {metrics['nrmse']:.4f}")
            
            print(f"  各基础模型性能:")
            for model_name in BASE_MODELS:
                if model_name in info.get('individual_metrics', {}):
                    model_metrics = info['individual_metrics'][model_name]
                    print(f"    {model_name}:")
                    print(f"      RMSE: {model_metrics['rmse']:.4f}")
                    print(f"      MAE:  {model_metrics['mae']:.4f}")
                    print(f"      相关系数: {model_metrics['correlation']:.4f}")
                    print(f"      MAPE: {model_metrics['mape']:.2f}%")
            
            state = next_state
            
            if terminated or truncated:
                break
        
        print(f"\n🎯 48小时预测演示完成!")
        print(f"总奖励: {total_reward:.4f}")
        print(f"平均奖励: {total_reward/(step+1):.4f}")
        
        # 计算整体性能提升
        if step_details and 'ensemble_metrics' in step_details[0]:
            ensemble_rmse = np.mean([step['ensemble_metrics']['rmse'] for step in step_details])
            individual_rmse = {}
            for model_name in BASE_MODELS:
                individual_rmse[model_name] = np.mean([
                    step['individual_metrics'][model_name]['rmse'] 
                    for step in step_details 
                    if model_name in step['individual_metrics']
                ])
            
            best_individual_rmse = min(individual_rmse.values())
            improvement = (best_individual_rmse - ensemble_rmse) / best_individual_rmse * 100
            
            print(f"\n📈 性能提升分析:")
            print(f"  最佳基础模型RMSE: {best_individual_rmse:.4f}")
            print(f"  DDPG集成模型RMSE: {ensemble_rmse:.4f}")
            print(f"  性能提升: {improvement:.2f}%")
        
        return step_details
    
    def compare_baseline_vs_ddpg(self, num_episodes=5):
        """
        对比基础模型单独预测与DDPG集成预测的性能
        
        Args:
            num_episodes: 对比回合数
            
        Returns:
            dict: 包含对比结果的字典
        """
        print(f"\n🔬 开始基础模型 vs DDPG集成模型对比分析...")
        print(f"预测时间: 48小时 | 对比回合数: {num_episodes}")
        
        baseline_results = {name: [] for name in BASE_MODELS}
        ddpg_results = []
        
        for episode in range(num_episodes):
            print(f"\n--- 对比回合 {episode + 1} ---")
            
            # 重置环境
            state, _ = self.env.reset()
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # DDPG选择动作
                ddpg_action = self.agent.select_action(state, add_noise=False)
                
                # 执行DDPG动作
                next_state, ddpg_reward, terminated, truncated, info = self.env.step(ddpg_action)
                
                # 记录DDPG结果
                if 'ensemble_metrics' in info:
                    ddpg_results.append(info['ensemble_metrics'])
                
                # 记录各基础模型结果
                for model_name in BASE_MODELS:
                    if model_name in info.get('individual_metrics', {}):
                        baseline_results[model_name].append(info['individual_metrics'][model_name])
                
                state = next_state
                
                if terminated or truncated:
                    break
        
        # 计算平均性能
        def calculate_avg_metrics(metrics_list):
            if not metrics_list:
                return {}
            avg_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
            return avg_metrics
        
        # 计算各模型平均性能
        baseline_avg = {}
        for model_name in BASE_MODELS:
            baseline_avg[model_name] = calculate_avg_metrics(baseline_results[model_name])
        
        ddpg_avg = calculate_avg_metrics(ddpg_results)
        
        # 打印对比结果
        print(f"\n" + "="*80)
        print("📊 基础模型 vs DDPG集成模型 - 48小时预测性能对比")
        print("="*80)
        
        print(f"\n{'模型':<12} {'RMSE':<10} {'MAE':<10} {'相关系数':<10} {'MAPE(%)':<10} {'NRMSE':<10}")
        print("-" * 80)
        
        # 显示各基础模型性能
        for model_name in BASE_MODELS:
            if baseline_avg[model_name]:
                metrics = baseline_avg[model_name]
                print(f"{model_name:<12} "
                      f"{metrics['rmse']:.4f}    "
                      f"{metrics['mae']:.4f}    "
                      f"{metrics['correlation']:.4f}    "
                      f"{metrics['mape']:.2f}    "
                      f"{metrics['nrmse']:.4f}")
        
        # 显示DDPG集成模型性能
        if ddpg_avg:
            print("-" * 80)
            print(f"{'DDPG集成':<12} "
                  f"{ddpg_avg['rmse']:.4f}    "
                  f"{ddpg_avg['mae']:.4f}    "
                  f"{ddpg_avg['correlation']:.4f}    "
                  f"{ddpg_avg['mape']:.2f}    "
                  f"{ddpg_avg['nrmse']:.4f}")
        
        # 计算性能提升
        if ddpg_avg and baseline_avg:
            print(f"\n📈 DDPG集成模型性能提升:")
            
            # 找到最佳基础模型
            best_rmse_model = min(baseline_avg.keys(), key=lambda x: baseline_avg[x]['rmse'])
            best_mae_model = min(baseline_avg.keys(), key=lambda x: baseline_avg[x]['mae'])
            best_corr_model = max(baseline_avg.keys(), key=lambda x: baseline_avg[x]['correlation'])
            
            rmse_improvement = (baseline_avg[best_rmse_model]['rmse'] - ddpg_avg['rmse']) / baseline_avg[best_rmse_model]['rmse'] * 100
            mae_improvement = (baseline_avg[best_mae_model]['mae'] - ddpg_avg['mae']) / baseline_avg[best_mae_model]['mae'] * 100
            corr_improvement = (ddpg_avg['correlation'] - baseline_avg[best_corr_model]['correlation']) / baseline_avg[best_corr_model]['correlation'] * 100
            
            print(f"  RMSE提升: {rmse_improvement:.2f}% (相比最佳基础模型 {best_rmse_model})")
            print(f"  MAE提升:  {mae_improvement:.2f}% (相比最佳基础模型 {best_mae_model})")
            print(f"  相关系数提升: {corr_improvement:.2f}% (相比最佳基础模型 {best_corr_model})")
        
        return {
            'baseline_results': baseline_results,
            'ddpg_results': ddpg_results,
            'baseline_avg': baseline_avg,
            'ddpg_avg': ddpg_avg
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试训练好的DDPG模型')
    parser.add_argument('--model_path', type=str, default='./models',
                       help='模型文件路径 (默认: ./models)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='评估回合数 (默认: 10)')
    parser.add_argument('--demo', action='store_true',
                       help='运行单回合详细演示')
    parser.add_argument('--render', action='store_true',
                       help='显示详细评估过程')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='保存图表的路径 (可选)')
    parser.add_argument('--compare', action='store_true',
                       help='运行基础模型与DDPG集成模型的对比分析')
    parser.add_argument('--compare_episodes', type=int, default=1,
                       help='对比分析的回合数 (默认: 1)')
    parser.add_argument('--datetime', type=datetime.datetime, default=datetime.datetime(2020, 1, 8),
                       help='对比分析的起始时间')
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = ModelTester(model_path=args.model_path, start_time=args.datetime)
        
        # 加载模型
        tester.load_model()
        
        if args.demo:
            # 运行演示
            tester.run_single_episode_demo()
        elif args.compare:
            # 运行对比分析
            comparison_results = tester.compare_baseline_vs_ddpg(num_episodes=args.compare_episodes)
        else:
            # 评估模型
            results = tester.evaluate_model(num_episodes=args.episodes, render=args.render)
            
            # 打印结果
            tester.print_results(results)
            
            # 绘制图表
            tester.plot_results(results, save_path=args.save_plot)
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        return 1
    
    print("\n✅ 测试完成!")
    return 0


if __name__ == "__main__":
    exit(main())


