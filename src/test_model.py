# src/test_model.py
"""
测试脚本：对比基础模型和DDPG集成模型的性能
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from src.environment import WeatherEnv
from src.agent import DDPGAgent
from src.config import *
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelTester:
    """模型测试器，用于对比基础模型和DDPG集成模型"""
    
    def __init__(self, model_path="./models"):
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

    
    def load_model(self, ROUND=366):
        """加载训练好的模型"""
        try:
            # 加载模型权重
            self.agent.load(self.model_path, ROUND=ROUND)
            print("✅ 模型加载成功!")
            
            # 将模型设置为评估模式
            self.agent.actor.eval()
            self.agent.critic.eval()
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    
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
                    avg_metrics[f'{key}_std'] = np.std(values)
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
    
    def plot_comparison(self, comparison_results, save_path=None):
        """
        绘制基础模型与DDPG集成模型的对比图表
        
        Args:
            comparison_results: compare_baseline_vs_ddpg返回的结果字典
            save_path: 保存图表的路径（可选）
        """
        baseline_avg = comparison_results['baseline_avg']
        ddpg_avg = comparison_results['ddpg_avg']
        
        if not baseline_avg or not ddpg_avg:
            print("❌ 没有可用的对比数据")
            return
        
        # 准备数据
        model_names = list(BASE_MODELS) + ['DDPG集成']
        
        # 提取各指标数据
        rmse_values = [baseline_avg[name]['rmse'] for name in BASE_MODELS] + [ddpg_avg['rmse']]
        mae_values = [baseline_avg[name]['mae'] for name in BASE_MODELS] + [ddpg_avg['mae']]
        correlation_values = [baseline_avg[name]['correlation'] for name in BASE_MODELS] + [ddpg_avg['correlation']]
        mape_values = [baseline_avg[name]['mape'] for name in BASE_MODELS] + [ddpg_avg['mape']]
        nrmse_values = [baseline_avg[name]['nrmse'] for name in BASE_MODELS] + [ddpg_avg['nrmse']]
        
        # 提取标准差（如果存在）
        rmse_stds = [baseline_avg[name].get('rmse_std', 0) for name in BASE_MODELS] + [ddpg_avg.get('rmse_std', 0)]
        mae_stds = [baseline_avg[name].get('mae_std', 0) for name in BASE_MODELS] + [ddpg_avg.get('mae_std', 0)]
        corr_stds = [baseline_avg[name].get('correlation_std', 0) for name in BASE_MODELS] + [ddpg_avg.get('correlation_std', 0)]
        mape_stds = [baseline_avg[name].get('mape_std', 0) for name in BASE_MODELS] + [ddpg_avg.get('mape_std', 0)]
        nrmse_stds = [baseline_avg[name].get('nrmse_std', 0) for name in BASE_MODELS] + [ddpg_avg.get('nrmse_std', 0)]
        
        # 设置颜色
        colors = ['#4A90E2', '#50C878', '#FF6B6B', '#FFA500', '#DC143C']  # 蓝色、绿色、红色、橙色、深红色
        
        # 创建图表
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. RMSE对比
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(model_names, rmse_values, alpha=0.8, color=colors[:len(model_names)])
        ax1.errorbar(model_names, rmse_values, yerr=rmse_stds, fmt='none', color='black', capsize=5, capthick=1.5)
        ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax1.set_title('RMSE对比', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. MAE对比
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(model_names, mae_values, alpha=0.8, color=colors[:len(model_names)])
        ax2.errorbar(model_names, mae_values, yerr=mae_stds, fmt='none', color='black', capsize=5, capthick=1.5)
        ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax2.set_title('MAE对比', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars2, mae_values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. 相关系数对比
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(model_names, correlation_values, alpha=0.8, color=colors[:len(model_names)])
        ax3.errorbar(model_names, correlation_values, yerr=corr_stds, fmt='none', color='black', capsize=5, capthick=1.5)
        ax3.set_ylabel('相关系数', fontsize=12, fontweight='bold')
        ax3.set_title('相关系数对比', fontsize=13, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars3, correlation_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. MAPE对比
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = ax4.bar(model_names, mape_values, alpha=0.8, color=colors[:len(model_names)])
        ax4.errorbar(model_names, mape_values, yerr=mape_stds, fmt='none', color='black', capsize=5, capthick=1.5)
        ax4.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax4.set_title('MAPE对比', fontsize=13, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars4, mape_values)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 5. NRMSE对比
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(model_names, nrmse_values, alpha=0.8, color=colors[:len(model_names)])
        ax5.errorbar(model_names, nrmse_values, yerr=nrmse_stds, fmt='none', color='black', capsize=5, capthick=1.5)
        ax5.set_ylabel('NRMSE', fontsize=12, fontweight='bold')
        ax5.set_title('NRMSE对比', fontsize=13, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars5, nrmse_values)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 6. 多指标综合对比（雷达图）
        ax6 = fig.add_subplot(gs[1, 2], projection='polar')
        
        # 准备雷达图数据（需要归一化）
        metrics = ['RMSE', 'MAE', '相关系数', 'MAPE', 'NRMSE']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 归一化数据（越小越好的指标取倒数并归一化，越大越好的指标直接归一化）
        # 找到最佳基础模型
        best_model_name = min(baseline_avg.keys(), key=lambda x: baseline_avg[x]['rmse'])
        best_rmse = baseline_avg[best_model_name]['rmse']
        best_mae = baseline_avg[best_model_name]['mae']
        best_corr = baseline_avg[best_model_name]['correlation']
        best_mape = baseline_avg[best_model_name]['mape']
        best_nrmse = baseline_avg[best_model_name]['nrmse']
        
        # DDPG指标
        ddpg_rmse_norm = best_rmse / ddpg_avg['rmse']  # 越小越好，取比值
        ddpg_mae_norm = best_mae / ddpg_avg['mae']
        ddpg_corr_norm = ddpg_avg['correlation'] / best_corr  # 越大越好，取比值
        ddpg_mape_norm = best_mape / ddpg_avg['mape']
        ddpg_nrmse_norm = best_nrmse / ddpg_avg['nrmse']
        
        # 最佳基础模型指标（归一化为1）
        best_metrics = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # DDPG指标
        ddpg_metrics = [ddpg_rmse_norm, ddpg_mae_norm, ddpg_corr_norm, ddpg_mape_norm, ddpg_nrmse_norm]
        
        # 闭合图形
        best_metrics += best_metrics[:1]
        ddpg_metrics += ddpg_metrics[:1]
        
        ax6.plot(angles, best_metrics, 'o-', linewidth=2, label=f'最佳基础模型({best_model_name})', color='#4A90E2')
        ax6.fill(angles, best_metrics, alpha=0.25, color='#4A90E2')
        ax6.plot(angles, ddpg_metrics, 'o-', linewidth=2, label='DDPG集成', color='#DC143C')
        ax6.fill(angles, ddpg_metrics, alpha=0.25, color='#DC143C')
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics)
        ax6.set_title('多指标综合对比（归一化）', fontsize=13, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax6.grid(True)
        
        # 7. 性能提升百分比
        ax7 = fig.add_subplot(gs[2, :])
        
        # 计算相对于最佳基础模型的提升
        best_rmse_val = baseline_avg[best_model_name]['rmse']
        best_mae_val = baseline_avg[best_model_name]['mae']
        best_corr_val = baseline_avg[best_model_name]['correlation']
        
        improvements = [
            (best_rmse_val - ddpg_avg['rmse']) / best_rmse_val * 100,  # RMSE提升
            (best_mae_val - ddpg_avg['mae']) / best_mae_val * 100,      # MAE提升
            (ddpg_avg['correlation'] - best_corr_val) / best_corr_val * 100,  # 相关系数提升
        ]
        improvement_labels = ['RMSE提升', 'MAE提升', '相关系数提升']
        
        bars7 = ax7.bar(improvement_labels, improvements, alpha=0.8, 
                       color=['#28a745' if x > 0 else '#dc3545' for x in improvements])
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax7.set_ylabel('性能提升 (%)', fontsize=12, fontweight='bold')
        ax7.set_title('DDPG集成模型相对最佳基础模型的性能提升', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar, val in zip(bars7, improvements):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{val:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=11, fontweight='bold')
        
        plt.suptitle('基础模型 vs DDPG集成模型性能对比分析', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='对比基础模型和DDPG集成模型的性能')
    parser.add_argument('--model_path', type=str, default='./models',
                       help='模型文件路径 (默认: ./models)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='对比分析的回合数 (默认: 5)')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='保存图表的路径 (可选)')
    parser.add_argument('--round', type=int, default=366,
                       help='模型轮次 (默认: 366)')
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = ModelTester(model_path=args.model_path)
        
        # 加载模型
        tester.load_model(ROUND=args.round)
        
        # 运行对比分析
        comparison_results = tester.compare_baseline_vs_ddpg(num_episodes=args.episodes)
        
        # 绘制对比图表
        tester.plot_comparison(comparison_results, save_path=args.save_plot)
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ 测试完成!")
    return 0


if __name__ == "__main__":
    exit(main())


