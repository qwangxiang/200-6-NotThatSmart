import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """计算基础评估指标"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100 if np.sum(y_true) > 0 else 0
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    @staticmethod
    def plot_comparison(y_true, y_pred, title='预测结果对比'):
        """基础预测结果对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='真实值', alpha=0.7, linewidth=2)
        plt.plot(y_pred, '--', label='预测值', linewidth=2)
        plt.title(title)
        plt.xlabel('时间步')
        plt.ylabel('负荷 (MW)')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def multi_model_comparison(models_data, true_values, 
                            model_names=None, 
                            title='多模型预测对比',
                            save_path=None):
        """多模型预测对比图（合并到同一张图）"""
        plt.figure(figsize=(14, 7))
        
        # 绘制真实值（仅画一次）
        plt.plot(true_values, label='真实值', 
                color='black', alpha=0.9, linewidth=3, zorder=100)
        
        # 定义颜色和线型列表
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
        linestyles = ['--', '-.', ':', '-']
        
        # 绘制每个模型的预测值
        for i, pred in enumerate(models_data):
            model_name = model_names[i] if model_names else f'Model {i+1}'
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            plt.plot(pred, 
                    linestyle=linestyle, 
                    linewidth=2.5,
                    color=color,
                    alpha=0.8,
                    label=model_name)
        
        plt.title(title, fontsize=14)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('负荷 (MW)', fontsize=12)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def accuracy_profile(y_true, predictions_list, 
                        threshold=10,  # 默认阈值设为10%
                        window_size=24, 
                        save_path=None):
        """预测准确率热力图（修正版）"""
        num_points = len(y_true) - window_size + 1
        num_models = len(predictions_list)
        
        # 初始化准确率矩阵 (时间窗口数 x 模型数)
        window_accuracy = np.zeros((num_points, num_models))
        
        # 计算每个窗口每个模型的准确率
        for i in range(num_points):
            for model_idx in range(num_models):
                # 提取当前窗口的预测和真实值
                window_true = y_true[i:i+window_size]
                window_pred = predictions_list[model_idx][i:i+window_size]
                
                # 计算误差百分比
                errors = np.abs((window_pred - window_true) / window_true) * 100
                # 统计准确率（误差 < threshold）
                accurate = np.sum(errors < threshold) / window_size
                window_accuracy[i, model_idx] = accurate
        
        # 计算平均准确率（按模型）
        avg_accuracy = np.mean(window_accuracy, axis=1)
        
        # 绘制热力图
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            avg_accuracy.reshape(-1, 1).T,  # 转换为 (1, num_points) 形状
            annot=False,
            cmap='YlGn',
            vmin=0, vmax=1,
            xticklabels=np.arange(num_points),
            yticklabels=['平均准确率']
        )
        plt.title(f'预测准确率热力图（阈值={threshold}%）')
        plt.xlabel('时间窗口起始点')
        plt.ylabel('')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def error_distribution(models_predictions, true_values, model_names, save_path=None):
        # 示例输入数据检查
        print(f"true_values 形状: {true_values.shape}")  # 应为 (n_samples,)
        print(f"第一个模型预测形状: {models_predictions[0].shape}")  # 应为 (n_samples,)# 展平所有数组并计算误差
        errors = []
        for pred, name in zip(models_predictions, model_names):
            # 确保 pred 和 true_values 是二维数组
            pred = np.array(pred).reshape(-1, 1)
            true = np.array(true_values).reshape(-1, 1)
            # 计算误差并展平为一维
            error = (true - pred).flatten()
            # 为每个误差添加模型标签
            errors.extend([(name, e) for e in error])
        
        # 转换为 DataFrame
        errors_df = pd.DataFrame(errors, columns=['Model', 'Error'])
        
        # 绘制箱型图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Error', data=errors_df)
        plt.title('模型误差分布对比')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def residual_analysis(y_true, y_pred, 
                        model_name='Model', 
                        save_path=None):
        """残差分析图"""
        # 确保输入为一维数组
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_pred - y_true
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=y_true, y=residuals, 
                        alpha=0.7,
                        label='残差分布',
                        s=80)
        plt.axhline(0, color='r', linestyle='--', linewidth=2)
        plt.title(f'{model_name} 残差分析')
        plt.xlabel('真实值')
        plt.ylabel('残差')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


def plot_regional_data(regional_data, title, save_path):
    """Plot regional load data."""
    plt.figure(figsize=(12, 8))
    
    for region, data in regional_data.items():
        plt.plot(data.index, data['load'], label=region)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_evaluation_radar(evaluation_results, save_path):
    """Plot radar chart of evaluation indices."""
    # Extract primary indices for each region
    regions = list(evaluation_results.keys())
    indices = ['ForecastReliability', 'ProvincialLoadImpact', 'ForecastingComplexity']
    
    # Create data for radar chart
    values = np.zeros((len(regions), len(indices)))
    for i, region in enumerate(regions):
        for j, index in enumerate(indices):
            values[i, j] = evaluation_results[region]['indices'][index]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(indices), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, region in enumerate(regions):
        values_closed = values[i].tolist()
        values_closed += values_closed[:1]  # Close the loop
        ax.plot(angles, values_closed, linewidth=2, label=region)
        ax.fill(angles, values_closed, alpha=0.1)
    
    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indices)
    ax.set_title('Evaluation Indices for Regions', size=15)
    ax.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weights(weights, save_path):
    """Plot PCA-derived weights for regions."""
    plt.figure(figsize=(10, 6))
    
    regions = list(weights.keys())
    weight_values = [weights[region] for region in regions]
    
    bars = plt.bar(regions, weight_values, color='royalblue')
    
    # Add weight values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.title('PCA-Derived Weights for Regions')
    plt.ylabel('Weight')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(actual, direct, fusion, save_path):
    """Plot comparison of forecasting methods."""
    plt.figure(figsize=(14, 8))
    
    # Ensure we have data to plot
    if actual.empty or direct.empty or fusion.empty:
        print(f"Warning: Empty data for comparison plot. Skipping {save_path}")
        return
    
    # Slice to 1 day for better visualization if we have enough data
    if len(actual) > 96:  # Only slice if we have enough data
        start_idx = len(actual) // 2
        end_idx = min(start_idx + 96, len(actual))  # 1 day (assuming 15-min intervals)
    else:
        start_idx = 0
        end_idx = len(actual)
    
    plt.plot(actual.index[start_idx:end_idx], 
             actual['load'][start_idx:end_idx], 
             'k-', label='Actual', linewidth=2)
    
    plt.plot(direct.index[start_idx:end_idx], 
             direct['load'][start_idx:end_idx], 
             'r--', label='Direct Aggregation', linewidth=2)
    
    plt.plot(fusion.index[start_idx:end_idx], 
             fusion['load'][start_idx:end_idx], 
             'b-.', label='Weighted Fusion', linewidth=2)
    
    plt.title('Comparison of Forecast Integration Methods')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution(actual, direct, fusion, save_path):
    """Plot error distribution comparison."""
    # Calculate APEs
    direct_ape = np.abs((actual['load'] - direct['load']) / actual['load']) * 100
    fusion_ape = np.abs((actual['load'] - fusion['load']) / actual['load']) * 100
    
    # Create dataframe for plotting
    error_df = pd.DataFrame({
        'Direct Aggregation': direct_ape,
        'Weighted Fusion': fusion_ape
    })
    
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(data=error_df)
    plt.title('Error Distribution Comparison')
    plt.ylabel('Absolute Percentage Error (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_performance_metrics(direct_metrics, fusion_metrics, save_path):
    """Plot performance metrics comparison."""
    plt.figure(figsize=(12, 6))
    
    metrics = list(direct_metrics.keys())
    x = np.arange(len(metrics))
    width = 0.35
    
    direct_values = [direct_metrics[metric] for metric in metrics]
    fusion_values = [fusion_metrics[metric] for metric in metrics]
    
    bars1 = plt.bar(x - width/2, direct_values, width, label='Direct Aggregation')
    bars2 = plt.bar(x + width/2, fusion_values, width, label='Weighted Fusion')
    
    # Add values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontsize=8)
    
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def setup_directories():
    """创建必要的输出目录"""
    directories = ['results/multi_regional', 'results/fusion', 'models/gru', 'models/lstm']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)