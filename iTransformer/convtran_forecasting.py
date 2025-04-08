#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch-based load forecasting script using TorchConvTransformer model.
Modified to perform intraday rolling forecasts with optional training.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import warnings
import argparse
warnings.filterwarnings('ignore')

# Import project components
from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from utils.evaluator import ModelEvaluator
from utils.scaler_manager import ScalerManager
from models.torch_models import TorchConvTransformer

def prepare_data_for_train(ts_data, seq_length=1440, pred_horizon=1, test_ratio=0.2):
    """
    准备训练和验证数据
    
    参数:
    ts_data (DataFrame): 时间序列格式的负荷数据
    seq_length (int): 输入序列长度
    pred_horizon (int): 预测步长
    test_ratio (float): 测试集比例
    
    返回:
    tuple: X_train, y_train, X_val, y_val
    """
    # 提取负荷数据
    load_values = ts_data['load'].values
    
    # 创建特征和标签
    X, y = [], []
    for i in range(len(load_values) - seq_length - pred_horizon + 1):
        X.append(load_values[i:i+seq_length])
        y.append(load_values[i+seq_length:i+seq_length+pred_horizon])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, pred_horizon)
    
    # 划分训练集和验证集
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 为模型添加特征维度
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    return X_train, y_train, X_val, y_val

def prepare_data_for_test(ts_data, seq_length=1440, pred_horizon=1):
    """
    准备测试数据
    
    参数:
    ts_data (DataFrame): 时间序列格式的负荷数据
    seq_length (int): 输入序列长度
    pred_horizon (int): 预测步长
    
    返回:
    tuple: X_test, y_test
    """
    # 提取负荷数据
    load_values = ts_data['load'].values
    
    # 创建特征和标签
    X, y = [], []
    for i in range(len(load_values) - seq_length - pred_horizon + 1):
        X.append(load_values[i:i+seq_length])
        y.append(load_values[i+seq_length:i+seq_length+pred_horizon])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, pred_horizon)
    
    # 为模型添加特征维度
    X_test = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X_test, y

def prepare_data_for_rolling_forecast(ts_data, start_time, seq_length=1440, pred_horizon=1):
    """
    准备滚动预测的输入数据
    
    参数:
    ts_data (DataFrame): 时间序列格式的负荷数据
    start_time (datetime): 开始预测的时间点
    seq_length (int): 输入序列长度
    pred_horizon (int): 预测步长
    
    返回:
    numpy.ndarray: 模型输入数据
    """
    # 找到起始时间之前的seq_length个点
    end_time = start_time - timedelta(minutes=1)  # 假设数据间隔为1分钟
    start_hist = end_time - timedelta(minutes=seq_length-1)
    
    # 提取历史数据
    historical_data = ts_data.loc[start_hist:end_time, 'load'].values
    
    # 确保长度正确
    if len(historical_data) < seq_length:
        # 如果历史数据不足，用第一个值填充
        padding = np.full(seq_length - len(historical_data), historical_data[0] if len(historical_data) > 0 else 0)
        historical_data = np.concatenate([padding, historical_data])
    
    # 添加特征维度并创建batch维度
    X = historical_data.reshape(1, seq_length, 1)
    
    return X

def train_forecast_model(train_start_date, train_end_date, forecast_date, retrain=True):
    """
    训练模型并预测负荷
    
    参数:
    train_start_date (str): 训练开始日期
    train_end_date (str): 训练结束日期
    forecast_date (str): 预测日期
    retrain (bool): 是否重新训练模型，即使已有训练好的模型
    
    返回:
    dict: 包含模型、预测结果、实际值和评估指标的字典
    """
    print(f"\n=== 训练模型并进行负荷预测 ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    print(f"预测日期: {forecast_date}")
    
    # 创建目录
    model_dir = "models/convtrans"
    results_dir = "results/convtrans"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化缩放器管理器
    scaler_dir = "models/scalers/convtrans"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 1. 将宽格式数据转换为时间序列格式
    ts_data_path = "data/timeseries_load.csv"
    
    # 检查是否需要重新转换数据
    if os.path.exists(ts_data_path):
        print(f"从 {ts_data_path} 加载时间序列数据...")
        ts_data = pd.read_csv(ts_data_path, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据加载器和数据集构建器
    data_loader = DataLoader()
    dataset_builder = DatasetBuilder(data_loader, standardize=False)
    
    # 筛选训练和预测日期范围的数据
    train_data = ts_data.loc[train_start_date:train_end_date]
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    forecast_data = ts_data.loc[forecast_date:forecast_end_date]
    
    # 创建数据集
    X_train, y_train, X_val, y_val = prepare_data_for_train(
        train_data, 
        seq_length=1440,  # 24小时，假设数据为1分钟间隔
        pred_horizon=1,
        test_ratio=0.2   # 20%的数据用于验证
    )
    
    X_test, y_test = prepare_data_for_test(
        forecast_data,
        seq_length=1440,
        pred_horizon=1,
    )
    
    # 创建时间戳
    train_timestamps = train_data.index[:len(y_train)]
    val_timestamps = train_data.index[-len(y_val):]
    test_timestamps = forecast_data.index[:len(y_test)]
    
    # 打印数据集信息
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # 标准化数据
    if not scaler_manager.has_scaler('X'):
        X_reshape = X_train.reshape(X_train.shape[0], -1)
        scaler_manager.fit('X', X_reshape)
    
    if not scaler_manager.has_scaler('y'):
        y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        scaler_manager.fit('y', y_train_reshaped)
    
    # 应用标准化
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_scaled = scaler_manager.transform('X', X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_val_scaled = scaler_manager.transform('X', X_val_reshaped)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    X_test_scaled = scaler_manager.transform('X', X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    y_train_shaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
    y_train_scaled = scaler_manager.transform('y', y_train_shaped)
    
    y_val_shaped = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
    y_val_scaled = scaler_manager.transform('y', y_val_shaped)
    
    # 模型配置
    convtrans_config = {
        'seq_length': X_train.shape[1],      # 输入序列长度
        'pred_length': 1,      # 预测步长
        'batch_size': 32,      # 批量大小
        'lr': 1e-4,            # 学习率
        'epochs': 50,          # 最大训练轮数
        'patience': 10         # 早停耐心值
    }
    
    # 创建和训练模型
    input_shape = X_train_scaled.shape[1:]
    convtrans_model = TorchConvTransformer(input_shape=input_shape, **convtrans_config)
    
    # 检查模型是否已存在
    model_path = f"{model_dir}/convtrans_model.pth"
    if os.path.exists(model_path) and not retrain:
        print(f"从 {model_path} 加载现有模型")
        convtrans_model = TorchConvTransformer.load(save_dir=model_dir)
    else:
        print("训练 ConvTransformer 模型...")
        convtrans_model.train(
            X_train_scaled, y_train_scaled.flatten() if len(y_train_scaled.shape) > 1 else y_train_scaled,
            X_val_scaled, y_val_scaled.flatten() if len(y_val_scaled.shape) > 1 else y_val_scaled,
            epochs=convtrans_config['epochs'],
            batch_size=convtrans_config['batch_size'],
            save_dir=model_dir
        )
        
        # 保存模型
        convtrans_model.save(save_dir=model_dir)
    
    # 做预测
    print("进行预测...")
    raw_pred = convtrans_model.predict(X_test_scaled)
    
    # 确保形状正确
    raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred
    
    # 反向标准化
    pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
    
    # 确保形状与y_test匹配
    if len(pred_inverse.shape) > 1 and len(y_test.shape) == 1:
        pred_inverse = pred_inverse.flatten()
        
    # 创建结果数据框
    results_df = pd.DataFrame({
        'datetime': test_timestamps[:len(pred_inverse)],
        'actual': y_test[:len(pred_inverse)].flatten(),
        'predicted': pred_inverse.flatten()
    })
    
    # 保存结果
    csv_path = f"{results_dir}/forecast_results_{pd.to_datetime(forecast_date).strftime('%Y%m%d')}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}")
    
    # 绘制结果
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['datetime'], results_df['actual'], 'b-', label='实际值', linewidth=2)
    plt.plot(results_df['datetime'], results_df['predicted'], 'r--', label='预测值', linewidth=1.5)
    plt.title('负荷预测')
    plt.xlabel('时间')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = f"{results_dir}/forecast_plot_{pd.to_datetime(forecast_date).strftime('%Y%m%d')}.png"
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")
    
    # 计算指标
    mae = np.mean(np.abs(y_test[:len(pred_inverse)] - pred_inverse))
    rmse = np.sqrt(np.mean((y_test[:len(pred_inverse)] - pred_inverse) ** 2))
    mape = np.mean(np.abs((y_test[:len(pred_inverse)] - pred_inverse) / y_test[:len(pred_inverse)])) * 100
    
    print(f"预测指标:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 保存指标
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    with open(f"{results_dir}/metrics_{pd.to_datetime(forecast_date).strftime('%Y%m%d')}.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    
    return {
        'model': convtrans_model,
        'predictions': pred_inverse,
        'actual': y_test[:len(pred_inverse)],
        'timestamps': test_timestamps[:len(pred_inverse)],
        'metrics': metrics
    }

def perform_rolling_forecast(start_date, end_date, forecast_interval=5):
    """
    使用已训练的模型进行日内滚动预测
    
    参数:
    start_date (str): 开始预测的日期 (YYYY-MM-DD)
    end_date (str): 结束预测的日期 (YYYY-MM-DD)
    forecast_interval (int): 预测间隔，单位为分钟
    
    返回:
    DataFrame: 包含预测结果的数据框
    """
    print(f"\n=== 进行日内滚动预测 ===")
    print(f"预测期间: {start_date} 到 {end_date}")
    print(f"预测间隔: {forecast_interval}分钟")
    
    # 创建目录
    model_dir = "models/convtrans"
    results_dir = "results/convtrans"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化缩放器管理器
    scaler_dir = "models/scalers/convtrans"
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 检查模型和缩放器是否存在
    if not os.path.exists(f"{model_dir}/convtrans_model.pth"):
        raise FileNotFoundError(f"模型文件不存在: {model_dir}/convtrans_model.pth")
    
    if not (scaler_manager.has_scaler('X') and scaler_manager.has_scaler('y')):
        raise FileNotFoundError("缩放器不存在，请先训练模型")
    
    # 加载时间序列数据
    ts_data_path = "data/timeseries_load.csv"
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据不存在: {ts_data_path}")
    
    print(f"从 {ts_data_path} 加载时间序列数据...")
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 确保我们有覆盖预测时间范围的数据
    # 注意：对于滚动预测，我们需要有实际数据用于比较和评估
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 检查是否有足够的历史数据用于第一次预测
    req_history_start = start_datetime - timedelta(minutes=1440)  # 需要24小时的历史数据
    if ts_data.index.min() > req_history_start:
        raise ValueError(f"没有足够的历史数据。需要从 {req_history_start} 开始的数据")
    
    # 获取预测期间的实际数据（用于评估）
    actual_data = ts_data.loc[start_datetime:end_datetime].copy()
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 加载模型
    print(f"从 {model_dir} 加载已训练的模型...")
    model = TorchConvTransformer.load(save_dir=model_dir)
    
    # 准备结果容器
    results = []
    
    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...")
    for i, forecast_time in enumerate(forecast_times):
        if i % 50 == 0:  # 每50次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}")
            
        try:
            # 准备输入数据
            X = prepare_data_for_rolling_forecast(
                ts_data, 
                forecast_time, 
                seq_length=1440,  # 24小时，假设数据为1分钟间隔
                pred_horizon=1
            )
            
            # 标准化输入数据
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = scaler_manager.transform('X', X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # 进行预测
            raw_pred = model.predict(X_scaled)
            
            # 确保形状正确
            raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred
            
            # 反向标准化
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
            predicted_value = pred_inverse.flatten()[0]
            
            # 获取实际值（如果有）
            actual_value = np.nan
            if forecast_time in actual_data.index:
                actual_value = actual_data.loc[forecast_time, 'load']
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value
            })
            
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}")
            # 继续下一个时间点的预测
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/rolling_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}")
    
    # 计算指标（仅对有实际值的时间点）
    valid_results = results_df.dropna(subset=['actual'])
    
    if len(valid_results) > 0:
        mae = np.mean(np.abs(valid_results['actual'] - valid_results['predicted']))
        rmse = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted']) ** 2))
        mape = np.mean(np.abs((valid_results['actual'] - valid_results['predicted']) / valid_results['actual'])) * 100
        
        print(f"预测指标 (仅对有实际值的时间点):")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 保存指标
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'period': f"{start_date} to {end_date}",
            'forecast_interval': forecast_interval
        }
        
        metrics_path = f"{results_dir}/rolling_metrics_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
    
    # 绘制结果
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['datetime'], results_df['actual'], 'b-', label='实际值', linewidth=2)
    plt.plot(results_df['datetime'], results_df['predicted'], 'r--', label='预测值', linewidth=1.5)
    plt.title('日内滚动负荷预测')
    plt.xlabel('时间')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = f"{results_dir}/rolling_forecast_plot_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")
    
    return results_df

def load_model_with_check(model_class, save_dir):
    """安全加载模型，带错误处理"""
    try:
        return model_class.load(save_dir=save_dir)
    except (FileNotFoundError, OSError) as e:
        print(f"模型加载失败: {e}")
        return None

def main():
    """主函数，提供命令行参数解析并根据参数执行相应功能"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='负荷预测工具')
    parser.add_argument('--mode', type=str, choices=['train', 'forecast', 'both'], default='both',
                      help='操作模式: train=仅训练, forecast=仅预测, both=训练并预测 (默认: both)')
    parser.add_argument('--retrain', action='store_true', default=True,
                      help='是否重新训练现有模型 (默认: False)')
    parser.add_argument('--train_start', type=str, default='2025-03-10',
                      help='训练数据开始日期 (默认: 2025-03-10)')
    parser.add_argument('--train_end', type=str, default='2025-04-05',
                      help='训练数据结束日期 (默认: 2025-03-30)')
    parser.add_argument('--forecast_start', type=str, default='2025-04-06',
                      help='预测开始日期 (默认: 2025-04-01)')
    parser.add_argument('--forecast_end', type=str, default='2025-04-07',
                      help='预测结束日期 (默认: 2025-04-07)')
    parser.add_argument('--interval', type=int, default=5,
                      help='滚动预测间隔（分钟）(默认: 5)')
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建必要目录
    os.makedirs("models/convtrans", exist_ok=True)
    os.makedirs("results/convtrans", exist_ok=True)
    os.makedirs("models/scalers/convtrans", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 根据模式执行相应操作
    if args.mode in ['train', 'both']:
        print("=== 开始模型训练 ===")
        train_forecast_model(
            train_start_date=args.train_start,
            train_end_date=args.train_end,
            forecast_date=args.forecast_start,
            retrain=args.retrain
        )
    
    if args.mode in ['forecast', 'both']:
        print("=== 开始滚动预测 ===")
        try:
            perform_rolling_forecast(
                start_date=args.forecast_start,
                end_date=args.forecast_end,
                forecast_interval=args.interval
            )
        except FileNotFoundError as e:
            print(f"错误: {e}")
            print("请先训练模型或提供已训练的模型文件")
    
    # 计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n执行完成，用时 {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")

if __name__ == "__main__":
    main()