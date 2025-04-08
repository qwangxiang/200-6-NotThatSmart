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
from iTransformer.data.data_loader import DataLoader
from iTransformer.data.dataset_builder import DatasetBuilder
from iTransformer.utils.evaluator import ModelEvaluator
from iTransformer.utils.scaler_manager import ScalerManager
from iTransformer.models.torch_models import TorchConvTransformer
import shutil

VOLATILITY_THRESHOLD = 1000

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
    ts_data_path = "data/Data/data.csv"
    
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

def perform_rolling_forecast(start_date, end_date, forecast_interval=5, apply_smoothing=True):
    """
    使用已训练的模型进行日内滚动预测
    
    参数:
    start_date (str): 开始预测的日期 (YYYY-MM-DD)
    end_date (str): 结束预测的日期 (YYYY-MM-DD)
    forecast_interval (int): 预测间隔，单位为分钟
    apply_smoothing (bool): 是否应用平滑处理
    
    返回:
    DataFrame: 包含预测结果的数据框
    """
    print(f"\n=== 进行日内滚动预测 ===")
    print(f"预测期间: {start_date} 到 {end_date}")
    print(f"预测间隔: {forecast_interval}分钟")
    print(f"平滑处理: {'启用' if apply_smoothing else '禁用'}")
    
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
    
    # 保存最近的实际负荷数据，用于波动性计算
    recent_loads = []
    if ts_data.index[-1] >= req_history_start:
        # 加载最近的历史数据作为初始值
        history_data = ts_data.loc[req_history_start:start_datetime-timedelta(minutes=1), 'load'].values
        if len(history_data) > 0:
            recent_loads = list(history_data[-100:])  # 保留最近的100个点
    
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
                # 更新最近负荷历史，用于后续波动性计算
                if not np.isnan(actual_value):
                    recent_loads.append(actual_value)
                    # 只保留最近的100个点
                    if len(recent_loads) > 100:
                        recent_loads = recent_loads[-100:]
            
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
    
    # 应用自适应平滑处理（如果启用）
    if apply_smoothing and len(results_df) > 0:
        # 平滑预测结果
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        # 如果不应用平滑，将原始预测值复制到平滑列
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/rolling_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}")
    
    # 计算指标（仅对有实际值的时间点）
    valid_results = results_df.dropna(subset=['actual'])
    
    if len(valid_results) > 0:
        # 计算原始预测的指标
        mae = np.mean(np.abs(valid_results['actual'] - valid_results['predicted']))
        rmse = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted']) ** 2))
        mape = np.mean(np.abs((valid_results['actual'] - valid_results['predicted']) / valid_results['actual'])) * 100
        
        # 计算平滑后预测的指标
        mae_smooth = np.mean(np.abs(valid_results['actual'] - valid_results['predicted_smoothed']))
        rmse_smooth = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted_smoothed']) ** 2))
        mape_smooth = np.mean(np.abs((valid_results['actual'] - valid_results['predicted_smoothed']) / valid_results['actual'])) * 100
        
        print(f"原始预测指标:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        if apply_smoothing:
            print(f"\n平滑后预测指标:")
            print(f"MAE: {mae_smooth:.2f}")
            print(f"RMSE: {rmse_smooth:.2f}")
            print(f"MAPE: {mape_smooth:.2f}%")
        
        # 保存指标
        metrics = {
            'original': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            },
            'smoothed': {
                'mae': mae_smooth,
                'rmse': rmse_smooth,
                'mape': mape_smooth
            },
            'period': f"{start_date} to {end_date}",
            'forecast_interval': forecast_interval
        }
        
        metrics_path = f"{results_dir}/rolling_metrics_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
    
    # 绘制结果
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['datetime'], results_df['actual'], 'b-', label='实际值', linewidth=2)
    
    if apply_smoothing:
        plt.plot(results_df['datetime'], results_df['predicted'], 'r--', label='原始预测值', linewidth=1, alpha=0.6)
        plt.plot(results_df['datetime'], results_df['predicted_smoothed'], 'g-', label='平滑预测值', linewidth=1.5)
    else:
        plt.plot(results_df['datetime'], results_df['predicted'], 'r--', label='预测值', linewidth=1.5)
    
    plt.title('日内滚动负荷预测')
    plt.xlabel('时间')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 为夜间时段添加背景色
    for date in pd.date_range(start=start_date, end=end_date):
        # 当天晚上10点到次日早上6点
        night_start = datetime.combine(date.date(), datetime.min.time().replace(hour=22))
        night_end = datetime.combine(date.date() + timedelta(days=1), datetime.min.time().replace(hour=6))
        
        # 确保在图表范围内
        if night_start >= start_datetime and night_start <= end_datetime:
            plt.axvspan(night_start, min(night_end, end_datetime), color='lightblue', alpha=0.2, label='_' if date > start_datetime else '夜间')
    
    plot_path = f"{results_dir}/rolling_forecast_plot_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")
    
    return results_df

def select_prediction_model(timestamp, recent_loads=None):
    """
    根据时间和最近负荷情况选择合适的预测模型
    
    参数:
    timestamp (datetime): 当前时间点
    recent_loads (array): 最近的负荷数据，用于计算波动性
    
    返回:
    str: 模型类型标识
    """
    # 将numpy.datetime64转换为可以获取小时的格式
    if isinstance(timestamp, np.datetime64):
        # 转换为pandas Timestamp对象
        ts = pd.Timestamp(timestamp)
        hour = ts.hour
    else:
        # 如果已经是datetime或pandas Timestamp
        hour = timestamp.hour
    
    # 计算负荷波动性 (如果有历史数据)
    volatility = 0
    if recent_loads is not None and len(recent_loads) > 10:
        # 使用最近10个点的标准差作为波动性指标
        volatility = np.std(recent_loads[-10:])
    
    # 夜间模式 (22:00-6:00)
    if hour >= 22 or hour < 6:
        if volatility < 50:  # 低波动阈值，需要根据实际数据调整
            return "night_stable"
        else:
            return "night_volatile"
    # 日间高峰模式 (8:00-20:00)
    elif 8 <= hour <= 20:
        return "daytime_peak"
    # 过渡时段
    else:
        return "transition"

def perform_rolling_forecast_with_patterns(start_date, end_date, forecast_interval=5):
    """
    使用已训练的模型进行日内滚动预测，加入模式识别和平滑处理
    
    参数:
    start_date (str): 开始预测的日期 (YYYY-MM-DD)
    end_date (str): 结束预测的日期 (YYYY-MM-DD)
    forecast_interval (int): 预测间隔，单位为分钟
    
    返回:
    DataFrame: 包含预测结果的数据框
    """
    print(f"\n=== 进行基于模式识别的日内滚动预测 ===")
    print(f"预测期间: {start_date} 到 {end_date}")
    print(f"预测间隔: {forecast_interval}分钟")
    
    # 创建目录
    model_dir = "iTransformer/models/convtrans"
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
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 检查是否有足够的历史数据
    req_history_start = start_datetime - timedelta(minutes=1440)
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
    
    # 准备结果容器
    results = []
    
    # 保存最近的预测和实际负荷数据，用于模式识别
    recent_loads = []
    
    # 存储已加载的模型，避免重复加载
    loaded_models = {}
    
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
                seq_length=1440,
                pred_horizon=1
            )
            
            # 基于时间和负荷历史识别模式
            pattern = select_prediction_model(forecast_time, recent_loads)
            
            # 加载对应模式的模型（如果尚未加载）
            if pattern not in loaded_models:
                try:
                    loaded_models[pattern] = load_model_for_pattern(pattern, model_dir)
                except Exception as e:
                    print(f"加载 {pattern} 模式模型失败: {e}，使用默认模型")
                    if 'default' not in loaded_models:
                        loaded_models['default'] = TorchConvTransformer.load(save_dir=model_dir)
                    pattern = 'default'
            
            # 获取模型
            model = loaded_models[pattern]
            
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
                # 更新最近负荷历史
                recent_loads.append(actual_value)
                # 只保留最近的N个点
                if len(recent_loads) > 100:
                    recent_loads = recent_loads[-100:]
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value,
                'pattern': pattern
            })
            
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}")
            # 继续下一个时间点的预测
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 应用自适应平滑处理
    if len(results_df) > 0:
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/rolling_forecast_pattern_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}")
    
    # 计算指标（仅对有实际值的时间点）
    valid_results = results_df.dropna(subset=['actual'])
    
    if len(valid_results) > 0:
        # 计算原始预测的指标
        mae = np.mean(np.abs(valid_results['actual'] - valid_results['predicted']))
        rmse = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted']) ** 2))
        mape = np.mean(np.abs((valid_results['actual'] - valid_results['predicted']) / valid_results['actual'])) * 100
        
        # 计算平滑后预测的指标
        mae_smooth = np.mean(np.abs(valid_results['actual'] - valid_results['predicted_smoothed']))
        rmse_smooth = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted_smoothed']) ** 2))
        mape_smooth = np.mean(np.abs((valid_results['actual'] - valid_results['predicted_smoothed']) / valid_results['actual'])) * 100
        
        print(f"原始预测指标:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        print(f"\n平滑后预测指标:")
        print(f"MAE: {mae_smooth:.2f}")
        print(f"RMSE: {rmse_smooth:.2f}")
        print(f"MAPE: {mape_smooth:.2f}%")
        
        # 保存指标
        metrics = {
            'original': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            },
            'smoothed': {
                'mae': mae_smooth,
                'rmse': rmse_smooth,
                'mape': mape_smooth
            },
            'period': f"{start_date} to {end_date}",
            'forecast_interval': forecast_interval
        }
        
        metrics_path = f"{results_dir}/rolling_metrics_pattern_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
    
    # 绘制结果
    plt.figure(figsize=(16, 8))
    plt.plot(results_df['datetime'], results_df['actual'], 'b-', label='实际值', linewidth=2)
    plt.plot(results_df['datetime'], results_df['predicted'], 'r--', label='原始预测值', linewidth=1, alpha=0.6)
    plt.plot(results_df['datetime'], results_df['predicted_smoothed'], 'g-', label='平滑预测值', linewidth=1.5)
    plt.title('日内滚动负荷预测（模式识别与平滑处理）')
    plt.xlabel('时间')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 添加模式标记
    # 为每个模式使用不同的背景颜色
    unique_patterns = results_df['pattern'].unique()
    colors = {'night_stable': 'lightblue', 'night_volatile': 'lightgreen', 
              'daytime_peak': 'lightyellow', 'transition': 'lightpink', 'default': 'white'}
    
    y_min, y_max = plt.ylim()
    current_pattern = None
    pattern_start = None
    
    for i, row in results_df.iterrows():
        if row['pattern'] != current_pattern:
            if current_pattern is not None:
                pattern_end = row['datetime']
                plt.axvspan(pattern_start, pattern_end, color=colors.get(current_pattern, 'white'), alpha=0.2)
            current_pattern = row['pattern']
            pattern_start = row['datetime']
    
    # 处理最后一个模式
    if current_pattern is not None:
        plt.axvspan(pattern_start, results_df.iloc[-1]['datetime'], color=colors.get(current_pattern, 'white'), alpha=0.2)
    
    # 添加模式图例
    pattern_patches = [plt.Rectangle((0,0),1,1, color=colors.get(p, 'white'), alpha=0.2) for p in unique_patterns]
    plt.legend(handles=[*plt.gca().get_legend().legend_handles, *pattern_patches],
               labels=[*[h.get_label() for h in plt.gca().get_legend().legend_handles], *unique_patterns])
    
    plot_path = f"{results_dir}/rolling_forecast_pattern_plot_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")
    
    return results_df

def train_pattern_specific_models(train_start_date, train_end_date, retrain=False):
    """
    为不同的负荷模式训练专门的模型
    
    参数:
    train_start_date (str): 训练开始日期
    train_end_date (str): 训练结束日期
    retrain (bool): 是否重新训练已存在的模型
    
    返回:
    dict: 包含各模式模型和训练指标的字典
    """
    print(f"\n=== 为不同负荷模式训练专门模型 ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    
    # 创建目录
    model_dir = "models/convtrans"
    os.makedirs(model_dir, exist_ok=True)
    
    # 初始化缩放器管理器
    scaler_dir = "models/scalers/convtrans"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = "data/timeseries_load.csv"
    
    if os.path.exists(ts_data_path):
        print(f"从 {ts_data_path} 加载时间序列数据...")
        ts_data = pd.read_csv(ts_data_path, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index)
    else:
        raise FileNotFoundError(f"时间序列数据不存在: {ts_data_path}")
    
    # 筛选训练日期范围的数据
    train_data = ts_data.loc[train_start_date:train_end_date].copy()
    
    # 根据时间和负荷特征将数据分为不同的模式
    train_data['hour'] = train_data.index.hour
    
    # 计算负荷滚动标准差（波动性）
    train_data['volatility'] = train_data['load'].rolling(window=10).std().fillna(0)
    
    # 定义各模式的数据筛选条件
    patterns = {
        # 夜间稳定模式：22:00-06:00 且波动性低
        'night_stable': (
            ((train_data['hour'] >= 22) | (train_data['hour'] < 6)) & 
            (train_data['volatility'] < 50)
        ),
        # 夜间波动模式：22:00-06:00 且波动性高
        'night_volatile': (
            ((train_data['hour'] >= 22) | (train_data['hour'] < 6)) & 
            (train_data['volatility'] >= 50)
        ),
        # 日间高峰模式：08:00-20:00
        'daytime_peak': (
            (train_data['hour'] >= 8) & (train_data['hour'] <= 20)
        ),
        # 过渡时段：06:00-08:00 或 20:00-22:00
        'transition': (
            ((train_data['hour'] >= 6) & (train_data['hour'] < 8)) | 
            ((train_data['hour'] > 20) & (train_data['hour'] < 22))
        )
    }
    
    # 训练结果容器
    model_results = {}
    
    # 先准备全部数据用于全局标准化器
    X_train_all, y_train_all, X_val_all, y_val_all = prepare_data_for_train(
        train_data[['load']], 
        seq_length=1440,
        pred_horizon=1,
        test_ratio=0.2
    )
    
    # 初始化全局标准化器（如果需要）
    if not scaler_manager.has_scaler('X'):
        print("初始化特征标准化器...")
        X_reshape = X_train_all.reshape(X_train_all.shape[0], -1)
        scaler_manager.fit('X', X_reshape)
    
    if not scaler_manager.has_scaler('y'):
        print("初始化目标标准化器...")
        y_train_shaped = y_train_all.reshape(-1, 1) if len(y_train_all.shape) == 1 else y_train_all
        scaler_manager.fit('y', y_train_shaped)
    
    # 默认模型配置
    model_config = {
        'seq_length': 1440,  # 输入序列长度
        'pred_length': 1,    # 预测步长
        'batch_size': 32,    # 批量大小
        'lr': 1e-4,          # 学习率
        'epochs': 20,        # 最大训练轮数
        'patience': 10       # 早停耐心值
    }
    
    # 首先检查并训练默认模型
    default_model_path = f"{model_dir}/convtrans_model.pth"
    if os.path.exists(default_model_path) and not retrain:
        print(f"从 {default_model_path} 加载现有默认模型")
        try:
            default_model = TorchConvTransformer.load(save_dir=model_dir)
            model_results['default'] = default_model
        except Exception as e:
            print(f"加载默认模型失败: {e}, 将重新训练")
            retrain = True
    
    # 如果需要训练默认模型
    if 'default' not in model_results:
        print("训练默认 ConvTransformer 模型（全部数据）...")
        
        # 数据标准化
        X_train_reshaped = X_train_all.reshape(X_train_all.shape[0], -1)
        X_train_scaled = scaler_manager.transform('X', X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(X_train_all.shape)
        
        X_val_reshaped = X_val_all.reshape(X_val_all.shape[0], -1)
        X_val_scaled = scaler_manager.transform('X', X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(X_val_all.shape)
        
        y_train_shaped = y_train_all.reshape(-1, 1) if len(y_train_all.shape) == 1 else y_train_all
        y_train_scaled = scaler_manager.transform('y', y_train_shaped)
        
        y_val_shaped = y_val_all.reshape(-1, 1) if len(y_val_all.shape) == 1 else y_val_all
        y_val_scaled = scaler_manager.transform('y', y_val_shaped)
        
        # 创建并训练默认模型
        input_shape = X_train_scaled.shape[1:]
        default_model = TorchConvTransformer(input_shape=input_shape, **model_config)
        
        default_model.train(
            X_train_scaled, y_train_scaled.flatten() if len(y_train_scaled.shape) > 1 else y_train_scaled,
            X_val_scaled, y_val_scaled.flatten() if len(y_val_scaled.shape) > 1 else y_val_scaled,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            save_dir=model_dir
        )
        
        model_results['default'] = default_model
    
    # 为每种模式训练专门的模型
    for pattern_name, pattern_mask in patterns.items():
        # 为当前模式从默认模型复制一个模型文件
        pattern_model_path = f"{model_dir}/convtrans_model_{pattern_name}.pth"
        
        # 检查是否已有该模式的模型且不需要重新训练
        if os.path.exists(pattern_model_path) and not retrain:
            print(f"检查到现有的 {pattern_name} 模式模型文件")
            try:
                # 尝试加载现有模型
                pattern_model = TorchConvTransformer.load(
                    save_dir=model_dir,
                    filename=f"convtrans_model_{pattern_name}.pth"
                )
                model_results[pattern_name] = pattern_model
                print(f"成功加载 {pattern_name} 模式模型")
                continue  # 已成功加载，跳过训练
            except Exception as e:
                print(f"尝试加载 {pattern_name} 模式模型时出错: {e}")
                print(f"将为 {pattern_name} 模式重新训练模型")
        
        # 提取该模式的数据
        pattern_data = train_data[pattern_mask][['load']].copy()
        
        # 检查该模式是否有足够的数据点
        if len(pattern_data) < 2000:  # 根据需要调整最小数据点要求
            print(f"模式 {pattern_name} 的数据点不足 ({len(pattern_data)} < 2000)，跳过训练专门模型")
            continue
        
        print(f"\n准备训练 {pattern_name} 模式的专门模型...")
        print(f"该模式数据点数量: {len(pattern_data)}")
        
        # 准备该模式的训练数据
        try:
            X_train_pattern, y_train_pattern, X_val_pattern, y_val_pattern = prepare_data_for_train(
                pattern_data, 
                seq_length=1440,
                pred_horizon=1,
                test_ratio=0.2
            )
            
            # 检查数据集大小
            if len(X_train_pattern) < 100 or len(X_val_pattern) < 20:
                print(f"模式 {pattern_name} 的处理后训练数据不足 (训练集: {len(X_train_pattern)}, 验证集: {len(X_val_pattern)})，跳过训练")
                continue
            
            # 数据标准化（使用全局标准化器）
            X_train_reshaped = X_train_pattern.reshape(X_train_pattern.shape[0], -1)
            X_train_scaled = scaler_manager.transform('X', X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train_pattern.shape)
            
            X_val_reshaped = X_val_pattern.reshape(X_val_pattern.shape[0], -1)
            X_val_scaled = scaler_manager.transform('X', X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(X_val_pattern.shape)
            
            y_train_shaped = y_train_pattern.reshape(-1, 1) if len(y_train_pattern.shape) == 1 else y_train_pattern
            y_train_scaled = scaler_manager.transform('y', y_train_shaped)
            
            y_val_shaped = y_val_pattern.reshape(-1, 1) if len(y_val_pattern.shape) == 1 else y_val_pattern
            y_val_scaled = scaler_manager.transform('y', y_val_shaped)
            
            # 为不同模式定制模型参数
            pattern_config = model_config.copy()
            if pattern_name == 'night_stable':
                # 夜间稳定模式可以使用更简单的模型，学习率更低以捕捉更平稳的模式
                pattern_config['lr'] = 5e-5
                pattern_config['patience'] = 15
            elif pattern_name == 'night_volatile':
                # 夜间波动模式需要更灵敏的模型来捕捉波动
                pattern_config['lr'] = 2e-4
            elif pattern_name == 'daytime_peak':
                # 日间高峰模式可能需要更复杂的模型来捕捉复杂模式
                pattern_config['batch_size'] = 24
                pattern_config['patience'] = 8
            
            # 创建模型
            input_shape = X_train_scaled.shape[1:]
            pattern_model = TorchConvTransformer(input_shape=input_shape, **pattern_config)
            
            print(f"开始训练 {pattern_name} 模式模型...")
            
            # 训练模型
            pattern_model.train(
                X_train_scaled, y_train_scaled.flatten() if len(y_train_scaled.shape) > 1 else y_train_scaled,
                X_val_scaled, y_val_scaled.flatten() if len(y_val_scaled.shape) > 1 else y_val_scaled,
                epochs=pattern_config['epochs'],
                batch_size=pattern_config['batch_size']
            )
            
            # 保存模型 - 使用正确的文件名参数
            # pattern_model.save(
            #     save_dir=model_dir, 
            #     filename=f"convtrans_model_{pattern_name}.pth"
            # )
            temp_dir = f"{model_dir}/temp_{pattern_name}"
            os.makedirs(temp_dir, exist_ok=True)
            pattern_model.save(save_dir=temp_dir)

            # 2. 手动复制和重命名文件

            source_path = f"{temp_dir}/{pattern_model.model_type}_model.pth"
            dest_path = f"{model_dir}/convtrans_model_{pattern_name}.pth"
            shutil.copy2(source_path, dest_path)

            # 3. 复制其他必要文件（如配置文件）
            source_config = f"{temp_dir}/{pattern_model.model_type}_config.json"
            dest_config = f"{model_dir}/{pattern_model.model_type}_config_{pattern_name}.json"
            shutil.copy2(source_config, dest_config)

            # 4. 复制输入形状文件
            source_shape = f"{temp_dir}/input_shape.json"
            dest_shape = f"{model_dir}/input_shape_{pattern_name}.json"
            shutil.copy2(source_shape, dest_shape)

            # 5. 可选：删除临时目录
            shutil.rmtree(temp_dir)
            
            print(f"{pattern_name} 模式模型训练完成并保存")
            model_results[pattern_name] = pattern_model
            
        except Exception as e:
            print(f"训练 {pattern_name} 模式模型时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== 模式专用模型训练完成 ===")
    print(f"成功训练或加载的模式模型: {list(model_results.keys())}")
    
    return model_results

def load_model_for_pattern(pattern, model_dir="models/convtrans"):
    """
    根据识别的模式加载相应的模型
    
    参数:
    pattern (str): 模式类型
    model_dir (str): 模型目录
    
    返回:
    model: 加载的模型对象
    """
    # 检查模式特定模型文件是否存在
    pattern_model_path = f"{model_dir}/convtrans_model_{pattern}.pth"
    pattern_config_path = f"{model_dir}/{TorchConvTransformer.model_type}_config_{pattern}.json"
    pattern_shape_path = f"{model_dir}/input_shape_{pattern}.json"
    
    if os.path.exists(pattern_model_path) and os.path.exists(pattern_config_path) and os.path.exists(pattern_shape_path):
        try:
            print(f"加载 {pattern} 模式的专用模型...")
            
            # 1. 创建临时目录
            temp_dir = f"{model_dir}/temp_load_{pattern}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 2. 复制文件到临时目录，使用标准名称
            shutil.copy2(pattern_model_path, f"{temp_dir}/{TorchConvTransformer.model_type}_model.pth")
            shutil.copy2(pattern_config_path, f"{temp_dir}/{TorchConvTransformer.model_type}_config.json")
            shutil.copy2(pattern_shape_path, f"{temp_dir}/input_shape.json")
            
            # 3. 从临时目录加载模型
            model = TorchConvTransformer.load(save_dir=temp_dir)
            
            # 4. 删除临时目录
            shutil.rmtree(temp_dir)
            
            return model
        except Exception as e:
            print(f"加载 {pattern} 模式模型失败: {e}，使用默认模型")
    else:
        print(f"未找到 {pattern} 模式的专用模型，使用默认模型...")
    
    # 如果无法加载特定模式模型，回退到默认模型
    return TorchConvTransformer.load(save_dir=model_dir)

def apply_adaptive_smoothing(predictions, timestamps, load_levels=None):
    """
    根据时间段和负荷水平自适应地应用平滑处理
    
    参数:
    predictions (array): 原始预测值
    timestamps (array): 对应的时间戳
    load_levels (array, optional): 对应的负荷水平，用于动态调整平滑因子
    
    返回:
    array: 平滑后的预测值
    """
    smoothed = np.copy(predictions)
    
    # 如果数据点太少，不进行平滑
    if len(predictions) < 5:
        return smoothed
    
    # 首先应用指数加权移动平均
    alpha_day = 0.5   # 日间平滑因子 (较小)
    alpha_night = 0.8  # 夜间平滑因子 (较大)
    
    for i in range(1, len(predictions)):
        # 将numpy.datetime64转换为可以获取小时的格式
        if isinstance(timestamps[i], np.datetime64):
            # 转换为pandas Timestamp对象
            ts = pd.Timestamp(timestamps[i])
            hour = ts.hour
        else:
            # 如果已经是datetime或pandas Timestamp
            hour = timestamps[i].hour
            
        # 选择合适的平滑因子
        if hour >= 22 or hour < 6:  # 夜间
            alpha = alpha_night
        else:  # 日间
            alpha = alpha_day
            
        # 应用指数平滑
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * predictions[i]
    
    # 对夜间数据点再应用中值滤波以去除毛刺
    for i in range(2, len(predictions)-2):
        # 将numpy.datetime64转换为可以获取小时的格式
        if isinstance(timestamps[i], np.datetime64):
            # 转换为pandas Timestamp对象
            ts = pd.Timestamp(timestamps[i])
            hour = ts.hour
        else:
            # 如果已经是datetime或pandas Timestamp
            hour = timestamps[i].hour
            
        if hour >= 22 or hour < 6:  # 仅对夜间数据应用
            # 使用5点中值滤波
            window = [smoothed[i-2], smoothed[i-1], smoothed[i], smoothed[i+1], smoothed[i+2]]
            smoothed[i] = np.median(window)
    
    return smoothed

def select_prediction_model(timestamp, recent_loads=None):
    """
    根据时间和最近负荷情况选择合适的预测模型
    
    参数:
    timestamp (datetime): 当前时间点
    recent_loads (array): 最近的负荷数据，用于计算波动性
    
    返回:
    str: 模型类型标识
    """
    # 将numpy.datetime64转换为可以获取小时的格式
    if isinstance(timestamp, np.datetime64):
        # 转换为pandas Timestamp对象
        ts = pd.Timestamp(timestamp)
        hour = ts.hour
    else:
        # 如果已经是datetime或pandas Timestamp
        hour = timestamp.hour
    
    # 计算负荷波动性 (如果有历史数据)
    volatility = 0
    if recent_loads is not None and len(recent_loads) > 10:
        # 使用最近10个点的标准差作为波动性指标
        volatility = np.std(recent_loads[-10:])
    
    # 夜间模式 (22:00-6:00)
    if hour >= 22 or hour < 6:
        if volatility < 50:  # 低波动阈值，需要根据实际数据调整
            return "night_stable"
        else:
            return "night_volatile"
    # 日间高峰模式 (8:00-20:00)
    elif 8 <= hour <= 20:
        return "daytime_peak"
    # 过渡时段
    else:
        return "transition"

def main(mode='forecast', retrain=False, train_start='2025-03-10', train_end='2025-03-30', forecast_start='2025-04-01', forecast_end='2025-04-07', interval=5, use_patterns=False, train_patterns=False):
    """主函数，提供命令行参数解析并根据参数执行相应功能"""
    # 记录开始时间
    start_time = time.time()
    
    # 创建必要目录
    os.makedirs("models/convtrans", exist_ok=True)
    os.makedirs("results/convtrans", exist_ok=True)
    os.makedirs("models/scalers/convtrans", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 根据模式执行相应操作
    if train_patterns and mode in ['train', 'both']:
        # 为不同模式分别训练模型
        print("=== 为不同负荷模式训练专门模型 ===")
        # 这里需要实现具体的分时段训练逻辑
        train_pattern_specific_models(
            train_start_date=train_start,
            train_end_date=train_end,
            retrain=retrain
        )
        pass
    elif mode in ['train', 'both']:
        print("=== 开始模型训练 ===")
        train_forecast_model(
            train_start_date=train_start,
            train_end_date=train_end,
            forecast_date=forecast_start,
            retrain=retrain
        )
    

    if mode in ['forecast', 'both']:
        print("=== 开始滚动预测 ===")
        try:
            if use_patterns:
                # 使用基于模式识别的预测方法
                result = perform_rolling_forecast_with_patterns(
                    start_date=forecast_start,
                    end_date=forecast_end,
                    forecast_interval=interval
                )
            else:
                # 使用原始的预测方法
                result = perform_rolling_forecast(
                    start_date=forecast_start,
                    end_date=forecast_end,
                    forecast_interval=interval
                )
        except FileNotFoundError as e:
            print(f"错误: {e}")
            print("请先训练模型或提供已训练的模型文件")
    
    # 计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n执行完成，用时 {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")

    return result

if __name__ == "__main__":
    main()