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

# import by wx
import matplotlib.pyplot as plt
import streamlit as st
from utils import ReadData
from Globals import devices_lib,PHONE_NUM,PASSWORD,TIME_INTERVAL




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

def resample_and_pivot(df:pd.DataFrame):
    # 将时间列转换为datetime格式
    df['Time'] = pd.to_datetime(df['Time'])

    # 检查是否有重复的时间戳，如果有，取平均值
    if df.duplicated('Time').any():
        print("发现重复的时间戳，对重复值取平均...")
        df = df.groupby('Time')['P'].mean().reset_index()

    # 设置时间为索引
    df.set_index('Time', inplace=True)

    # 确定时间序列的起止时间
    start_time = df.index.min()
    end_time = df.index.max()

    # 创建完整的分钟级时间索引
    full_range = pd.date_range(start=start_time, end=end_time, freq='1min')

    # 重新索引并前向填充缺失值
    resampled = df['P'].reindex(full_range).ffill()

    # 创建新的列来提取日期和时间
    resampled_df = resampled.reset_index()
    resampled_df.columns = ['Time', 'P'] # 重命名列
    resampled_df['Date'] = resampled_df['Time'].dt.date
    resampled_df['Minute'] = resampled_df['Time'].dt.strftime('%H:%M')

    resampled_df = resampled_df[['Time', 'P']]
    # 将resampled_df的列名修改为['timestamp', 'load']
    resampled_df.columns = ['timestamp', 'load']


    return resampled_df

@st.cache_data(ttl=1*60)
def Load_Data(date_str:str):
    '''
    加载数据
    '''

    # 计算date_str前一天的日期
    date = datetime.strptime(date_str, '%Y-%m-%d')
    yesterday = date - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    df_date = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=date_str, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df_yesterday = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=yesterday_str, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P').iloc[:-1,:]

    df_date_resampled = resample_and_pivot(df_date)
    df_yesterday_resampled = resample_and_pivot(df_yesterday)

    return df_date_resampled, df_yesterday_resampled

@st.cache_resource
def Load_Model(model_dir:str='iTransformer/models/convtrans'):
    '''
    加载模型
    '''
    convtrans_model = TorchConvTransformer.load(save_dir=model_dir)
    return convtrans_model

# 增量式预测
def Incremental_Predict(model_dir, data_time:str):
    '''
    增量式预测函数
    '''



def Predict():
    '''
    预测函数
    '''
    # 选择日期
    date_str = str(st.date_input("选择日期", value=datetime.now()))

    # 加载数据集
    df_date_resampled, df_yesterday_resampled = Load_Data(date_str)
    # 获取今天数据的所有时间戳
    today_timestamps = df_date_resampled['timestamp'].to_list()
    # 将Timestamp转换为字符串
    today_timestamps = [timestamp.strftime('%Y-%m-%d %H:%M:%S')[-8:] for timestamp in today_timestamps]
    
    # 将两个df合并
    forecast_data = pd.concat([df_yesterday_resampled, df_date_resampled], axis=0)

    X_test, y_test = prepare_data_for_test(
        forecast_data,
        seq_length=1440,
        pred_horizon=1,
    )

    # 初始化缩放器管理器
    scaler_dir = "iTransformer/models/scalers/convtrans"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)

    # 数据标准化
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    X_test_scaled = scaler_manager.transform('X', X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # 加载模型
    convtrans_model = Load_Model()
    # 创建一个空元组
    # raw_pred = ()
    # for data in X_test_scaled:
    #     # 添加一个维度
    #     data = np.expand_dims(data, axis=0)
    #     pred_step = convtrans_model.predict(data)
    #     # 将预测结果添加到元组中，不改变元组的维度
    #     raw_pred += (pred_step,)
    raw_pred = convtrans_model.predict(X_test_scaled)
    # 删除最后一个维度
    raw_pred = np.array(raw_pred).squeeze(axis=1)
    raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred

    pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
    if len(pred_inverse.shape) > 1 and len(y_test.shape) == 1:
        pred_inverse = pred_inverse.flatten()

    st.write(pred_inverse)
    st.write(y_test)

    pass


if __name__=='__page__':

    st.write('运行开始..')


    # 1. 加载数据集
    # LoadData_(ts_data_path='iTransformer/data/timeseries_load.csv')

    Predict()


    pass






