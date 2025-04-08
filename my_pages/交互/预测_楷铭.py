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
import torch
from pyecharts.charts import Line
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts as st_echarts
from iTransformer import convtran_forecasting
from streamlit_extras.card import card
from iTransformer import convtran_forecasting



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

    df_date = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=date_str, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P').iloc[:-1,:]
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

def Predict(date_str:str):
    '''
    预测函数

    Returns:
    --------
    today_timestamps: list
        今天的时间戳列表
    pred_inverse: np.ndarray
        预测值
    data_of_next_min: float
        下一分钟的预测值
    mae: float
        平均绝对误差
    rmse: float
        均方根误差
    mape: float
        平均绝对百分比误差
    '''

    # 加载数据集
    df_date_resampled, df_yesterday_resampled = Load_Data(date_str)
 

    # 获取今天数据的所有时间戳
    today_timestamps_origin = df_date_resampled['timestamp'].to_list()

    # 计算today_timestamps最后一个事时间字符串的下一分钟对应的字符串
    last_timestamp = today_timestamps_origin[-1].strftime('%Y-%m-%d %H:%M:%S')

    # 转换为字符串
    today_timestamps = [timestamp.strftime('%Y-%m-%d %H:%M:%S')[-8:-3] for timestamp in today_timestamps_origin]


    last_timestamp = datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=1)
    last_timestamp = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    # # 将最后一个时间戳添加到today_timestamps中
    today_timestamps.append(last_timestamp[-8:-3])

    # 将两个df合并
    forecast_data = pd.concat([df_yesterday_resampled, df_date_resampled], axis=0)

    X_test, y_test = prepare_data_for_test(
        forecast_data,
        seq_length=1440,
        pred_horizon=1,
    )
    x_last = forecast_data['load'].to_numpy()[-1440:]
    x_last = x_last.reshape(1, 1440, 1)
    X_test = np.concatenate((x_last, X_test), axis=0)

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
    raw_pred = convtrans_model.predict(X_test_scaled)
    # 删除最后一个维度
    raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred

    # 反标准化
    pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
    if len(pred_inverse.shape) > 1 and len(y_test.shape) == 1:
        pred_inverse = pred_inverse.flatten()

    # 计算误差指标
    mae = np.mean(np.abs(y_test[-100:] - pred_inverse[-101:-1]))
    rmse = np.sqrt(np.mean((y_test[-100:] - pred_inverse[-101:-1])**2))
    mape = np.mean(np.abs((y_test[-100:] - pred_inverse[-101:-1]) / y_test[-100:])) * 100

    # 获取最后一个预测值
    data_of_next_min = pred_inverse[-1]

    # 将预测值转换为numpy数组并去掉最后一个维度
    pred_inverse = pred_inverse[:-1].reshape(-1)
    y_test = y_test.reshape(-1)
    data_of_next_min = data_of_next_min.reshape(-1)[0]

    # 平滑
    pred_inverse = convtran_forecasting.apply_adaptive_smoothing(pred_inverse, today_timestamps_origin)

    # 控制小数点位数
    pred_inverse = np.round(pred_inverse, 2)
    y_test = np.round(y_test, 2)
    data_of_next_min = np.round(data_of_next_min, 2)

    return today_timestamps, y_test, pred_inverse, data_of_next_min, round(mae,2), round(rmse,2), round(mape,2)

def refresh_data(date1, date2, date3):
    df_1 = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=date1, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df_2 = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=date2, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df_3 = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=date3, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')

    df_1 = resample_and_pivot(df_1)
    df_2 = resample_and_pivot(df_2)
    df_3 = resample_and_pivot(df_3)

    # 三个数据合并
    df = pd.concat([df_1, df_2, df_3], axis=0)

    # 保存为csv文件
    df.to_csv('iTransformer/data/Data/data.csv', index=False)


def Predict_1(date_str:list, interval:int=5):
    # 更新数据文件
    refresh_data(date_str[0], date_str[1], date_str[2])

    res = convtran_forecasting.main(forecast_start=date_str[0], forecast_end=date_str[0], interval=interval)

    return res
    
    pass

def Predict_UI():
    '''
    预测页面UI
    '''

    # # 获取今天的日期
    today = datetime.now()

    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     # 选择日期
    date_str = str(st.date_input("选择日期", value=today))
    # with col2:
    #     # 选择时间间隔
    #     interval = st.number_input("选择时间间隔(分钟)", value=5, min_value=1, max_value=60, step=1)
    # 获取date_str昨天和前天的日期
    yesterday = datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    day_before_yesterday = datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=2)
    day_before_yesterday_str = day_before_yesterday.strftime('%Y-%m-%d')


    today_timestamps, y_test, pred_inverse, data_of_next_min, mae, rmse, mape = Predict(date_str=date_str)

    dataset = [['时间', '实际值', '预测值']]
    
    for i in range(len(pred_inverse)):
        dataset.append([today_timestamps[i], y_test[i], round(float(pred_inverse[i]),2)])
    if today_timestamps[-1] != '00:00':
        dataset.append([today_timestamps[-1], None, round(float(data_of_next_min),2)])
    
    # 画图
    figure = (
                Line(init_opts=opts.InitOpts(width='1000px'))
                .add_dataset(source=dataset)
                .add_yaxis(series_name='预测值', y_axis=[], encode={'x': '时间', 'y': '预测值'}, is_connect_nones=True, itemstyle_opts=opts.ItemStyleOpts(color='red', opacity=0.7))
                .add_yaxis(series_name='实际值', y_axis=[], encode={'x': '时间', 'y': '实际值'}, is_connect_nones=True, linestyle_opts=opts.LineStyleOpts(width=2))
                

                .set_global_opts(
                    title_opts=opts.TitleOpts(title='日功率曲线'),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                    yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False), type_='value'),
                    xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=100, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False), type_='category'),
                )

                .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=False),
                )
            )
    

    if 'figure' in locals():
        st_echarts(figure, height=container_height-250)

    col1,col2,col3,col4 = st.columns([1,1,1,1])
    with col1:
        Card(
            title= '下一分钟功率值(单位:W)',
            text= [str(data_of_next_min)],
            image= 'Pictures/FDEBAA.png'
        )
    with col2:
        Card(
            title= '平均绝对误差(MAE)',
            text= [str(mae)+'W'],
            image= 'Pictures/E3BBED.png'
        )
    with col3:
        Card(
            title= '均方根误差(RMSE)',
            text= [str(rmse)+'W'],
            image= 'Pictures/BED0F9.png'
        )
    with col4:
        Card(
            title= '平均绝对百分比误差(MAPE)',
            text= [str(mape)+'%'],
            image= 'Pictures/FDEBAA.png'
        )

def Card(title:str, text:list, image:str):
    card(
        title=title,
        text=text,
        image= ReadData.image2base64(image),
        styles={
            'card':{
                'width':'100%',
                'height':'100%',
                # 'margin-top':'-20%',
                # 'margin-bottom':'-20%'
                # 去除margin
                'margin':'0px',
            },
            'text':{
                'color':'white',
                'font-size':'15px'
            },
            'title':{
                'color':'white',
                'font-size':'18px'
            },
            'filter':{
                'background':'rgba(0,0,0,0.4)'
            }

        }
    )


if __name__=='__page__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建Data文件夹，如果有请忽略
    if not os.path.exists('iTransformer\data\Data'):
        os.makedirs('iTransformer\data\Data')


    container_height = 800
    with st.container(border=True, height=container_height):
        st.header("基于iTransformer的超短期负荷预测-总表", help='下述所有误差基于最后100个点计算得出')
        Predict_UI()
        



    pass






