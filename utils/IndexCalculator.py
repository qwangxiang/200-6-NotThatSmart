import numpy as np
from utils import ReadData
from Globals import TIME_INTERVAL
import streamlit as st

@st.cache_data(ttl=TIME_INTERVAL*60)
def PeaKAndValley(beeId:str, mac:str, date:str, PhoneNum:str, Password:str):
    '''
    计算负荷峰谷值、峰值时间，峰谷差
    Returns:
        float: 峰值
        str: 峰值时间
        float: 谷值
        float: 峰谷差
    '''
    df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=Password, DataType='P')
    df = ReadData.TimeIntervalTransform(df, date=date, time_interval=TIME_INTERVAL)
    time_list = time_list = [str(i)[-8:] for i in df['Time'].tolist()]
    peak = df['P'].max()
    peak_time = time_list[df['P'].argmax()]
    valley = df['P'].min()
    peak_valley_diff = peak-valley
    return peak, peak_time, valley, peak_valley_diff

@st.cache_data(ttl=TIME_INTERVAL*60)
def Varibility(beeId:str, mac:str, date:str, PhoneNum:str, Password:str):
    '''
    计算负荷波动性
    Returns:
        float: 负荷波动性指标，即标准差
    '''
    df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=Password, DataType='P')
    if df.empty:
        return 0.
    else:
        data = df['P'].to_numpy()
        diff = np.abs(np.diff(data))
        return np.mean(diff), np.max(diff), np.min(diff)


@st.cache_data(ttl=TIME_INTERVAL*60)
def LoadFactor(beeId:str, mac:str, date:str, PhoneNum:str, Password:str):
    '''
    计算负荷因子
    Returns:
        float: 负荷因子
    '''
    df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=Password, DataType='P')
    df = ReadData.TimeIntervalTransform(df, date=date, time_interval=TIME_INTERVAL)
    return np.sum(df['P'])/(np.max(df['P'])*len(df['P']))

@st.cache_data(ttl=TIME_INTERVAL*60)
def RisingEdgeAndFallingEdge(beeId:str, mac:str, date:str, PhoneNum:str, Password:str):
    '''
    计算最大上升沿和最大下降沿
    '''
    df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=Password, DataType='P')
    df = ReadData.TimeIntervalTransform(df, date=date, time_interval=TIME_INTERVAL)
    time_list = [str(i)[-8:] for i in df['Time'].tolist()]
    data = df['P'].to_numpy()
    data = np.diff(data)
    rising_edge = np.max(data)
    falling_edge = np.min(data)
    rising_edge_time = time_list[np.argmax(data)]
    falling_edge_time = time_list[np.argmin(data)]
    return rising_edge, rising_edge_time, falling_edge, falling_edge_time



