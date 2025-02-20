import pandas as pd
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from Globals import TIME_INTERVAL, NUM_POINTS, CHANGE_LOWER, CHANGE_UPPER
import base64

import os,time
 
# os.environ['TZ'] = 'Asia/Shanghai'
# time.tzset() #Python time tzset() 根据环境变量TZ重新初始化时间相关设置。


def str2timestamp(str_time:str, time_format:str='%Y-%m-%d %H:%M:%S')->int:
    '''
    将年月日时分秒的数据转化为时间戳，未乘1000
    '''
    dt = datetime.datetime.strptime(str_time, time_format)
    return int(dt.timestamp())

def timestamp2str(timestamp:int)->str:
    '''
    将时间戳转换为年月日时分秒，未乘1000
    '''
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def login(PhoneNum:str, Password:str)->str:
    '''
    登录
    '''
    url = 'http://test.beepower.com.cn:30083/jwt/login'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {'loginId': PhoneNum, 'password': Password}
    response = requests.post(url, headers=headers, json=data)
    return eval(response.text)['token']

@st.cache_data(ttl=TIME_INTERVAL*60)
def ReadData_Day(beeId:str, mac:str, time:str, PhoneNum:str, password:str, DataType:str='P')->pd.DataFrame:
    '''
    查询某个设备某天的数据，需要指定网关和设备的mac，以及提供登陆的手机号和密码

    Parameters
    beeId : str
        网关的beeId
    mac : str
        设备的mac
    PhoneNum : str
        登陆的手机号
    Password : str
        登陆的密码
    time : str
        查询的时间，格式为'%Y-%m-%d'
    DataType : str
        查询的数据类型，P为功率，E为电量
    '''
    def str2int(x):
        x['TimeStamp'] = int(int(x['TimeStamp'])/1000)
        x[DataType] = float(x[DataType])*1.0

    time = time+'~'+timestamp2str(str2timestamp(time+' 00:00:00', '%Y-%m-%d %H:%M:%S')+86400)[0:10]
    # 2024 01-07~2024-01-08
    url = f'http://test.beepower.com.cn:30083/api/mqtt/v1?beeId={beeId}&methodType=query3&phone={PhoneNum}'

    token = login(PhoneNum, password)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = f'curve.{mac}.{DataType}:{time}'
    response = requests.post(url, headers=headers, data=data)
    # text = eval(response.text)
    # 把eval函数换成了json函数，观察后期会不会报错
    text = response.json()
    if f'curve.{mac}.{DataType}:{time}' in text.keys():
        data = text[f'curve.{mac}.{DataType}:{time}']
        # 把数据转化为dataframe，第一列是时间，第二列是数据
        data = np.array([
            list(data.keys()),
            list(data.values())
        ]).T
        df = pd.DataFrame(data, columns=['TimeStamp', DataType])
        df.apply(str2int, axis=1)
        # 个根据时间戳大小从小到大排序
        df = df.sort_values(by='TimeStamp', ignore_index=True)
        # 添加一列Time，把时间戳转化为年月日时分秒
        df['Time'] = df['TimeStamp'].apply(timestamp2str)
        return df
    else:
        print('未查询到有效数据，已返回空表。')
        return pd.DataFrame(columns=['TimeStamp', 'Time', DataType])

@st.cache_data(ttl=TIME_INTERVAL*60)
def TimeIntervalTransform(df:pd.DataFrame, date:str, time_interval:int=TIME_INTERVAL, DataType:str='P'):
    '''
    将时间间隔转化为指定的时间间隔
    '''
    # 生成date日按照time_interval的时间间隔的时间戳列表
    timestamp_list = [str2timestamp(date+' 00:00:00', '%Y-%m-%d %H:%M:%S')+i*time_interval*60 for i in range(24*60//time_interval)]
    # 根据时间戳列表生成时间字符串列表
    time_list = [timestamp2str(i) for i in timestamp_list]
    df_temp = pd.DataFrame(columns=['TimeStamp', 'Time'])
    df_temp['TimeStamp'] = timestamp_list
    df_temp['Time'] = time_list

    # 如果df为空，则用0填充
    if df.empty:
        df = pd.DataFrame(columns=['TimeStamp', DataType])
        df['TimeStamp'] = timestamp_list
        df[DataType] = [0 for i in range(len(timestamp_list))]
        df['Time'] = time_list
        return df
    
    # 接下来根据datatype分别处理功率和电量数据
    if DataType=='P' or DataType=='P2Energy':
        data = []
        for i in range(len(timestamp_list)):
            # 选取时间戳在timestamp_list[i]和timestamp_list[i+1]之间的数据
            temp = df[(df['TimeStamp']>=timestamp_list[i]) & (df['TimeStamp']<timestamp_list[i]+time_interval*60)]
            # 如果temp为空，则用np.nan填充
            if temp.empty:
                data.append(np.nan)
            else:
                # 使用积分平均值
                temp_timetsamp = temp['TimeStamp'].to_numpy()
                temp_P = temp['P'].to_numpy()
                temp_energy = np.trapezoid(temp_P, temp_timetsamp)+temp_P[0]*(temp_timetsamp[0]-timestamp_list[i])+temp_P[-1]*(timestamp_list[i]+time_interval*60-temp_timetsamp[-1])
                data.append(round(temp_energy/(time_interval*60),2))
        data = pd.Series(data)
        # 对data进行处理，如果data中有nan，则采用线性插值
        if date == str(datetime.datetime.now().date()):
            # 前向用线性插值，后面用0填充
            data.interpolate(inplace=True, limit_direction='backward', method='linear')
            data.fillna(0, inplace=True)
        else:
            data.interpolate(inplace=True, limit_direction='both', method='linear')
        if DataType=='P':
            df_temp['P'] = np.round(data.to_numpy(), 2)
        elif DataType=='P2Energy':
            df_temp['Energy'] = np.round(data.to_numpy()/(60/time_interval)/1000, 2)
    elif DataType=='Energy':
        #计算每一个区间的用电量，没有就是0
        data = []
        for i in range(len(timestamp_list)):
            temp = df[(df['TimeStamp']>=timestamp_list[i]) & (df['TimeStamp']<timestamp_list[i]+time_interval*60)]
            if temp.empty:
                data.append(0)
            else:
                data.append(temp['Energy'].max()-temp['Energy'].min())
        df_temp['Energy'] = np.round(data, 2)
    else:
        data = []
        for i in range(len(timestamp_list)):
            temp = df[(df['TimeStamp']>=timestamp_list[i]) & (df['TimeStamp']<=timestamp_list[i]+time_interval*60)]
            if temp.empty:
                data.append(0)
            else:
                temp_timetsamp = temp['TimeStamp'].to_numpy()
                temp_data = temp[DataType].to_numpy()
                temp_energy = np.trapezoid(temp_data, temp_timetsamp)+temp_data[0]*(temp_timetsamp[0]-timestamp_list[i])+temp_data[-1]*(timestamp_list[i]+time_interval*60-temp_timetsamp[-1])
                data.append(round(temp_energy/(time_interval*60),2))
        data = pd.Series(data)
        if date == str(datetime.datetime.now().date()):
            data.interpolate(inplace=True, limit_direction='backward', method='linear')
            data.fillna(0, inplace=True)
        else:
            data.interpolate(inplace=True, limit_direction='both', method='linear')
        df_temp[DataType] = np.round(data.to_numpy(), 2)
    return df_temp

@st.cache_data(ttl=TIME_INTERVAL*60)
def ReadWeather(date:str=None):
    '''
    获取天气信息
    Parameters
    date : str
        查询的日期，格式为'%Y-%m-%d'
    Returns
    -------
    str
        天气
    float
        温度
    float
        湿度
    '''
    if (not date) or date == str(datetime.datetime.now().date()):
        # 如果没有指定日期，则返回当天的天气
        url = 'https://api.weatherapi.com/v1/current.json?key=4ac8a9797998437d985122700242412&q=ShangHai/MinHang&aqi=no'
        response = requests.get(url)
        # 加载成字典
        data = response.json()
        # 返回天气和温度
        return data['current']['condition']['text'],data['current']['temp_c'],data['current']['humidity']
    else:
        # 如果指定了日期，则返回指定日期的天气
        url = 'https://api.weatherapi.com/v1/history.json?key=4ac8a9797998437d985122700242412&q=ShangHai&dt='+date
        response = requests.get(url)
        data = response.json()
        return data['forecast']['forecastday'][0]['day']['condition']['text'],data['forecast']['forecastday'][0]['day']['avgtemp_c'],data['forecast']['forecastday'][0]['day']['avghumidity']

def image2base64(image_path:str)->str:
    '''
    将图片转化为base64编码
    '''
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data)
    return "data:image/png;base64," + encoded.decode("utf-8")

@st.cache_data(ttl=TIME_INTERVAL*60)
def ReadInnerTemperature(PhoneNum:str, Password:str, date:str=None):
    '''
    获取室内温度
    '''
    mac = 'Irc-M1-7cdfa1b89d38'
    if (not date) or date == str(datetime.datetime.now().date()):
        date = str(datetime.datetime.now().date())
        df = ReadData_Day(beeId='86200001289', mac=mac, time=date, PhoneNum=PhoneNum, password=Password, DataType='Temperature')
        # 返回最后一个数
        if df.empty:
            return 0.0
        return df['Temperature'].iloc[-1]
    else:
        df = ReadData_Day(beeId='86200001289', mac=mac, time=date, PhoneNum=PhoneNum, password=Password, DataType='Temperature')
        return df['Temperature'].iloc[-1]
@st.cache_data(ttl=TIME_INTERVAL*60)
def find_change_point(data, change_lower=CHANGE_LOWER, change_upper=CHANGE_UPPER, num_points=NUM_POINTS):
    # 这里默认是96个点
    length = data.shape[0]
    # 求差分
    diff = np.diff(data)
    for i in range(length-num_points-1):
        if_point = True
        for j in range(num_points):
            if abs(diff[i+j])>change_lower:
                if_point = False
        if diff[i+num_points]<change_upper:
            if_point = False
        if if_point:
            return i+num_points
    return 0

@st.cache_data(ttl=TIME_INTERVAL*60)
def Each_Weekday(date:str=None):
    '''
    判断星期几
    '''
    week_day_dict = {'Monday':'星期一', 'Tuesday':'星期二', 'Wednesday':'星期三', 'Thursday':'星期四', 'Friday':'星期五', 'Saturday':'星期六', 'Sunday':'星期日'}
    if (not date) or date == str(pd.to_datetime('today').date()):
        week_day = time.strftime('%A', time.localtime())
    else:
        week_day = pd.to_datetime(date).day_name()
    return week_day_dict[week_day]


def ReadData_RealTime(beeId:str, PhoneNum:str, password:str, DataType:str='P'):
    '''
    查询某个网关下所有设备的实时功率
    '''
    url = f'http://test.beepower.com.cn:30083/api/mqtt/v1?beeId={beeId}&methodType=query3&phone={PhoneNum}'
    token = login(PhoneNum, password)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = 'state.'+DataType
    response = requests.post(url, headers=headers, data=data)
    text = eval(response.text)
    if f'state.{DataType}' in text.keys():
        return text[f'state.{DataType}']
    else:
        return None

if __name__ == '__main__':
    phone_num = '15528932507'
    password = '123456'
    date = '2025-01-07'
    mac = 'Sck-M1-7cdfa1b89c00'

    df = ReadData_Day(beeId='86200001289', mac='SckM17cdfa1b89c00', time=date, PhoneNum=phone_num, password=password, DataType='P')
    df.plot(x='Time', y='P')

    plt.show()
    pass

















