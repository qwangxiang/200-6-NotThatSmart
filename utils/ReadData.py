import pandas as pd
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st

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

# @st.cache_data
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
    url = f'http://test.beepower.com.cn:30083/api/mqtt/v1?beeId={beeId}&methodType=query3&phone={PhoneNum}'

    token = login(PhoneNum, password)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = f'curve.{mac}.{DataType}:{time}'
    response = requests.post(url, headers=headers, data=data)
    text = eval(response.text)
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
        return pd.DataFrame(columns=['TimeStamp', DataType])

@st.cache_data
def TimeIntervalTransform(df:pd.DataFrame, date:str, time_interval:int=15, DataType:str='P'):
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
        df['P' if DataType=='P' else 'Energy'] = [0 for i in range(len(timestamp_list))]
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
                # 使用算数平均值
                # data.append(temp['P'].mean())
                # 使用积分平均值
                temp_timetsamp = temp['TimeStamp'].to_numpy()
                temp_P = temp['P'].to_numpy()
                temp_energy = np.trapezoid(temp_P, temp_timetsamp)+temp_P[0]*(temp_timetsamp[0]-timestamp_list[i])+temp_P[-1]*(timestamp_list[i]+time_interval*60-temp_timetsamp[-1])
                data.append(temp_energy/(time_interval*60))
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
    return df_temp


if __name__ == '__main__':
    phone_num = '15528932507'
    password = '123456'
    BeeID = '86200001289'
    BeeID2 = '86200001187'
    mac = 'Sck-M1-7cdfa1b85f04'
    time = '2024-12-22'
    people_detector = 'Irs-M1-7cdfa1b85e28'
    all_mac = 'Mt3-M1-84f703120b64'

    df = ReadData_Day(beeId=BeeID2, mac=all_mac, time=time, PhoneNum=phone_num, password=password, DataType='P')
    print(df)

    df = TimeIntervalTransform(df, time, time_interval=15, DataType='P')
    print(df[40:])

    plt.show()
    # print(type(df['TimeStamp'][0]))
















