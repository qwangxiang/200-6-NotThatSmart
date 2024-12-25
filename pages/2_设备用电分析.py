import streamlit as st
from utils import ReadData
from Globals import PHONE_NUM, PASSWORD
import pandas as pd
import matplotlib.pyplot as plt

def ShowDeviceUse():
    global date

    col1,col2,col3 = st.columns([3,2,5])
    with col1:
        # 选择日期
        date = str(st.date_input('选择日期', value='today'))
    with col2:
        # 选择区域
        area = st.selectbox('选择区域', ['生活区', '学生办公区', '办公室/会议室'])
    with col3:
        # 选择设备
        device = st.selectbox('选择设备', list(devices[area].keys()))
    
    beeId = beeID[area]
    mac = devices[area][device]
    Datatype = 'Induction' if 'Irs' in mac else 'P'
    if Datatype == 'P':
        df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType=Datatype)
        # df = ReadData.TimeIntervalTransform(df, date, time_interval=15, DataType=Datatype)
        df['datetime'] = pd.to_datetime(df['TimeStamp'], unit='s')
        figure,ax = plt.subplots(figsize=(6,6))
        df.plot(x='datetime', y='P', ax=ax)
        st.write(figure)
    pass


if __name__ == '__main__':

    # 账户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD

    # 其他变量
    beeID = {
        '生活区': '86200001187',
        '学生办公区': '86200001289',
        '办公室/会议室': '86200001290',
    }
    devices = {
        '生活区': {
            '打印机': 'Sck-M1-84f703123a18',
            '进门灯': 'Lk3-M1-7cdfa1b8a1a0',
            '小会议室灯': 'Lk1-M1-7cdfa1b8640c',
            '进门人体感应': 'Irs-M1-84f703112028',
            '展示区人体感应': 'Irs-M1-84f703122288',
            '小会议室人体感应': 'Irs-M1-84f7031218b4',
            '华为子路由器': 'Sck-M1-7cdfa1b660dc',
            '冰箱': 'Sck-M1-7cdfa1b852e0',
            '网络设备': 'Sck-M1-7cdfa1b89d5',
            # 这里实际上咖啡机和烧水壶的mac示范的
            '咖啡机': 'Sck-M1-84f703123c88',
            '烧水壶': 'Sck-M1-7cdfa1b89d20',
            # 这个mac也是改过的
            '微波炉': 'Sck-M1-84f70310ee40',
        },
        '学生办公区': {
            '人体感应1': 'Irs-M1-7cdfa1b84cb4',
            '人体感应2': 'Irs-M1-7cdfa1b85e28',
        },
        '办公室/会议室': {
            # 这几个人感不知道为什么查不到数据
            '大会议室人体感应器': 'Irs-M1-84f703101f5c',
            '办公室B人体感应器': 'Irs-M1-84f70310d0f4',
            '办公室B灯': 'Lk1-M1-7cdfa1b867d8',
            '办公室C人体感应器': 'Irs-M1-7cdfa1b85e50',
            '办公室C灯': 'Lk1-M1-7cdfa1b87b18',
            '办公室C空气传感器': 'Env-lt_0004@modbus 01',
            '办公室D灯': 'Lk1-M1-7cdfa1b84bc4',
            '办公室D空气传感器': 'Env-lt_0001@modbus 01',
            '大会议室灯': 'Lk2-M1-7cdfa1b86a10',
        }
    }

    date = None

    st.set_page_config(layout='wide')
    st.title('设备用电分析')
    with st.container(border=True):
        ShowDeviceUse()
    



























