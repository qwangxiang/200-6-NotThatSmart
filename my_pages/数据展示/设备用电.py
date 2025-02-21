import streamlit as st
from Globals import PHONE_NUM, PASSWORD, Inductions, TIME_INTERVAL, devices_lib
from utils import ReadData
import datetime
import numpy as np

from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=TIME_INTERVAL * 60 * 1000, key="autorefresh")

def Def_CSS():
    '''
    定义CSS样式
    '''
    # 卡片样式
    st.markdown(
        """
        <style>
        /* 卡片CSS */
        .card {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2vw;
            border-radius: 1.5vw;
            background-size: cover;
            background-position: center;
            margin-bottom: 2vw;
            color: black;
            font-family: 'Microsoft YaHei', sans-serif; /* 改成微软雅黑 */
            border: 0.2vw solid #ccc; /* 添加边框 */
            box-shadow: 0 0.4vw 0.8vw rgba(0, 0, 0, 0.2); /* 添加阴影 */
            height: 15vw; /* 设置卡片高度 */
            transition: transform 0.5s ease, box-shadow 0.5s ease; /* 添加过渡效果 */
        }

        .card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2); /* 半透明白色蒙版 */
            border-radius: 1.5vw;
            z-index: 1;
        }

        .card .content {
            position: relative;
            z-index: 2;
        }

        .card.grayscale {
            filter: grayscale(100%);
        }

        .card:hover {
            transform: scale(1.02); /* 鼠标悬停时放大 */
            box-shadow: 0 0.8vw 1.6vw rgba(0, 0, 0, 0.2); /* 鼠标悬停时阴影加深 */
        }

        .card .device-name {
            position: absolute;
            top: 1vw;
            left: 2vw;
            font-size: 2vw;
            font-weight: bold;
        }

        .card .device-power {
            position: absolute;
            bottom: 1vw;
            right: 2vw;
            font-size: 1vw;
            font-weight: bold;
            text-align: center; /* 居中对齐 */
        }


        /* Induction_Button */
        .induction-button {
            position: absolute;
            width: 1vw; /* 半径大小和父组件的宽度成比例 */
            height: 1vw; /* 半径大小和父组件的宽度成比例 */
            border-radius: 50%; /* 圆形 */
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            transition: transform 0.3s ease;
            z-index: 3;
        }

        .induction-button:hover {
            transform: scale(1.1); /* 鼠标悬浮时放大 */
        }

        .induction-button:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }

        .tooltip {
            position: absolute;
            top: 50%; /* 调整位置 */
            left: 110%;
            transform: translateY(-50%);
            background-color: #333;
            color: #fff;
            text-align: center;
            padding: 0.5vw;
            border-radius: 0.5vw;
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s ease;
            font-family: 'KaiTi', serif; /* 楷体 */
            font-size: 1vw; /* 字号和组件半径成比例 */
            white-space: nowrap; /* 使对话框横向显示 */
        }

        .tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateY(-50%);
            border-width: 0.5vw;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def Printer(date):
    '''
    打印机卡片
    '''
    # 获取打印机今日数据
    data = ReadData.ReadData_Day(devices_lib['打印机']['beeID'], devices_lib['打印机']['mac'], date, PhoneNum, password, 'P')
    
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['Printer']}');">
                <div class="device-name">打印机</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 找到有几个大于500的变化点
        use_nums = np.sum(data_diff>=500)
        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['Printer']}');">
                <div class="device-name">打印机</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日使用次数 <br> <span style="color: blue;">{use_nums}次</span> </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Fridge(date):
    '''
    冰箱卡片
    '''
    data = ReadData.ReadData_Day(devices_lib['冰箱']['beeID'], devices_lib['冰箱']['mac'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['冰箱']}');">
                <div class="device-name">冰箱</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        time = data['TimeStamp'].to_numpy()
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 积分算出今日用电量
        power_use_today = round(np.trapezoid(data, time)/1000/3600,2)

        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['冰箱']}');">
                <div class="device-name">冰箱</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Network_Device(date):
    '''
    网络设备卡片
    '''
    data = ReadData.ReadData_Day(devices_lib['网络设备']['beeID'], devices_lib['网络设备']['mac'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['网络设备']}');">
                <div class="device-name">网络设备</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        time = data['TimeStamp'].to_numpy()
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 积分算出今日用电量
        power_use_today = round(np.trapezoid(data, time)/1000/3600,2)

        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['网络设备']}');">
                <div class="device-name">网络设备</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Coffee_Machine(date):
    '''
    咖啡机卡片
    '''
    data = ReadData.ReadData_Day(devices_lib['咖啡机']['beeID'], devices_lib['咖啡机']['mac'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['咖啡机']}');">
                <div class="device-name">咖啡机</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        time = data['TimeStamp'].to_numpy()
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 积分算出今日用电量
        power_use_today = round(np.trapezoid(data, time)/1000/3600,2)

        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['咖啡机']}');">
                <div class="device-name">咖啡机</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Microwave_Oven(date):
    '''
    微波炉卡片
    '''
    data = ReadData.ReadData_Day(devices_lib['微波炉']['beeID'], devices_lib['微波炉']['mac'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['微波炉']}');">
                <div class="device-name">微波炉</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        time = data['TimeStamp'].to_numpy()
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 积分算出今日用电量
        power_use_today = round(np.trapezoid(data, time)/1000/3600,2)

        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['微波炉']}');">
                <div class="device-name">微波炉</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Display_1(date):
    '''
    展示区电视卡片
    '''
    data = ReadData.ReadData_Day(devices_lib['展示区电视']['beeID'], devices_lib['展示区电视']['mac'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['展示区电视']}');">
                <div class="device-name">展示区电视</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        time = data['TimeStamp'].to_numpy()
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 积分算出今日用电量
        power_use_today = round(np.trapezoid(data, time)/1000/3600,2)

        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['展示区电视']}');">
                <div class="device-name">展示区电视</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Display_2(date):
    '''
    大会议室电视卡片
    '''
    data = ReadData.ReadData_Day(devices_lib['大会议室电视']['beeID'], devices_lib['大会议室电视']['mac'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['大会议室电视']}');">
                <div class="device-name">大会议室电视</div>
                <div class="content" style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div class="device-status" style="text-align: center; font-size: 18px; color: red; font-weight: bold;">设备离线...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        time = data['TimeStamp'].to_numpy()
        data = data['P'].to_numpy()
        realtime_power = round(data[-1],2)
        data_diff = np.diff(data)
        color = 'red' if data_diff[-1]>0 else 'green'
        # 积分算出今日用电量
        power_use_today = round(np.trapezoid(data, time)/1000/3600,2)

        st.markdown(
            f"""
            <div class="card" style="background-image: url('{links['大会议室电视']}');">
                <div class="device-name">大会议室电视</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )



def Show_Devices():
    '''
    展示所有的设备
    '''    
    date = str(datetime.datetime.now().date())
    with st.container(border=True):
        col1_1,col1_2,col1_3,col1_4 = st.columns([1,1,1,1])
        with col1_1:
            Printer(date)
        with col1_2:
            Display_1(date)
        with col1_3:
            Fridge(date)
        with col1_4:
            Network_Device(date)
        col2_1,col2_2,col2_3 = st.columns([1,1,1])
        with col2_1:
            Coffee_Machine(date)
        with col2_2:
            Microwave_Oven(date)
        with col2_3:
            Display_2(date)

@st.cache_data(ttl=TIME_INTERVAL*60)
def Single_Induction(name:str):
    '''
    展示单个人体传感器的状态，其中：
    0---离线---灰色
    1---无人---红色
    2---有人---绿色
    '''
    color = {
        0:'gray',
        1:'red',
        2:'green'
    }
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(Inductions[name]['beeID'], Inductions[name]['mac'], date, PhoneNum, password, 'Induction')
    res = name+'<br>状态：'
    if data.empty:
        status = 0
        res += '离线'
    else:
        status = data['Induction'].to_numpy()[-1]+1
        res += '有人' if status==2 else '无人'
        res += ' 时间：'+data['Time'].to_numpy()[-1][-8:]

    return color[status], res

def Show_Induction():
    '''
    展示所有的人体感应器
    '''
    names = list(Inductions.keys())

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <div class="card" style="width: 85%; padding-bottom: 43%; position: relative; transition: none; box-shadow: none; transform: none;">
                <img src="https://img.picui.cn/free/2025/02/10/67a9be749980e.png" style="width: 100%; height: 100%; object-fit: contain; position: absolute; top: 0; left: 0; border-radius: 0.0vw;">
                <div class="induction-button" style="background-color: {Single_Induction(names[0])[0]}; top: 50%; left: 14%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[0])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[1])[0]}; top: 15%; left: 14%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[1])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[2])[0]}; top: 80%; left: 16%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[2])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[3])[0]}; top: 88%; left: 30%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[3])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[4])[0]}; top: 88%; left: 65%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[4])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[5])[0]}; top: 88%; left: 83%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[5])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[6])[0]}; top: 50%; left: 80%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[6])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[7])[0]}; top: 16%; left: 62%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[7])[1]}</span>
                </div>
                <div class="induction-button" style="background-color: {Single_Induction(names[8])[0]}; top: 55%; left: 62%; transform: translate(-50%, -50%);">
                    <span class="tooltip">{Single_Induction(names[8])[1]}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pass



if __name__=='__page__':
    # 账户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD
    links = {
        'Printer':'https://img.picui.cn/free/2025/02/02/679efa44cbd6b.jpg',
        '冰箱':'https://img.picui.cn/free/2025/02/02/679f01d0ae73f.jpg',
        '网络设备':'https://img.picui.cn/free/2025/02/02/679f02f7560ac.jpg',
        '咖啡机':'https://img.picui.cn/free/2025/02/02/679f05e5ddcb3.jpg',
        '微波炉':'https://img.picui.cn/free/2025/02/19/67b543978b55d.jpg',
        '大会议室电视':'https://img.picui.cn/free/2025/02/20/67b6915dc5849.jpg',
        '展示区电视':'https://img.picui.cn/free/2025/02/20/67b6925dcc60d.jpg'
    }

    st.title('设备用电')

    # 定义CSS样式
    Def_CSS()

    tab1,tab2 = st.tabs(['设备', '人体感应器'])

    with tab1:
        Show_Devices()
    with tab2:
        Show_Induction()

