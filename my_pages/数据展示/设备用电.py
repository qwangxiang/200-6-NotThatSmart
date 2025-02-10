import streamlit as st
from Globals import PHONE_NUM, PASSWORD, beeID, devices
from utils import ReadData
import datetime
import numpy as np

def Def_CSS():
    '''
    定义CSS样式
    '''
    # 卡片样式
    st.markdown(
        """
        <style>
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
        </style>
        """,
        unsafe_allow_html=True
    )

def Printer():
    '''
    打印机卡片
    '''
    # 获取打印机今日数据
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['打印机'], date, PhoneNum, password, 'P')
    
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

def Subrouter():
    '''
    子路由器卡片
    '''
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['华为子路由器'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['子路由器']}');">
                <div class="device-name">子路由器</div>
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
            <div class="card" style="background-image: url('{links['子路由器']}');">
                <div class="device-name">子路由器</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    pass

def Fridge():
    '''
    冰箱卡片
    '''
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['冰箱'], date, PhoneNum, password, 'P')
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

def Network_Device():
    '''
    网络设备卡片
    '''
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['网络设备'], date, PhoneNum, password, 'P')
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

def Coffee_Machine():
    '''
    咖啡机卡片
    '''
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['咖啡机'], date, PhoneNum, password, 'P')
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

def Kettle():
    '''
    烧水壶卡片
    '''
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['烧水壶'], date, PhoneNum, password, 'P')
    if data.empty:
        st.markdown(
            f"""
            <div class="card grayscale" style="background-image: url('{links['烧水壶']}');">
                <div class="device-name">烧水壶</div>
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
            <div class="card" style="background-image: url('{links['烧水壶']}');">
                <div class="device-name">烧水壶</div>
                <div class="device-power">实时功率 <br> <span style="color: {color};">{realtime_power} W</span> <br> 今日用电量 <br> {power_use_today} kWh </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def Microwave_Oven():
    '''
    微波炉卡片
    '''
    date = str(datetime.datetime.now().date())
    data = ReadData.ReadData_Day(beeID['生活区'], devices['生活区']['微波炉'], date, PhoneNum, password, 'P')
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


def Show_Devices():
    '''
    展示所有的设备
    '''    
    with st.container(border=True):
        col1_1,col1_2,col1_3,col1_4 = st.columns([1,1,1,1])
        with col1_1:
            Printer()
        with col1_2:
            Subrouter()
        with col1_3:
            Fridge()
        with col1_4:
            Network_Device()
        col2_1,col2_2,col2_3 = st.columns([1,1,1])
        with col2_1:
            Coffee_Machine()
        with col2_2:
            Kettle()
        with col2_3:
            Microwave_Oven()
        
def Show_Induction():
    '''
    展示所有的人体感应器
    '''
    Inductoions = {'生活区':'86200001187', '进门人体感应':'Irs-M1-84f703112028', '展示区人体感应':'Irs-M1-84f703122288', '小会议室人体感应':'Irs-M1-84f7031218b4', '人体感应1':'Irs-M1-7cdfa1b84cb4', '人体感应2':'Irs-M1-7cdfa1b85e28', '大会议室人体感应器':'Irs-M1-84f703101f5c', '办公室B人体感应器':'Irs-M1-84f70310d0f4', '办公室C人体感应器':'Irs-M1-7cdfa1b85e50'}
    pass

def Show_Air_Condition():
    '''
    展示所有的空调
    '''
    with st.container(border=True):
        
        pass

if __name__=='__page__':
    # 账户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD
    links = {
        'Printer':'https://img.picui.cn/free/2025/02/02/679efa44cbd6b.jpg',
        '子路由器':'https://img.picui.cn/free/2025/02/02/679f00d6b5783.jpg',
        '冰箱':'https://img.picui.cn/free/2025/02/02/679f01d0ae73f.jpg',
        '网络设备':'https://img.picui.cn/free/2025/02/02/679f02f7560ac.jpg',
        '咖啡机':'https://img.picui.cn/free/2025/02/02/679f05e5ddcb3.jpg',
        '烧水壶':'https://img.picui.cn/free/2025/02/02/679f08ddf2959.jpg',
        '微波炉':'https://img.picui.cn/free/2025/02/02/679f0992e0509.jpg',
    }

    

    # P: 打印机、子路由器、冰箱、网络设备、咖啡壶。烧水壶、微波炉
    # Induction: 进门人体感应、展示区人体感应、小会议室人体感应、人体感应1、人体感应2、大会议室人体感应器、办公室B人体感应器、办公室C人体感应器
    

    st.title('设备用电')

    # 定义CSS样式
    Def_CSS()

    tab1,tab2,tab3 = st.tabs(['设备', '空调', '人体感应器'])

    with tab1:
        Show_Devices()
    with tab2:
        pass
    with tab3:
        pass


