import streamlit as st
from utils import ReadData
import pandas as pd


if __name__ == '__page__':
    st.title('数据下载')

    # 选择网关
    beeID = st.selectbox('选择网关', ['86200001187', '86200001183', '86200001289'])

    # 选择设备
    mac = st.text_input('输入mac地址', 'Mt3-M1-84f703120b64')

    # 选择日期
    col1, col2, = st.columns([0.5, 0.5])
    with col1:
        start_date = str(st.date_input('选择开始日期', value='today', min_value=None, max_value=None, key=None))
    with col2:
        end_date = str(st.date_input('选择结束日期', value='today', min_value=None, max_value=None, key=None))
    # 判断结束日期是否在开始日期之后
    if not pd.to_datetime(start_date) <= pd.to_datetime(end_date):
        st.error('结束日期必须在开始日期之后')
        st.stop()

    # 生成从start_date到end_date的日期列表
    date = [start_date]
    while date[-1] != end_date:
        date.append(str(pd.to_datetime(date[-1]) + pd.Timedelta(days=1))[:10])
    
    # 选择数据类型
    DataType = st.selectbox('选择数据类型', ['功率(W)', '电量(kWh)'])
    DataType_dict = {'功率(W)': 'P', '电量(kWh)': 'Energy'}
    DataType = DataType_dict[DataType]

    num_dates = len(date)
    df = ReadData.ReadData_Day(beeId=beeID, mac=mac, time=date[0], PhoneNum='15528932507', password='123456', DataType=DataType)
    for i in range(1, num_dates):
        df_ = ReadData.ReadData_Day(beeId=beeID, mac=mac, time=date[i], PhoneNum='15528932507', password='123456', DataType=DataType)
        df = pd.concat([df, df_])

    col1,col2,col3,col4 = st.columns([0.25,0.25,0.25,0.25])
    with col2:
        with open('DeviceInfo.txt', 'rb') as f:
            data = f.read()
        st.download_button(label='mac文件下载', data=data, file_name='DeviceInfo.txt', mime='text/plain', help='下载mac地址参考文件')
    with col3:
        st.download_button(label='下载数据', data=df.to_csv(encoding='utf-8-sig'), file_name='data.csv', mime='text/csv', help='根据上述选择下载数据')
            
    # 下载按钮居中
    

    pass























