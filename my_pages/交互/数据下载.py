import streamlit as st
from utils import ReadData


if __name__ == '__page__':
    st.title('数据下载')

    # 选择网关
    beeID = st.selectbox('选择网关', ['86200001187', '86200001183', '86200001289'])

    # 选择设备
    mac = st.text_input('输入mac地址', 'Mt3-M1-84f703120b64')

    # 选择日期
    date = str(st.date_input('选择日期', value='today', min_value=None, max_value=None, key=None))

    # 选择数据类型
    DataType = st.selectbox('选择数据类型', ['功率(W)', '电量(kWh)'])
    DataType_dict = {'功率(W)': 'P', '电量(kWh)': 'Energy'}
    DataType = DataType_dict[DataType]

    # if st.button('下载数据'):
    df = ReadData.ReadData_Day(beeId=beeID, mac=mac, time=date, PhoneNum='15528932507', password='123456', DataType=DataType)

    col1,col2,col3,col4 = st.columns([0.25,0.25,0.25,0.25])
    with col2:
        with open('DeviceInfo.txt', 'rb') as f:
            data = f.read()
        st.download_button(label='mac文件下载', data=data, file_name='DeviceInfo.txt', mime='text/plain', help='下载mac地址参考文件')
    with col3:
        st.download_button(label='下载数据', data=df.to_csv(encoding='utf-8-sig'), file_name='data.csv', mime='text/csv', help='根据上述选择下载数据')
            
    # 下载按钮居中
    

    pass























