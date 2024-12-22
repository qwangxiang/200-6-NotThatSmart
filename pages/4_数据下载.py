import streamlit as st
from utils import ReadData


if __name__ == '__main__':
    st.title('数据下载')

    # 选择网关
    gateway = st.selectbox('选择网关', ['生活区/进门区', '办公室/会议室', '学生办公区'])
    gateway_dict = {'生活区/进门区': '86200001187', '办公室/会议室': '86200001183', '学生办公区': '86200001289'}
    BeeID = gateway_dict[gateway]

    mac = st.text_input('输入mac地址', 'Mt3-M1-84f703120b64')

    # 选择日期
    date = str(st.date_input('选择日期', value='today', min_value=None, max_value=None, key=None))

    # 选择数据类型
    DataType = st.selectbox('选择数据类型', ['功率(W)', '电量(kWh)'])
    DataType_dict = {'功率(W)': 'P', '电量(kWh)': 'Energy'}
    DataType = DataType_dict[DataType]

    # if st.button('下载数据'):
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum='15528932507', password='123456', DataType=DataType)

    col1,col2,col3 = st.columns([0.4,0.2,0.4])
    with col2:
        st.download_button(label='下载数据', data=df.to_csv(encoding='utf-8-sig'), file_name='data.csv', mime='text/csv')
    # 下载按钮居中
    

    pass























