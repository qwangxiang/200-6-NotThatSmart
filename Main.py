import streamlit as st
from streamlit_extras.app_logo import add_logo
from streamlit_extras.card import card
import base64
from utils import ReadData


if __name__ == '__main__':
    # st.title('200-6数据分析器')
    # 标题居中
    st.markdown("<h1 style='text-align: center;'>200-6简易数据分析</h1>", unsafe_allow_html=True)
    st.balloons()

    mac = 'Irc-M1-7cdfa1b89d38'
    time = '2024-12-22'
    df = ReadData.ReadData_Day(beeId='86200001289', mac=mac, time=time, PhoneNum='15528932507', password='123456', DataType='Temperature')

    print(df)























