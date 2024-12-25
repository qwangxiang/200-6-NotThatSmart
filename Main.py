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

    card(
        title='标摊',
        text='标摊是什么？',
    )























