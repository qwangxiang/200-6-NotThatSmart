import streamlit as st


'''
定义帮助构建应用的通用工具函数
'''

def Show_message(history_flag:str='chat_history'):
    '''
    负责展示历史消息
    '''
    for message in st.session_state[history_flag]:
        with st.chat_message(message[0]):
            st.markdown(message[1])