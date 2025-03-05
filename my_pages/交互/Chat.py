import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from Globals import API_SERVER
import time
from AI.BuildAgent import Link_To_LangSmith, Create_Tool_Agent

def Show_message():
    '''
    负责展示历史消息
    '''
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def Chat():
    '''
    chat工作流
    '''
    agent_executor = Create_Tool_Agent(api_server, model, verbose=False)

    user_input = st.chat_input("Type something...")
    if user_input:
        # 展示用户输入
        st.session_state.messages.append({"role": "user", "content": user_input})
        Show_message()

        # 获取模型回答
        answer = agent_executor.invoke({'input': user_input})['output']

        st.session_state.messages.append({"role": "ai", "content": answer})
        with st.chat_message("ai"):
            st.write(answer)



if __name__ == '__page__':
    # 定义使用的api服务商和模型
    api_server = 'huoshan'
    model = 'doubao-pro-32k-241215'

    # 连接到LangSmith
    Link_To_LangSmith(api_server)
    
    # 初始化消息存储区域
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 聊天
    Chat()

    pass

