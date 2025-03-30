import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from Globals import API_SERVER
import time
from AI.BuildAgent import Link_To_LangSmith, Create_Tool_Agent, Get_Agent_With_History, StreamHandler
import asyncio
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

def Show_message():
    '''
    负责展示历史消息
    '''
    for message in st.session_state.chat_history:
        with st.chat_message(message[0]):
            st.write(message[1])

def Colored_text(text:str, color:str):
    return f'<span style="color:{color}">{text}</span>'

def Chat():
    '''
    chat工作流
    '''
    user_input = st.chat_input("Type something...")

    if user_input:
        

        # 展示用户输入
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append(('user', user_input))

        # 准备回答容器
        answer_container = st.chat_message("ai")
        
        with answer_container:
            stream_handler = StreamHandler(st.empty())
            # 创建一个占位符
            placeholder = st.empty()

        agent_executor = Get_Agent_With_History(api_server, model, current_query=user_input, stream_handler=stream_handler)
        answer = agent_executor.invoke({'current_query':user_input})['output']

        # 处理答案形成最终答案
        # 提取<figure>标签中的内容
        # answer = answer.split('<figure>')[0]+answer.split('</figure>')[-1]
    

        # 存储最终答案
        st.session_state.chat_history.append(('ai', answer))
        # 重新加载页面但是不刷新session_state
        st.rerun(scope='app')

# 在页面上渲染html字符串的参考代码
# with open('Figure/figure01.html', 'r', encoding="utf-8") as f:
#         html_content = f.read()
#     with st.container(border=True):
#         components.html(html_content, height=500)



if __name__ == '__page__':
    # 定义使用的api服务商和模型
    api_server = 'huoshan'
    model = 'doubao-pro-32k-241215'
    # api_server = 'siliconflow'
    # model = 'Qwen/QwQ-32B'

    # 连接到LangSmith
    Link_To_LangSmith(api_server)
    
    # 初始化消息存储区域
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史消息
    Show_message()

    # 聊天
    Chat()
    # asyncio.run(Chat())


    pass

