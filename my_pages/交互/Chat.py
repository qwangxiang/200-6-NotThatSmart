import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from Globals import API_SERVER
import time
from AI.BuildAgent import Link_To_LangSmith, Create_Tool_Agent, Get_Agent_With_History
import asyncio
from langchain.callbacks.base import BaseCallbackHandler
import threading

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
        # 创建agent
        agent_executor = Get_Agent_With_History(api_server, model, current_query=user_input)

        # 展示用户输入
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append(('user', user_input))


        # 准备回答容器
        answer_container = st.chat_message("ai")
        
        # 创建一个共享变量来存储回答
        result = {"answer": None, "done": False}
        
        # 创建一个线程来获取模型回答
        def get_answer():
            result["answer"] = agent_executor.invoke({'current_query':user_input})['output']
            result["done"] = True
            
        thread = threading.Thread(target=get_answer)
        thread.start()
        
        # 在等待回答时显示动画
        with answer_container:
            placeholder = st.empty()
            dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while not result["done"]:
                placeholder.markdown('*'+Colored_text('任务进行中'+dots[i], "gray")+'*', unsafe_allow_html=True)
                time.sleep(0.2)
                i = (i + 1) % len(dots)

            placeholder.write('*'+Colored_text('任务完成!', "green")+'*', unsafe_allow_html=True)
            # 显示最终答案
            st.write(result["answer"])


        st.session_state.chat_history.append(('ai', result["answer"]))
        
if __name__ == '__page__':
    # 定义使用的api服务商和模型
    api_server = 'huoshan'
    model = 'doubao-pro-32k-241215'

    # 连接到LangSmith
    Link_To_LangSmith(api_server)
    
    # 初始化消息存储区域
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史消息
    Show_message()

    # 聊天
    Chat()

    pass

