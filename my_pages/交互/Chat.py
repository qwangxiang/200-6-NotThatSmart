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


# 尝试以异步的方式解析模型返回的event，实现流式输出，但是一直会报错，可能是框架本身的问题
async def process_stream(agent_executor, user_input):
    with st.chat_message('ai'):
        with st.expander(':bulb: Thinking'):
            thinking_placeholder = st.empty()
        thinking_full_message = ''
        answer_placeholder = st.empty()
        answer_full_message = ''
        full_message = ''
        answer_start = False
        
        async for event in agent_executor.astream_events({'current_query':user_input}):
            # 处理事件...
            if event['event']=='on_chat_model_stream':
                full_message += event['data']['chunk'].content
                if answer_start:
                    answer_full_message += event['data']['chunk'].content
                    answer_placeholder.markdown(answer_full_message)
                else:
                    thinking_full_message += event['data']['chunk'].content
                    thinking_placeholder.markdown(Colored_text(thinking_full_message, "gray"), unsafe_allow_html=True)
                if '#$&' in full_message:
                    answer_start = True
                    thinking_full_message = full_message.split('#$&')[0]
                    answer_full_message = full_message.split('#$&')[1]
                    thinking_placeholder.markdown(Colored_text(thinking_full_message, "gray"), unsafe_allow_html=True)
        
        # 循环自然结束后执行
        return answer_full_message


async def aChat():
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

        answer = await process_stream(agent_executor, user_input)               

        st.session_state.chat_history.append(('ai', answer))
    
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
            result["res"] = agent_executor.invoke({'current_query':user_input})['output']
            result["done"] = True
            
        thread = threading.Thread(target=get_answer)
        thread.start()
        
        # 在等待回答时显示动画
        with answer_container:
            with st.expander(':bulb: Thinking...', expanded=True):
                placeholder_thinking = st.empty()
            dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while not result["done"]:
                placeholder_thinking.markdown('*'+Colored_text('任务进行中'+dots[i], "gray")+'*', unsafe_allow_html=True)
                time.sleep(0.2)
                i = (i + 1) % len(dots)

            thinking = result["res"].split('#$&')[0]
            answer = result["res"].split('#$&')[-1]
            placeholder_thinking.markdown(Colored_text(thinking, "gray"), unsafe_allow_html=True)
            st.write(answer)

            # 存储最终答案
            st.session_state.chat_history.append(('ai', answer))

if __name__ == '__page__':
    # 定义使用的api服务商和模型
    # api_server = 'huoshan'
    # model = 'doubao-pro-32k-241215'
    api_server = 'siliconflow'
    model = 'Qwen/QwQ-32B'

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

