import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from Globals import API_SERVER
from AI.Tools import Get_Tools
from AI.BuildPrompt import Get_Prompt_Template
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from typing import Literal

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text, unsafe_allow_html=True)

def Link_To_LangSmith(api_server:str):
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_API_KEY'] = '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'
    os.environ['LANGSMITH_PROJECT'] = 'LANGSMITH_PROJECT="200-6-NotThatSmart"'
    os.environ['OPENAI_API_KEY'] = API_SERVER[api_server]['API_KEY']

def Create_Tool_Agent(api_server:str, model:str, prompt_template, verbose:bool=False, stream_handler=None, Type:Literal['Chat','SideBar']='Chat'):
    tools = Get_Tools()
    llm = ChatOpenAI(
        model=model,
        api_key=API_SERVER[api_server]['API_KEY'],
        base_url=API_SERVER[api_server]['BASE_URL'],
        callbacks=[stream_handler]
    )
    if Type == 'Chat':
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    elif Type == 'SideBar':
        agent = create_tool_calling_agent(
            llm=llm,
            tools = [],
            prompt=prompt_template,
        )
        agent_executor = AgentExecutor(agent=agent, tools=[], verbose=verbose, max_iterations=1, return_intermediate_steps=True)
    return agent_executor

def Get_Agent_With_History(api_server, model, current_query=None, verbose=False, stream_handler=None, AgentType:Literal['Chat','SideBar']='Chat', history_flag:str='chat_history'):
    """创建带有历史记忆的Agent"""
    
    # 构建带有历史的prompt_template
    if AgentType == 'Chat':
        prompt_template = Get_Prompt_Template(current_query=current_query, Type='Chat', history_flag=history_flag)
    elif AgentType == 'SideBar':
        # 侧边栏的prompt_template不需要current_query
        prompt_template = Get_Prompt_Template(current_query=current_query, Type='SideBar', history_flag=history_flag)

    agent_executor = Create_Tool_Agent(
        api_server=api_server, 
        model=model, 
        prompt_template=prompt_template,
        verbose=verbose,
        stream_handler=stream_handler,
        Type=AgentType
    )
    
    return agent_executor

if __name__=='__main__':
    pass









