import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from Globals import API_SERVER
from AI.Tools import Get_Tools
from AI.BuildPrompt import Get_Prompt_Template
import streamlit as st
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

def Link_To_LangSmith(api_server:str):
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_API_KEY'] = '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'
    os.environ['LANGSMITH_PROJECT'] = 'LANGSMITH_PROJECT="200-6-NotThatSmart"'
    os.environ['OPENAI_API_KEY'] = API_SERVER[api_server]['API_KEY']

def Create_Tool_Agent(api_server:str, model:str, verbose:bool=False):
    if f'agent_{api_server}_{model}_{verbose}' in st.session_state:
        return st.session_state[f'agent_{api_server}_{model}_{verbose}']

    tools = Get_Tools()
    prompt_template = Get_Prompt_Template()
    llm = ChatOpenAI(
        model=model,
        api_key=API_SERVER[api_server]['API_KEY'],
        base_url=API_SERVER[api_server]['BASE_URL'],
        streaming=True,
        callbacks=[FinalStreamingStdOutCallbackHandler()],
    )
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    st.session_state[f'agent_{api_server}_{model}_{verbose}'] = agent_executor
    return agent_executor

if __name__=='__main__':
    pass









