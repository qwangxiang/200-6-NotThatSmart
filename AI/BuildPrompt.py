import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
# from langchain_community.vectorstores import Chroma
from Globals import API_SERVER


# 预定义示例，每个示例是一个(user_query, ai_response)对
examples_for_selection = [
    {
        'user': '帮我查询打印机的信息', 
        'ai': '这里所指的信息应该是设备的beeID和mac，由于相关的设备信息没有存储在我的记忆中，我应该注意用户定义的工具中有没有类似的工具，调用类似的工具完成任务。'
    },
    {
        'user': '帮我查询冰箱今天的功率序列', 
        'ai': '用户让我帮他查询冰箱的用电序列，由于此时明确了返回功率序列，所以我只需要查询到数据返回即可。首先我应该调用日期构造工具，构造指定日的日期字符串，比如今天就应该传入today，然后调用数据查询工具，传入上述两个参数，最后将获得的功率序列返回用户。'
    },
    {
        'user': '今天的日期是多少？', 
        'ai': '用户问今天的日期，我应该调用日期构造工具，传入date=today，就能获得符合格式要求的日期字符串。'
    },
    {
        'user': '2月12日的日期字符串', 
        'ai': '今年是2025年，我应该调用日期构造工具，传入year=2025, date=0212，就能获得符合格式要求的日期字符串。'
    },
    {
        'user': '打印机今天的用电情况怎么样？',
        'ai': '用户让我分析打印机的用电情况，为了更加清晰，我应该使用图表进行分析。因此，我应该先调用日期字符串构造工具，得到今天的日期字符串；然后调用数据查询工具，传入打印机的Device_name和Date，，获得用电数据；最后调用图表工具，传入用电数据，接收到图表的html字符串，并直接将其返回给用户。'
    }
]
examples_fixed = [
]


def select_relevant_examples(query, examples=examples_for_selection, top_k=100):
    """选择与当前查询最相关的几个示例"""
    # 示例筛选暂时不缺api，后面有时间再做
    return examples

def Get_Prompt_Template(current_query, max_history=10, Type='Chat', history_flag='chat_history'):
    if Type == 'Chat':
        prompt_template = Get_Prompt_Template_Chat(current_query=current_query, max_history=max_history)
    elif Type == 'SideBar':
        prompt_template = Get_Prompt_Template_SideBar(current_query, history_flag=history_flag)
    return prompt_template

def Get_CurrentQuery_SideBar(current_query:dict):
    '''
    根据当前页面的内容生成当前的query

    Parameters:
    ---
    current_query: dict
        name: 当前页面的名称
        data: 待分析的数据

    Returns:
    ---
    current_query: str
    '''
    # 用电总览页面的SideBarAI
    if current_query['name'] == '用电总览':
        query = f"这个是整个实验室的用电数据：{current_query['Data']}，对应的时间点序列是{current_query['Time']}，数据类型是{current_query['DataType']},"+'{current_query}。'

    return query

def Get_Prompt_Template_SideBar(current_query, history_flag):
    '''
    构建用于侧边栏AI的提示模板

    Parameters:
    ---
    history_flag: 历史对话列表，每项为(role, content)
    
    Returns:
    ---
    prompt_template: 提示模板对象
    '''

    max_history = 10

    # 系统消息
    system_message = (
        'system',
        """
        你是实验室用电数据的专业分析助手，负责对当前页面展示的用电数据进行自动化分析，当前页面的数据内容会由下面一条用户消息提供给你。你的主要任务是提供简洁、专业的数据洞察，无需用户详细指令即可工作。

        分析重点:
        1. 用电曲线趋势分析：识别上升/下降趋势、周期性模式、峰谷分布
        2. 异常检测：发现用电异常点，判断是否存在设备故障或非正常工作
        3. 能效评估：评估能源使用效率，提出优化建议
        4. 用电负荷特征：识别基础负荷、波动负荷，判断设备使用模式
        5. 与历史数据比较：与过去同期数据对比，发现长期变化

        输出规范:
        1. 保持分析简洁（不超过3-5个关键观察点）
        2. 使用专业但易懂的语言
        3. 当发现异常情况时，明确标识并提供可能的原因
        4. 以要点形式呈现，便于快速阅读
        5. 如有可能，提供数据支持的具体建议

        注意事项:
        1. 你只需分析当前页面展示的数据，不需处理用户的其他查询
        2. 分析应适应当前上下文（当前页面是设备级还是实验室级数据）
        3. 不要询问用户更多信息，根据可见数据直接给出最佳分析
        4. 如果数据不足以得出确定结论，坦率指出并给出基于有限信息的初步分析
        5. 你只需要对提供给你的数据按照你的知识库进行分析即可，不需要调用工具之类的操作，如果没有读取到有效数据，你可以直接在回答中指出这一点
        6. 注意将数据的变化和人的行为信息相结合，比如对于整个实验室的用电数据在早上突升，那么可能是大家都来上班了
        7. 如果从某个点到一天结束数据都为0，那是因为未来的数据还没有到达，所以你可以直接忽略掉这一点
        8. 一般来说，数据都是15分钟一个点，且提供给你的数据序列和时间序列中的点是一一对应的，即某个数据发生在该时间点
        
        你的分析将帮助实验室人员快速理解当前用电状况，发现潜在问题，并做出数据驱动的决策。
        """
    )

    # 构建消息列表
    messages = [system_message]
    # SidebarAI只提供提供初步的探索功能，所以不需要加载历史消息。


    # 构建针对当前页面的提示模板
    current_query = Get_CurrentQuery_SideBar(current_query)
    messages.append(('user', current_query))
    messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))

    return ChatPromptTemplate.from_messages(messages)

def Get_Prompt_Template_Chat(current_query, max_history=10):
    """
    构建带有历史相关示例的提示模板记忆和
    
    参数:
    chat_history: 历史对话列表，每项为(role, content)
    current_query: 当前用户查询，用于选择相关示例
    max_history: 保留的最大历史消息数量
    """
    chat_history = st.session_state.chat_history if 'chat_history' in st.session_state else None

    
    # 选择相关示例
    # relevant_examples = example_selector({'user': current_query})
    relevant_examples = select_relevant_examples(current_query)
    
    # 构建示例消息
    example_messages = []
    for chat_message in relevant_examples:
        example_messages.append(('user', chat_message['user']))
        example_messages.append(('ai', chat_message['ai']))
    for chat_message in examples_fixed:
        example_messages.append(('user', chat_message['user']))
        example_messages.append(('ai', chat_message['ai']))
    
    # 系统消息
    system_message = (
        'system',
        """
        你是一位专业的数据分析助手，可以帮助用户执行多步骤的数据处理任务。

        你可以调用多个工具按顺序完成复杂任务。当任务需要多个步骤时，请：
        1. 分析整个任务流程，确定需要使用的工具顺序
        2. 逐步执行每个工具，并在每次调用后思考下一步
        3. 将前一个工具的输出作为下一个工具的输入（需要时进行适当转换）
        4. 清晰地解释每一步的过程和结果
        5. 最后提供综合分析和总结

        严格遵循以下规则：
        1. 仅执行用户明确要求的任务，绝不主动执行未被要求的内容
        2. 确认用户请求是否需要多步骤，如不需要，只执行单一工具调用
        3. 调用工具前，确保传入的参数是严格的JSON格式，不要有其他多余的字符。

        关于图表展示：
        1. 在为用户展示数据时，优先调用图表工具，以图表的形式展示数据；

        确保准确跟踪工具调用之间的数据流动，并在必要时对中间结果进行处理，同时只需要完成用户要求你做的事情，不要额外完成其他内容。今年是2025年。
        """
    )
    
    # 构建基础消息列表 = 系统消息 + 示例 + 当前用户输入
    messages = [system_message] + example_messages
    
    # 添加历史对话（如果有）
    history_promt = []
    if chat_history:
        # 保留最近max_history条的
        if len(chat_history) > max_history:
            # 保留最近的max_history条消息
            truncated_history = chat_history[-max_history:]
            for role, content in truncated_history:
                history_promt.append((role, content))

            # 总结前面的消息
            conclusion_promt = """
            请总结上述对话，需要：
            1. 保留所有关键信息点和重要问答
            2. 维持对话的逻辑顺序和上下文连贯性
            3. 删减重复内容、客套语和不必要的细节
            4. 使用简洁直接的语言表达
            5. 特别注意保留与数据查询、时间日期和设备相关的具体细节
            
            总结后的内容应当让模型理解之前的对话背景，但字数不超过原文的30%。
            """
            history_previous = chat_history[:-4]+[('user',conclusion_promt)]
            summarize_prompt = ChatPromptTemplate.from_messages(history_previous)
            llm = ChatOpenAI(
                model='deepseek-ai/DeepSeek-V3',
                api_key=API_SERVER['siliconflow']['API_KEY'],
                base_url=API_SERVER['siliconflow']['BASE_URL'],
            )
            summarize_chat = summarize_prompt | llm
            summarization = summarize_chat.invoke(input={}).content
            # 将上述内容添加到prompt中去
            messages.append(('system', summarization))
            messages = messages + truncated_history
        else:
            messages = messages + chat_history

    # 添加当前用户输入和agent_scratchpad占位符
    messages.append(('user', '{current_query}'))
    messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
    
    return ChatPromptTemplate.from_messages(messages)



if __name__=='__main__':
    query = '打印机的信息'
    prompt_template = Get_Prompt_Template(current_query=query)
    print(prompt_template)
