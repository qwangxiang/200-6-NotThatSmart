import streamlit as st
import streamlit.components.v1 as components

if __name__ == '__page__':
    # 在页面中渲染html图表
    # st.write('Figure/figure01.html')
    # 将Figure/figure01.html加载成字符串
    with open('Figure/figure01.html', 'r', encoding="utf-8") as f:
        html_content = f.read()
    with st.container(border=True):
        components.html(html_content, height=500)







