import streamlit as st

st.set_page_config(layout='wide')

pages = {
    ' ': [
        st.Page('Main.py', title='主页')
    ],

    '数据展示':[
        st.Page('my_pages/数据展示/用电总览.py', title='用电总览'),
        st.Page('my_pages/数据展示/设备用电.py', title='设备用电'),
        st.Page('my_pages/数据展示/工位用电.py', title='工位用电'),
    ],
        
    '数据分析':[
        st.Page('my_pages/数据分析/总体用电分析.py', title='总体用电分析'),
        st.Page('my_pages/数据分析/工位用电分析.py', title='工位用电分析'),
        st.Page('my_pages/数据分析/设备用电分析.py', title='设备用电分析'),
    ],

    '交互': [
        # st.Page('my_pages/交互/Chat.py', title='Chat'),
        st.Page('my_pages/交互/预测.py', title='预测'),
        st.Page('my_pages/交互/数据下载.py', title='数据下载'),
    ]
}

st.navigation(pages, position='sidebar').run()

