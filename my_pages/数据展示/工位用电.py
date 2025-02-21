import streamlit as st
from streamlit_extras.card import card
from utils import ReadData
from Globals import PHONE_NUM, PASSWORD, TIME_INTERVAL, workstation_lib
import numpy as np
from streamlit_echarts import st_pyecharts as st_echarts
from pyecharts.charts import HeatMap, Line
from pyecharts import options as opts

from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=TIME_INTERVAL * 60 * 1000, key="autorefresh")

@st.cache_data(ttl=TIME_INTERVAL*60)
def Get_WorkStation_RealTime()->list:
    '''
    获取每个工位的实施功率，现在是针对24个工位的版本
    '''
    text_1 = ReadData.ReadData_RealTime(beeId='86200001289', PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    text_2 = ReadData.ReadData_RealTime(beeId='86200001187', PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    # 合并两个字典->>这里是因为两个text中有key是一样的，所以不能直接合并
    # text = {**text_1, **text_2}
    # st.write(text)
    res = np.zeros((8, 3))
    for i in range(3, 24):
        for j in range(3):
            res[i//3,i%3] += text_1[workstation_lib[i-2]['mac'][j]] if workstation_lib[i-2]['mac'][j] in text_1 else 0.0
            res[i//3,i%3] += text_2[workstation_lib[i-2]['mac'][j]] if workstation_lib[i-2]['mac'][j] in text_2 else 0.0
    # 三个额外的工位只有两个插座1，且编号是22-24
    for i in range(0,3):
        for j in range(2):
            res[0,i] += text_1[workstation_lib[i+22]['mac'][j]] if workstation_lib[i+22]['mac'][j] in text_1 else 0.0
            res[0,i] += text_2[workstation_lib[i+22]['mac'][j]] if workstation_lib[i+22]['mac'][j] in text_2 else 0.0
    res = np.round(res, 2)
    return [[i, j, res[i, j]] for i in range(8) for j in range(3)]
def RealTime_Overview():
    '''
    实时用电概览
    '''
    global container_height

    data = Get_WorkStation_RealTime()
    figure = (
        HeatMap(init_opts=opts.InitOpts(width='1000px', height='1000px'))
        .add_xaxis([i for i in range(8)])
        .add_yaxis('工位功率', [i for i in range(3)], data)

        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=150),
            title_opts=opts.TitleOpts(title='工位用电情况'),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
        )

        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(border_color='white', border_width=5, border_radius=10),
        )
    )
    if 'figure' in locals():
        st_echarts(figure, height=container_height-220, width='100%')
    pass
def RealTime_Overview_Side():
    '''
    实时用电概览侧边栏
    '''
    data = np.array(Get_WorkStation_RealTime())
    col1,col2,col3 = st.columns([1, 1, 1])
    with col1:
        card(
            title='实时最大功率',
            text=str(np.max(data))+'W',
            image= ReadData.image2base64('Pictures/D63344.png'),
            styles={
                'card':{
                    'width':'100%',
                    'height':'80%',
                    'margin':'0px',
                },
                'filter':{
                    'background':'rgba(0,0,0,0.4)'
                }
            }
        )
    with col2:
        card(
            title='实时最小功率',
            text=str(np.min(data))+'W',
            image= ReadData.image2base64('Pictures/3AB744.png'),
            styles={
                'card':{
                    'width':'100%',
                    'height':'80%',
                    'margin':'0px',
                },
                'filter':{
                    'background':'rgba(0,0,0,0.4)'
                }
            }
        )
    with col3:
        card(
            title='实时平均功率',
            text=str(round(np.mean(data),2))+'W',
            image= ReadData.image2base64('Pictures/3EABF6.png'),
            styles={
                'card':{
                    'width':'100%',
                    'height':'80%',
                    'margin':'0px',
                },
                'filter':{
                    'background':'rgba(0,0,0,0.4)'
                }
            }
        )
    pass

if __name__ == '__page__':
    st.title('工位用电')

    container_height = 700

    with st.container(border=True, height=container_height):
        RealTime_Overview()
        RealTime_Overview_Side()
        

