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

def RealTime_Overview():
    '''
    实时用电概览
    '''
    global container_height

    data = ReadData.Get_WorkStation_RealTime()
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
    data = np.array(ReadData.Get_WorkStation_RealTime())[:,-1]
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
        

