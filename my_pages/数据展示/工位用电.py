import streamlit as st
from streamlit_extras.card import card
from utils import ReadData
from Globals import PHONE_NUM, PASSWORD, TIME_INTERVAL
import numpy as np
from streamlit_echarts import st_pyecharts as st_echarts
from pyecharts.charts import HeatMap, Line
from pyecharts import options as opts
import datetime
import pandas as pd

workstation_lib = {
    'beeTd': '86200001289',
    1:  ['Sck-M1-7cdfa1b66100', 'Sck-M1-7cdfa1b881e8', 'Sck-M1-84f703123840'],
    2:  ['Sck-M1-7cdfa1b85208', 'Sck-M1-84f7031237dc', 'Sck-M1-84f703121774'],
    3:  ['Sck-M1-84f70310590c', 'Sck-M1-84f7031206c4', 'Sck-M1-84f70312297c'],
    4:  ['Sck-M1-7cdfa1b85f04', 'Sck-M1-7cdfa1b89c00', 'Sck-M1-7cdfa1b89d04'],
    5:  ['Sck-M1-7cdfa1b85210', 'Sck-M1-7cdfa1b66104', 'Sck-M1-7cdfa1b84ea0'],
    6:  ['Sck-M1-7cdfa1b88ab0', 'Sck-M1-84f703121628', 'Sck-M1-84f703120888'],
    7:  ['Sck-M1-84f703110ba0', 'Sck-M1-7cdfa1b85f8c', 'Sck-M1-84f7031219d4'],
    8:  ['Sck-M1-84f70311fbe4', 'Sck-M1-84f703123a30', 'Sck-M1-7cdfa1b89edc'],
    9:  ['Sck-M1-7cdfa1b89db8', 'Sck-M1-7cdfa1b86a84', 'Sck-M1-84f70310e620'],
    10: ['Sck-M1-84f7031218a4', 'Sck-M1-7cdfa1b66158', 'Sck-M1-84f7031217c8'],
    11: ['Sck-M1-7cdfa1b8519c', 'Sck-M1-7cdfa1b84d48', 'Sck-M1-7cdfa1b660d4'],
    12: ['Sck-M1-84f703120a1c', 'Sck-M1-84f70311fbac', 'Sck-M1-7cdfa1b88b34'],
    13: ['Sck-M1-7cdfa1b89f40', 'Sck-M1-84f70310d18c', 'Sck-M1-7cdfa1bff7dc'],
    14: ['Sck-M1-84f703113ea0', 'Sck-M1-84f7031238bc', 'Sck-M1-84f703123074'],
    15: ['Sck-M1-7cdfa1b869ec', 'Sck-M1-84f70311feec', 'Sck-M1-84f703120a50'],
    16: ['Sck-M1-84f703115e64', 'Sck-M1-84f703121af4', 'Sck-M1-7cdfa1b85268'],
    17: ['Sck-M1-84f703119434', 'Sck-M1-7cdfa1b66154', 'Sck-M1-7cdfa1b89cd0'],
    18: ['Sck-M1-84f70312128c', 'Sck-M1-7cdfa1b87dcc', 'Sck-M1-84f703120088'],
    19: ['Sck-M1-7cdfa1b6611c', 'Sck-M1-7cdfa1b6614c', 'Sck-M1-84f703121b38'],
    20: ['Sck-M1-84f70310f2c8', 'Sck-M1-7cdfa1b89f78', 'Sck-M1-7cdfa1b89f70'],
    21: ['Sck-M1-7cdfa1b951a8', 'Sck-M1-84f703122cf0', 'Sck-M1-84f70311f3bc']
}

@st.cache_data(ttl=TIME_INTERVAL*60)
def Get_WorkStation_RealTime()->list:
    '''
    获取每个工位的实施功率
    '''
    text = ReadData.ReadData_RealTime(beeId=workstation_lib['beeTd'], PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    res = np.zeros((7, 3))
    for i in range(0, 21):
        for j in range(3):
            res[i//3,i%3] += text[workstation_lib[i+1][j]] if workstation_lib[i+1][j] in text else 0
    res = np.round(res, 2)
    return [[i, j, res[i, j]] for i in range(7) for j in range(3)]
def RealTime_Overview():
    '''
    实时用电概览
    '''
    global container_height

    data = Get_WorkStation_RealTime()
    figure = (
        HeatMap(init_opts=opts.InitOpts(width='1000px', height='1000px'))
        .add_xaxis([i for i in range(7)])
        .add_yaxis('工位功率', [i for i in range(3)], data)

        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=100),
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
    data = np.array(Get_WorkStation_RealTime())[:,-1]
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
        

