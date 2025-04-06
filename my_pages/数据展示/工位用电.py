import streamlit as st
from streamlit_extras.card import card
from utils import ReadData
from Globals import TIME_INTERVAL
import numpy as np
from streamlit_echarts import st_pyecharts as st_echarts
from pyecharts.charts import HeatMap
from pyecharts import options as opts
import seaborn as sns

from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=TIME_INTERVAL * 60 * 1000, key="autorefresh")

def Def_Css():
    '''
    自定义CSS样式
    '''

    st.markdown(
        """
        <style>
        /* 卡片CSS */
        .card {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2vw;
            border-radius: 1vw;
            background-size: cover;
            background-position: center;
            margin-bottom: 1vw;
            color: black;
            font-family: 'Microsoft YaHei', sans-serif; /* 改成微软雅黑 */
            border: 0.2vw solid #ccc; /* 添加边框 */
            box-shadow: 0 0.4vw 0.8vw rgba(0, 0, 0, 0.2); /* 添加阴影 */
            height: 15vw; /* 设置卡片高度 */
            transition: transform 0.5s ease, box-shadow 0.5s ease; /* 添加过渡效果 */
        }

        .card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2); /* 半透明白色蒙版 */
            border-radius: 1.5vw;
            z-index: 1;
        }

        .card .content {
            position: relative;
            z-index: 2;
        }

        .card.grayscale {
            filter: grayscale(100%);
        }

        .card:hover {
            transform: scale(1.02); /* 鼠标悬停时放大 */
            box-shadow: 0 0.8vw 1.6vw rgba(0, 0, 0, 0.2); /* 鼠标悬停时阴影加深 */
        }

        /*圆角矩形组件*/
        .rounded-rect {
            position: absolute;  /* 绝对定位使其可以放在card上的任意位置 */
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0.6vw 1.2vw;  /* 从8px 16px改为vw单位 */
            width: 9%;  /* 固定宽度 */
            height: 9%;  /* 固定高度 */
            border-radius: 10px;  /* 圆角效果保持不变 */
            
            /* 默认样式 */
            background-color: #3498db;
            color: white;
            font-family: 'Microsoft YaHei', sans-serif;
            font-size: 1vw;  /* 字体大小改为vw单位 */
            font-weight: 500;
            
            /* 装饰 */
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            
            /* 使文本可选择 */
            user-select: text;
            
            /* 层级确保显示在上层 */
            z-index: 100;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def RealTime_Overview_old():
    '''
    实时用电概览
    '''
    global container_height

    data = ReadData.Get_WorkStation_RealTime(dim=2)
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

@st.cache_data(ttl=TIME_INTERVAL*60)
def Get_Color(max_value, min_value, value):
    '''
    获取颜色值

    Parameters
    ----------
    max_value : float
        最大值
    min_value : float
        最小值
    value : float
        当前值

    Returns
    -------
    str
        颜色值
    '''
    # 归一化
    normalized_value = (value - min_value) / (max_value - min_value)
    # 利用sns进行颜色映射
    color = sns.color_palette("coolwarm", as_cmap=True)(normalized_value)
    # 转换为十六进制颜色值
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    return hex_color


def RealTime_Overview_new():
    data = ReadData.Get_WorkStation_RealTime(dim=1)
    max_value = np.max(data)
    min_value = np.min(data)

    font_family = 'Times New Roman'

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <div class="card" style="width: 90%; padding-bottom: 43%; position: relative; transition: none; box-shadow: none; transform: none;">
                <img src="https://img.picui.cn/free/2025/04/06/67f2269c0b3b2.png" style="width: 100%; height: 100%; object-fit: contain; position: absolute; top: 0; left: 0; border-radius: 0.0vw;"/>
                <div class="rounded-rect" style="top: 65%; left: 26%; background-color: {Get_Color(max_value,min_value,data[0])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[0]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 26%; background-color: {Get_Color(max_value,min_value,data[1])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[1]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 26%; background-color: {Get_Color(max_value,min_value,data[2])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[2]} W</span>
                </div>
                <div class="rounded-rect" style="top: 65%; left: 37%; background-color: {Get_Color(max_value,min_value,data[3])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[3]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 37%; background-color: {Get_Color(max_value,min_value,data[4])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[4]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 37%; background-color: {Get_Color(max_value,min_value,data[5])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[5]} W</span>
                </div>
                <div class="rounded-rect" style="top: 65%; left: 47.5%; background-color: {Get_Color(max_value,min_value,data[6])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[6]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 47.5%; background-color: {Get_Color(max_value,min_value,data[7])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[7]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 47.5%; background-color: {Get_Color(max_value,min_value,data[8])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[8]} W</span>
                </div>
                <div class="rounded-rect" style="top: 65%; left: 58%; background-color: {Get_Color(max_value,min_value,data[9])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[9]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 58%; background-color: {Get_Color(max_value,min_value,data[10])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[10]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 58%; background-color: {Get_Color(max_value,min_value,data[11])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[11]} W</span>
                </div>
                <div class="rounded-rect" style="top: 65%; left: 68.5%; background-color: {Get_Color(max_value,min_value,data[12])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[12]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 68.5%; background-color: {Get_Color(max_value,min_value,data[13])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[13]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 68.5%; background-color: {Get_Color(max_value,min_value,data[14])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[14]} W</span>
                </div>
                <div class="rounded-rect" style="top: 65%; left: 79.5%; background-color: {Get_Color(max_value,min_value,data[15])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[15]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 79.5%; background-color: {Get_Color(max_value,min_value,data[16])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[16]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 79.5%; background-color: {Get_Color(max_value,min_value,data[17])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[17]} W</span>
                </div>
                <div class="rounded-rect" style="top: 65%; left: 90%; background-color: {Get_Color(max_value,min_value,data[18])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[18]} W</span>
                </div>
                <div class="rounded-rect" style="top: 37%; left: 90%; background-color: {Get_Color(max_value,min_value,data[19])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[19]} W</span>
                </div>
                <div class="rounded-rect" style="top: 13%; left: 90%; background-color: {Get_Color(max_value,min_value,data[20])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[20]} W</span>
                </div>
                <div class="rounded-rect" style="top: 64%; left: 3%; background-color: {Get_Color(max_value,min_value,data[21])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[21]} W</span>
                </div>
                <div class="rounded-rect" style="top: 47%; left: 3%; background-color: {Get_Color(max_value,min_value,data[22])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[22]} W</span>
                </div>
                <div class="rounded-rect" style="top: 30%; left: 3%; background-color: {Get_Color(max_value,min_value,data[23])};">
                    <span style="font-size: 1vw; font-family: {font_family}; font-weight: bold;">{data[23]} W</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



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
    Def_Css()


    with st.container(border=True, height=container_height):
        RealTime_Overview_new()
        RealTime_Overview_Side()
        

