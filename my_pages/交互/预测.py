import streamlit as st
import streamlit.components.v1 as components
import torch
from datetime import datetime, timedelta
from utils import ReadData
from Globals import devices_lib,PHONE_NUM,PASSWORD,TIME_INTERVAL
from pyecharts.charts import Line
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts as st_echarts
import numpy as np
from streamlit_extras.card import card


# 加载模型
@st.cache_resource
def load_model(model_path:str, device:torch.device, weights_only:bool=False):
    '''
    加载模型
    '''
    model = torch.load(model_path, map_location=device, weights_only=weights_only)
    return model

# 加载今天和昨天以五分钟为时间间隔的序列
# @st.cache_data(ttl=TIME_INTERVAL*60)
@st.cache_data(ttl=5*60)
def load_data():
    '''
    加载今天和昨天以五分钟为时间间隔的序列

    Returns:
    --------
    today_data: np.ndarray
        今天的数据
    yesterday_data: np.ndarray
        昨天的数据
    '''
    # 日期字符串
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 生成今天从开始到现在每5min的时间戳
    today_timestamps = [int((datetime.strptime(today_str, '%Y-%m-%d') + timedelta(minutes=5*(i+1))).timestamp()) for i in range(int((datetime.now() - datetime.strptime(today_str, '%Y-%m-%d')).seconds/300))]

    yesterday_timestamps = [int((datetime.strptime(yesterday_str, '%Y-%m-%d') + timedelta(minutes=5*(i+1))).timestamp()) for i in range(24*12)]
    # 生成此时此刻最靠近的整5分钟的时间戳

    # 读取数据
    df_today = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=today_str, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df_yesterday = ReadData.ReadData_Day(beeId=devices_lib['总表']['beeID'], mac=devices_lib['总表']['mac'], time=yesterday_str, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')

    today_data = []
    yesterday_data = []
    for timestamp in today_timestamps:
        today_data.append(df_today[df_today['TimeStamp'] <= timestamp].iloc[-1]['P'])
    for timestamp in yesterday_timestamps:
        yesterday_data.append(df_yesterday[df_yesterday['TimeStamp'] <= timestamp].iloc[-1]['P'])

    num_today = len(today_data)
    data = yesterday_data + today_data

    data_x = []
    for i in range(0,num_today+1):
        data_x.append(data[i:i+24*12])
    # 将data_x转换为tensor并在axis=1插入一个维度
    data_x = torch.tensor(data_x).unsqueeze(1).to(device)

    return data_x, num_today, today_timestamps, today_data


# 页面展示和可视化
def Predict():
    '''
    预测页面
    '''
    model_path_dict = {
        '总表': 'Model/LSTM_ATT_SCNLoad.pkl'
    }


    # 选择设备
    device_name = st.selectbox('选择设备', list(model_path_dict.keys()), index=0)

    # 加载模型
    model = load_model(model_path_dict[device_name], device)

    # 加载数据
    data_x, num_today, today_timestamps, today_data = load_data()
    mean = data_x.mean()
    std = data_x.std()
    data_x = (data_x-mean)/std
    y = model(data_x).detach().numpy()[:,0,0].astype(float)*std.item()+mean.item()
    y = np.round(y, 2)

    # 生成从00:05开始每隔5min的num_today+1个时间字符串
    time_list = [ReadData.timestamp2str(timestamp)[-8:-3] for timestamp in today_timestamps+[today_timestamps[-1]+300]]

    dataset = [['Time', 'Data', 'RawData']]

    for i in range(1, num_today+2):
        if i==num_today+1:
            dataset.append([time_list[i-1], y[i-1], None])
        else:
            dataset.append([time_list[i-1], y[i-1], today_data[i-1]])
            
    # 画图
    figure = (
                Line(init_opts=opts.InitOpts(width='1000px'))
                .add_dataset(source=dataset)
                .add_yaxis(series_name='P_Predict', y_axis=[], encode={'x': 'Time', 'y': 'Data'}, is_connect_nones=True, itemstyle_opts=opts.ItemStyleOpts(color='gray', opacity=0.7))
                .add_yaxis(series_name='P_Real', y_axis=[], encode={'x': 'Time', 'y': 'RawData'}, is_connect_nones=True, linestyle_opts=opts.LineStyleOpts(width=2))
                

                .set_global_opts(
                    title_opts=opts.TitleOpts(title='日功率曲线'),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                    yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False), type_='value'),
                    xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=100, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False), type_='category'),
                )

                .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=False),
                    markpoint_opts=opts.MarkPointOpts(data=[
                                                            opts.MarkPointItem(type_='max', name='最大值', symbol_size=80, itemstyle_opts=opts.ItemStyleOpts(opacity=0.8)),
                                                            opts.MarkPointItem(type_='min', name='最小值', symbol_size=40, itemstyle_opts=opts.ItemStyleOpts(opacity=0.8)),
                                                            ]
                                                    ),
                    markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_='average', name='平均值')]),
                )
            )
    col1,col2 = st.columns([8,2])

    with col1:
        st_echarts(figure, height=container_height-100)
    with col2:
        Show_Next_Power(time_list[-1], y[-1])
        Show_RMSE(np.array(y[:num_today]), np.array(today_data))

def Show_Next_Power(time, P_predict):
    '''展示下一时刻功率值'''
    card(
        title= '预测'+time+'功率值(单位:W)',
        text= [P_predict],
        image= ReadData.image2base64('Pictures/Curve_Predict.jpeg'),
        # 和container一样宽
        styles={
            'card':{
                'width':'100%',
                'height':'100%',
                # 'margin-top':'-20%',
                # 'margin-bottom':'-20%'
                # 去除margin
                'margin':'0px',
            },
            'filter':{
                'background':'rgba(0,0,0,0.4)'
            },
            'title': {
                'font-size': '24px',  # 设置标题字体大小
                'font-weight': 'bold',  # 设置字体粗细
                'color': 'white'  # 设置字体颜色
            },
            # 控制内容字体
            'text': {
                'font-size': '36px',  # 设置内容字体大小
                'font-weight': '600',  # 设置字体粗细
                'color': '#FF9900'  # 设置字体颜色
            }

        }
    )
    
def Show_RMSE(P_predict:np.ndarray, P_real:np.ndarray):
    '''展示均方根误差'''
    rmse = np.round(np.sqrt(np.mean((P_predict-P_real)**2)),2)
    card(
        title= '均方根误差(单位:W)',
        text= [rmse],
        image= ReadData.image2base64('Pictures/RMSE.png'),
        # 和container一样宽
        styles={
            'card':{
                'width':'100%',
                'height':'100%',
                # 'margin-top':'-20%',
                # 'margin-bottom':'-20%'
                # 去除margin
                'margin':'0px',
            },
            'filter':{
                'background':'rgba(0,0,0,0.4)'
            },
            'title': {
                'font-size': '24px',  # 设置标题字体大小
                'font-weight': 'bold',  # 设置字体粗细
                'color': 'white'  # 设置字体颜色
            },
            # 控制内容字体
            'text': {
                'font-size': '36px',  # 设置内容字体大小
                'font-weight': '600',  # 设置字体粗细
                'color': '#FF9900'  # 设置字体颜色
            }

        }
    )


if __name__ == '__page__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    container_height = 700
    with st.container(border=True, height=container_height):
        Predict()










