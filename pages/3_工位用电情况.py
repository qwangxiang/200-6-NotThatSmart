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

# 使用宽布局
st.set_page_config(layout='wide')

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
    res = np.zeros((7, 3))
    for i in range(0, 21):
        for j in range(3):
            res[i//3, i%3] += ReadData.ReadData_RealTime(BeeId='86200001289', mac=workstation_lib[i+1][j], PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
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
@st.cache_data(ttl=TIME_INTERVAL*60)
def Form_Dataset(df, data_raw, datatype):
    data_raw.rename(columns={datatype:datatype+'_Raw'}, inplace=True)
    dataset = pd.concat([df[['Time', datatype]].set_index('Time'), data_raw[['Time', datatype+'_Raw']].set_index('Time')], axis=1).sort_index().reset_index()
    dataset['Time'] =  dataset['Time'].apply(lambda x:x[-8:])
    dataset[datatype+'_Raw'][np.where(dataset[datatype+'_Raw'].isna().to_numpy())[0]] = dataset[datatype][np.where(dataset[datatype+'_Raw'].isna().to_numpy())[0]]
    dataset1 = [['Time', 'Data', 'RawData']] + dataset.to_numpy().tolist()
    # 这里注意需要删除最后一个幽灵数据，多出一根线的原因：最后一个数据和第一个数据的横坐标一样
    return dataset1[:-1]
def EnergySum(data:list, time_interval:int=TIME_INTERVAL):
    '''
    展示日用电量
    '''
    energysum = np.sum(np.array(data))/1000/(60/time_interval)
    energysum = round(energysum, 2)
    card(
        title='日用电量',
        text=str(energysum)+' kWh',
        image= ReadData.image2base64('Pictures/E3BBED.png'),
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
def AveragePower(data:list):
    '''
    展示平均功率
    '''
    average_power = np.mean(data)
    average_power = round(average_power, 2)
    card(
        title='平均功率',
        text=str(average_power)+' W',
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
def StartTime(data:list, time_list:list):
    '''
    展示起始时间
    '''
    data = np.array(data)
    start_index = ReadData.find_change_point(data, change_lower=10, change_upper=15)
    start_time = time_list[start_index]

    card(
        title='用电起始时间',
        text=start_time,
        image= ReadData.image2base64('Pictures/B8DBB3.png'),
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
def EndTime(data:list, time_list:list):
    '''
    展示结束时间
    '''
    data = np.array(data)
    # data倒过来
    data = data[::-1]
    # timelist倒过来
    time_list = time_list[::-1]
    end_index = ReadData.find_change_point(data, change_lower=10, change_upper=15, num_points=1)
    end_time = time_list[end_index]

    card(
        title='用电结束时间',
        text=end_time,
        image= ReadData.image2base64('Pictures/E29135.png'),
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
def ShowWorkStation(index:int):
    '''
    展示工位用电情况
    '''
    global container_height

    col1,col2,col3 = st.columns([2, 2, 1])
    with col1:
        date = str(st.date_input('选择日期', value='today'))
    with col2:
        view_type = st.selectbox('查看类型', ['总览', '插座1', '插座2', '插座3'])
    if view_type != '总览':
        with col3:
            show_raw_data = st.toggle('显示原始数据', False)
    df1_raw = ReadData.ReadData_Day(beeId=workstation_lib['beeTd'], mac=workstation_lib[index][0], time=date, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df1 = ReadData.TimeIntervalTransform(df1_raw, date=date, time_interval=TIME_INTERVAL, DataType='P')
    df2_raw = ReadData.ReadData_Day(beeId=workstation_lib['beeTd'], mac=workstation_lib[index][1], time=date, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df2 = ReadData.TimeIntervalTransform(df2_raw, date=date, time_interval=TIME_INTERVAL, DataType='P')
    df3_raw = ReadData.ReadData_Day(beeId=workstation_lib['beeTd'], mac=workstation_lib[index][2], time=date, PhoneNum=PHONE_NUM, password=PASSWORD, DataType='P')
    df3 = ReadData.TimeIntervalTransform(df3_raw, date=date, time_interval=TIME_INTERVAL, DataType='P')
    df_raw = [df1_raw, df2_raw, df3_raw]
    df = [df1, df2, df3]
    if view_type == '总览':
        data = (df1['P'].to_numpy()+df2['P'].to_numpy()+df3['P'].to_numpy()).tolist()
        time_list = [i[-8:] for i in df1['Time'].to_list()]
        # 折线图
        figure = (
            Line(init_opts=opts.InitOpts())
            .add_xaxis(time_list)
            .add_yaxis('总功率', data)

            .set_global_opts(
                title_opts=opts.TitleOpts(title='日功率曲线'),
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False)),
                xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=4, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False)),
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
    else:
        if not show_raw_data:
            data = df[int(view_type[-1])-1]['P'].to_list()
            time_list = [i[-8:] for i in df[int(view_type[-1])-1]['Time'].to_list()]
            # 折线图
            figure = (
                Line()
                .add_xaxis(time_list)
                .add_yaxis('功率', data, is_connect_nones=True)

                .set_global_opts(
                    title_opts=opts.TitleOpts(title='日功率曲线'),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                    yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False)),
                    xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=4, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False)),
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
        else:
            data_raw = df_raw[int(view_type[-1])-1]
            data = df[int(view_type[-1])-1]
            time_list = [i[-8:] for i in data['Time'].to_list()]
            dataset = Form_Dataset(data, data_raw, 'P')
            data = data['P'].to_list()
            figure = (
                Line()
                .add_dataset(source=dataset)
                .add_yaxis(series_name='P'+'_Raw', y_axis=[], encode={'x': 'Time', 'y': 'RawData'}, is_connect_nones=False, itemstyle_opts=opts.ItemStyleOpts(color='gray', opacity=0.7))
                .add_yaxis(series_name='P', y_axis=[], encode={'x': 'Time', 'y': 'Data'}, is_connect_nones=True, linestyle_opts=opts.LineStyleOpts(width=2))
                

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
    st_echarts(figure, height=container_height-310, width='100%')

    # 展示总功率
    col1_,col2_,col3_,col4_ = st.columns([1, 1, 1, 1])
    with col1_:
        EnergySum(data)
    with col2_:
        AveragePower(data)
    with col3_:
        StartTime(data, time_list)
    with col4_:
        EndTime(data, time_list)


    pass

if __name__ == '__main__':
    st.title('工位用电情况')

    container_height = 700

    with st.container(border=True, height=container_height):
        select = st.selectbox('查看对象', ['实时总览']+['工位'+str(i) for i in range(1, 22)])
        if select == '实时总览':
            RealTime_Overview()
            RealTime_Overview_Side()
        else:
            ShowWorkStation(int(select[2:]))

