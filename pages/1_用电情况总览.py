from utils import ReadData
import streamlit as st
from streamlit_echarts import st_pyecharts as st_echarts
from pyecharts.charts import Bar,Line,HeatMap,Pie
from pyecharts import options as opts
import time
import numpy as np
import pandas as pd
from Globals import TIME_INTERVAL, PHONE_NUM, PASSWORD
from streamlit_extras.card import card
import base64

st.set_page_config(layout='wide')

@st.cache_data(ttl=TIME_INTERVAL*60)
def Form_Dataset(df, data_raw, datatype):
    data_raw.rename(columns={datatype:datatype+'_Raw'}, inplace=True)
    dataset = pd.concat([df[['Time', datatype]].set_index('Time'), data_raw[['Time', datatype+'_Raw']].set_index('Time')], axis=1).sort_index().reset_index()
    dataset['Time'] =  dataset['Time'].apply(lambda x:x[-8:])
    dataset[datatype+'_Raw'][np.where(dataset[datatype+'_Raw'].isna().to_numpy())[0]] = dataset[datatype][np.where(dataset[datatype+'_Raw'].isna().to_numpy())[0]]
    dataset1 = [['Time', 'Data', 'RawData']] + dataset.to_numpy().tolist()
    # 这里注意需要删除最后一个幽灵数据，多出一根线的原因：最后一个数据和第一个数据的横坐标一样
    return dataset1[:-1]

def power_curve():
    '''
    用电曲线
    '''
    global date
    height = containe1_height-140

    clo1, col2, col3 = st.columns([0.5, 0.3, 0.2])
    with clo1:
        # 日期选择器
        date = str(st.date_input('选择日期', value='today', min_value=None, max_value=None, key=None))
    with col2:
        # 数据类型选择器，默认是'P'
        DataType = st.selectbox('数据类型', ['功率', '电量'], help='电量是根据功率数据计算得出')
    if DataType=='功率':
        with col3:
            show_raw_data = st.toggle('显示原始数据', False, help='是否显示原始数据，若为“是”，则可能减慢运行速度。')
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType='P')
    data_raw = df.copy()
    DataType = 'P' if DataType=='功率' else 'Energy'
    if DataType=='P':
        df = ReadData.TimeIntervalTransform(df, date, time_interval=TIME_INTERVAL, DataType=DataType)
        if show_raw_data:
            dataset = Form_Dataset(df, data_raw, DataType)
            figure = (
                Line(init_opts=opts.InitOpts(width='1000px', height=height))
                .add_dataset(source=dataset)
                .add_yaxis(series_name=DataType+'_Raw', y_axis=[], encode={'x': 'Time', 'y': 'RawData'}, is_connect_nones=False, itemstyle_opts=opts.ItemStyleOpts(color='gray', opacity=0.7))
                .add_yaxis(series_name=DataType, y_axis=[], encode={'x': 'Time', 'y': 'Data'}, is_connect_nones=True, linestyle_opts=opts.LineStyleOpts(width=2))
                

                .set_global_opts(
                    title_opts=opts.TitleOpts(title='日功率曲线' if DataType=='P' else '日电量曲线'),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                    yaxis_opts=opts.AxisOpts(name='功率(W)' if DataType=='P' else '电量(kWh)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False), type_='value'),
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
        else:
            x_axis = [i[-8:] for i in df['Time'].tolist()]
            figure = (
                Line(init_opts=opts.InitOpts(width='1000px', height=height))
                .add_xaxis(x_axis)
                .add_yaxis(DataType, df[DataType].tolist())

                .set_global_opts(
                    title_opts=opts.TitleOpts(title='日功率曲线' if DataType=='P' else '日电量曲线'),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                    yaxis_opts=opts.AxisOpts(name='功率(W)' if DataType=='P' else '电量(kWh)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False)),
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
        energy = ReadData.TimeIntervalTransform(df, date, time_interval=TIME_INTERVAL, DataType='P2Energy')
        # 画柱形图
        x_axis = [i[-8:] for i in energy['Time'].tolist()]
        figure = (
            Bar(init_opts=opts.InitOpts(width='1000px', height=height))
            .add_xaxis(x_axis)
            .add_yaxis('电量', energy['Energy'].tolist())

            .set_global_opts(
                title_opts=opts.TitleOpts(title='日电量曲线'),
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                yaxis_opts=opts.AxisOpts(name='电量(kWh)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False)),
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
    st_echarts(figure, height=height)



@st.cache_data(ttl=TIME_INTERVAL*60)
def calculate_start_and_end(date1):
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date1, PhoneNum=PhoneNum, password=password, DataType='P')
    df = ReadData.TimeIntervalTransform(df, date, time_interval=TIME_INTERVAL, DataType='P')
    time_list = [str(i)[-8:] for i in df['Time'].tolist()]
    data = df['P'].to_numpy()
    start_index = ReadData.find_change_point(data)
    end_index = 96-ReadData.find_change_point(data[::-1])-1
    return time_list,start_index, end_index

def start_and_end():
    '''
    用电开始和结束时间
    '''
    global date
    time_list, start_index, end_index = calculate_start_and_end(date)

    st.success('用电开始时间：'+time_list[start_index])
    st.warning('用电结束时间：'+time_list[end_index])

@st.cache_data(ttl=TIME_INTERVAL*60)
def Calculate_Energy_Sum(date):
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType='P')
    energy = ReadData.TimeIntervalTransform(df, date, time_interval=TIME_INTERVAL, DataType='P2Energy')
    energy_sum = energy['Energy'].sum()
    # 获取date上一天的用电量
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=str(pd.to_datetime(date)-pd.Timedelta(days=1)).split()[0], PhoneNum=PhoneNum, password=password, DataType='P')
    energy = ReadData.TimeIntervalTransform(df, str(pd.to_datetime(date)-pd.Timedelta(days=1)).split()[0], time_interval=TIME_INTERVAL, DataType='P2Energy')
    energy_sum_yesterday = energy['Energy'].sum()
    return energy_sum, energy_sum_yesterday

def Energy_Sum():
    '''
    日用电量
    '''
    global date
    energy_sum, energy_sum_yesterday = Calculate_Energy_Sum(date)
    st.metric(label='日用电量', value=str(round(energy_sum,2))+' kWh', delta=str(round(energy_sum-energy_sum_yesterday,2))+' kWh', delta_color='normal' if energy_sum-energy_sum_yesterday>0 else 'inverse')


def Show_Weather():
    '''展示天气情况'''
    global date
    condition, temp, humidity = ReadData.ReadWeather(date)
    weekday = ReadData.Each_Weekday(date)
    inner_temp = ReadData.ReadInnerTemperature(PhoneNum, password, date)
    card(
        title= weekday+' '+condition,
        text=['温度：'+str(temp)+'℃\n湿度：'+str(humidity)+'%', '室内温度：'+str(inner_temp)+'℃'],
        image= ReadData.image2base64('Pictures/clouds.jpg'),
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
            }

        }
    )

@st.cache_data(ttl=TIME_INTERVAL*60)
def Calculate_Peak_Valley_Prop(date):
    '''
    计算高峰用电占比和低谷用电量
    '''
    time_list, start_index, end_index = calculate_start_and_end(date)
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType='P')
    df = ReadData.TimeIntervalTransform(df, date, time_interval=TIME_INTERVAL, DataType='P2Energy')
    data = df['Energy'].to_numpy()
    peak_energy = data[start_index:end_index].sum()
    valley_energy = data[:start_index].sum()+data[end_index:].sum()
    return peak_energy, valley_energy

def Show_Peak_Prop():
    '''
    计算高峰用电占比和低谷用电占比
    '''
    global date
    peak_energy, valley_energy = Calculate_Peak_Valley_Prop(date)
    data = {'高峰用电':round(peak_energy,2), '低谷用电':round(valley_energy,2)}
    # 环形图
    figure = (
        Pie(init_opts=opts.InitOpts(width='100px', height='100px'))
        .add("", [list(z) for z in data.items()], radius=["40%", "75%"])
        
        .set_global_opts(
            title_opts={"text": "用电占比"},
            graphic_opts=[
                {
                    "type": "text",
                    "left": "center",
                    "top": "center",
                    "style": {
                        "text": "高峰占比\n\n"+str(round(data['高峰用电']/(data['高峰用电']+data['低谷用电']),2)*100)+'%',
                        "textAlign": "center",
                        "fill": "red",
                        "fontSize": 15,
                        # 加粗
                        "fontWeight": "bold",
                    },
                }
            ],
            legend_opts=opts.LegendOpts(pos_left="right", orient="vertical")
        )
        # .set_series_opts(label_opts={"formatter": "{b}"})
    )
    st_echarts(figure, height=250, width=250)

if __name__=='__main__':
    st.title('用电情况总览')

    # 用户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD
    BeeID = '86200001187'
    mac = 'Mt3-M1-84f703120b64'
    containe1_height = 700

    # 其他参数
    
    

    # 全局变量
    date = None
    # 如果高峰用电占比少于60%，则显示警告
    with st.container(height=containe1_height, border=True):
        col1, col2 = st.columns([8, 2])
        with col1:
            power_curve()
        with col2:
            Show_Weather()
            start_and_end()
            Energy_Sum()
            Show_Peak_Prop()
    if Calculate_Peak_Valley_Prop(date)[0]/(Calculate_Peak_Valley_Prop(date)[0]+Calculate_Peak_Valley_Prop(date)[1])<0.7:
        st.warning('高峰用电占比过少，请检查用电情况！')













