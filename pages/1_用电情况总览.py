from utils import ReadData
import streamlit as st
from streamlit_echarts import st_pyecharts as st_echarts
from pyecharts.charts import Bar,Line,HeatMap
from pyecharts import options as opts
import time
import numpy as np
import pandas as pd

st.set_page_config(layout='wide')

@st.cache_data
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
        df = ReadData.TimeIntervalTransform(df, date, time_interval=15, DataType=DataType)
        if show_raw_data:
            dataset = Form_Dataset(df, data_raw, DataType)
            figure = (
                Line(init_opts=opts.InitOpts(width='1000px'))
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
                Line(init_opts=opts.InitOpts(width='1000px', height='1000px'))
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
        energy = ReadData.TimeIntervalTransform(df, date, time_interval=15, DataType='P2Energy')
        # 画柱形图
        x_axis = [i[-8:] for i in energy['Time'].tolist()]
        figure = (
            Bar(init_opts=opts.InitOpts(width='1000px', height='1000px'))
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
    st_echarts(figure, height=400)

def find_change_point(data):
    # 这里默认是96个点
    length = data.shape[0]
    # 求差分
    diff = np.diff(data)
    for i in range(length-num_points-1):
        if_point = True
        for j in range(num_points):
            if abs(diff[i+j])>change_lower:
                if_point = False
        if diff[i+num_points]<change_upper:
            if_point = False
        if if_point:
            return i+num_points
    return 0

def start_and_end():
    '''
    用电开始和结束时间
    '''
    global date

    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType='P')
    df = ReadData.TimeIntervalTransform(df, date, time_interval=15, DataType='P')
    data = df['P'].to_numpy()
    time_list = [str(i)[-8:] for i in df['Time'].tolist()]
    start_index = find_change_point(data)
    end_index = 96-find_change_point(data[::-1])-1

    st.success('用电开始时间：'+time_list[start_index])
    st.warning('用电结束时间：'+time_list[end_index])

def Energy_Sum():
    '''
    日用电量
    '''
    global date

    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType='P')
    energy = ReadData.TimeIntervalTransform(df, date, time_interval=15, DataType='P2Energy')
    energy_sum = energy['Energy'].sum()
    # 获取date上一天的用电量
    df = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=str(pd.to_datetime(date)-pd.Timedelta(days=1)).split()[0], PhoneNum=PhoneNum, password=password, DataType='P')
    energy = ReadData.TimeIntervalTransform(df, str(pd.to_datetime(date)-pd.Timedelta(days=1)).split()[0], time_interval=15, DataType='P2Energy')
    energy_sum_yesterday = energy['Energy'].sum()

    st.metric(label='日用电量', value=str(energy_sum)+' kWh', delta=str(round(energy_sum-energy_sum_yesterday,2))+' kWh', delta_color='normal' if energy_sum-energy_sum_yesterday>0 else 'inverse')




if __name__=='__main__':
    st.title('用电情况总览')

    # 用户信息
    PhoneNum = '15528932507'
    password = '123456'
    BeeID = '86200001187'
    mac = 'Mt3-M1-84f703120b64'

    # 其他参数
    time_interval = 15
    change_lower = 400
    change_upper = 500
    num_points = 2

    # 全局变量
    date = None

    with st.container(height=530, border=True):
        col1, col2 = st.columns([8, 2])
        with col1:
            power_curve()
        with col2:
            start_and_end()
            Energy_Sum()
            st.write('这边似乎应该有点什么东西，但我还没想到......')













