import streamlit as st
from utils import ReadData
from Globals import PHONE_NUM, PASSWORD, TIME_INTERVAL, beeID, devices
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Line
from streamlit_echarts import st_pyecharts as st_echarts
import numpy as np

@st.cache_data(ttl=TIME_INTERVAL*60)
def Form_Dataset(df, data_raw, datatype):
    data_raw.rename(columns={datatype:datatype+'_Raw'}, inplace=True)
    dataset = pd.concat([df[['Time', datatype]].set_index('Time'), data_raw[['Time', datatype+'_Raw']].set_index('Time')], axis=1).sort_index().reset_index()
    dataset['Time'] =  dataset['Time'].apply(lambda x:x[-8:])
    dataset[datatype+'_Raw'][np.where(dataset[datatype+'_Raw'].isna().to_numpy())[0]] = dataset[datatype][np.where(dataset[datatype+'_Raw'].isna().to_numpy())[0]]
    dataset1 = [['Time', 'Data', 'RawData']] + dataset.to_numpy().tolist()
    # 这里注意需要删除最后一个幽灵数据，多出一根线的原因：最后一个数据和第一个数据的横坐标一样
    return dataset1[:-1]

def ShowDeviceUse():
    global date

    col1,col2,col3,col4 = st.columns([3,2,5,2])
    with col1:
        # 选择日期
        date = str(st.date_input('选择日期', value='today'))
    with col2:
        # 选择区域
        area = st.selectbox('选择区域', ['生活区', '学生办公区', '办公室/会议室'])
    with col3:
        # 选择设备
        device = st.selectbox('选择设备', list(devices[area].keys()))
    
    beeId = beeID[area]
    mac = devices[area][device]
    Datatype = 'Induction' if 'Irs' in mac else 'P'
    if Datatype == 'P':
        df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType=Datatype)
        if df.empty:
            st.write('暂时无数据。')
        else:
            if device == '打印机':
                df['Time'] = df['Time'].apply(lambda x: x[-8:])
                dataset = [['TimeStamp','P','Time']]+df.values.tolist()
                figure = (
                    Line()

                    .add_dataset(dataset)
                    .add_yaxis(Datatype, y_axis=[], encode={'x': 'Time', 'y': Datatype}, is_connect_nones=True)

                    .set_global_opts(
                        title_opts=opts.TitleOpts(title='日功率曲线'),
                        tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                        yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False), type_='value'),
                        xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=100, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False), type_='category'),
                    )

                    .set_series_opts(
                        label_opts=opts.LabelOpts(is_show=False)
                    )
                )
            else:
                with col4:
                    # 是否展示原曲线
                    show_raw_data = st.toggle('显示原始数据', False, help='是否显示原始数据，若为“是”，则可能减慢运行速度。')
                if not show_raw_data:
                    df = ReadData.TimeIntervalTransform(df, date=date, time_interval=TIME_INTERVAL, DataType=Datatype)
                    df['Time'] = df['Time'].apply(lambda x: x[-8:])
                    figure = (
                        Line()

                        .add_xaxis(df['Time'].to_list())
                        .add_yaxis(Datatype, df[Datatype], is_connect_nones=True)

                        .set_global_opts(
                            title_opts=opts.TitleOpts(title='日功率曲线'),
                            tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                            yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False)),
                            xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=4, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False)),
                        )

                        .set_series_opts(
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                    )
                else:
                    df_transformed = ReadData.TimeIntervalTransform(df, date=date, time_interval=TIME_INTERVAL, DataType=Datatype)
                    dataset = Form_Dataset(df_transformed, df, Datatype)
                    figure = (
                        Line(init_opts=opts.InitOpts())
                        .add_dataset(source=dataset)
                        .add_yaxis(series_name=Datatype+'_Raw', y_axis=[], encode={'x': 'Time', 'y': 'RawData'}, is_connect_nones=False, itemstyle_opts=opts.ItemStyleOpts(color='gray', opacity=0.7))
                        .add_yaxis(series_name=Datatype, y_axis=[], encode={'x': 'Time', 'y': 'Data'}, is_connect_nones=True, linestyle_opts=opts.LineStyleOpts(width=2))
                        

                        .set_global_opts(
                            title_opts=opts.TitleOpts(title='日功率曲线'),
                            tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                            yaxis_opts=opts.AxisOpts(name='功率(W)', axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False), type_='value'),
                            xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=30, rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False), type_='category'),
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
                    
    elif Datatype == 'Induction':
        df = ReadData.ReadData_Day(beeId=beeId, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType=Datatype)
        df['datetime'] = pd.to_datetime(df['TimeStamp'], unit='s')
        if df.empty:
            st.write('暂时无数据。')
        else:
            st.write('功能正在开发中')

    if 'figure' in locals():
        st_echarts(figure, height=500)
    pass


if __name__ == '__page__':

    # 账户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD

    date = None

    st.title('设备用电分析')
    with st.container(border=True):
        ShowDeviceUse()
    



























