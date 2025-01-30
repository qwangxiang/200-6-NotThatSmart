import streamlit as st
from utils import ReadData
from Globals import PHONE_NUM, PASSWORD, TIME_INTERVAL
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

        # figure,ax = plt.subplots(figsize=(6,6))
        # df.plot(x='datetime', y='P', ax=ax)
        # st.write(figure)
    if 'figure' in locals():
        st_echarts(figure, height=500)
    pass


if __name__ == '__page__':

    # 账户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD

    # 其他变量
    beeID = {
        '生活区': '86200001187',
        '学生办公区': '86200001289',
        '办公室/会议室': '86200001290',
    }
    devices = {
        '生活区': {
            '打印机': 'Sck-M1-84f703123a18',
            '进门灯': 'Lk3-M1-7cdfa1b8a1a0',
            '小会议室灯': 'Lk1-M1-7cdfa1b8640c',
            '进门人体感应': 'Irs-M1-84f703112028',
            '展示区人体感应': 'Irs-M1-84f703122288',
            '小会议室人体感应': 'Irs-M1-84f7031218b4',
            '华为子路由器': 'Sck-M1-7cdfa1b660dc',
            '冰箱': 'Sck-M1-7cdfa1b852e0',
            '网络设备': 'Sck-M1-7cdfa1b89d5',
            # 这里实际上咖啡机和烧水壶的mac示范的
            '咖啡机': 'Sck-M1-84f703123c88',
            '烧水壶': 'Sck-M1-7cdfa1b89d20',
            # 这个mac也是改过的
            '微波炉': 'Sck-M1-84f70310ee40',
        },
        '学生办公区': {
            '人体感应1': 'Irs-M1-7cdfa1b84cb4',
            '人体感应2': 'Irs-M1-7cdfa1b85e28',
        },
        '办公室/会议室': {
            # 这几个人感不知道为什么查不到数据
            '大会议室人体感应器': 'Irs-M1-84f703101f5c',
            '办公室B人体感应器': 'Irs-M1-84f70310d0f4',
            '办公室B灯': 'Lk1-M1-7cdfa1b867d8',
            '办公室C人体感应器': 'Irs-M1-7cdfa1b85e50',
            '办公室C灯': 'Lk1-M1-7cdfa1b87b18',
            '办公室C空气传感器': 'Env-lt_0004@modbus 01',
            '办公室D灯': 'Lk1-M1-7cdfa1b84bc4',
            '办公室D空气传感器': 'Env-lt_0001@modbus 01',
            '大会议室灯': 'Lk2-M1-7cdfa1b86a10',
        }
    }

    date = None

    st.title('设备用电分析')
    with st.container(border=True):
        ShowDeviceUse()
    



























