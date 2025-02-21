import streamlit as st
from utils import ReadData, IndexCalculator
from Globals import PHONE_NUM, PASSWORD
import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts as st_echarts
import numpy as np
from Globals import TIME_INTERVAL

from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=TIME_INTERVAL * 60 * 1000, key="autorefresh")

def Calculate_Parameters(dataset:pd.DataFrame, DataType:str):
    # 最大值
    a_max = dataset[DataType+'A'].max()
    b_max = dataset[DataType+'B'].max()
    c_max = dataset[DataType+'C'].max()
    a_max_time = dataset[dataset[DataType+'A']==a_max]['Time'].values[0]
    b_max_time = dataset[dataset[DataType+'B']==b_max]['Time'].values[0]
    c_max_time = dataset[dataset[DataType+'C']==c_max]['Time'].values[0]
    # 最小值
    a_min = dataset[DataType+'A'].min()
    b_min = dataset[DataType+'B'].min()
    c_min = dataset[DataType+'C'].min()
    a_min_time = dataset[dataset[DataType+'A']==a_min]['Time'].values[0]
    b_min_time = dataset[dataset[DataType+'B']==b_min]['Time'].values[0]
    c_min_time = dataset[dataset[DataType+'C']==c_min]['Time'].values[0]
    # 均值
    a_mean = dataset[DataType+'A'].mean()
    b_mean = dataset[DataType+'B'].mean()
    c_mean = dataset[DataType+'C'].mean()
    # 波动：对差分结果的绝对值进行操作
    if dataset.shape[0] == 1:
        var_a_mean,var_b_mean,var_c_mean = 0.,0.,0.
        var_a_max,var_b_max,var_c_max = 0.,0.,0.
        var_a_min,var_b_min,var_c_min = 0.,0.,0.
    else:
        a_diff = np.abs(np.diff(dataset[DataType+'A'].dropna().to_numpy()))
        b_diff = np.abs(np.diff(dataset[DataType+'B'].dropna().to_numpy()))
        c_diff = np.abs(np.diff(dataset[DataType+'C'].dropna().to_numpy()))
        var_a_mean,var_b_mean,var_c_mean = np.mean(a_diff), np.mean(b_diff), np.mean(c_diff)
        var_a_max,var_b_max,var_c_max = np.max(a_diff), np.max(b_diff), np.max(c_diff)
        var_a_min,var_b_min,var_c_min = np.min(a_diff), np.min(b_diff), np.min(c_diff)

    parameters_dict = {'a_max':a_max, 'b_max':b_max, 'c_max':c_max, 'a_max_time':a_max_time, 'b_max_time':b_max_time, 'c_max_time':c_max_time, 'a_min':a_min, 'b_min':b_min, 'c_min':c_min, 'a_min_time':a_min_time, 'b_min_time':b_min_time, 'c_min_time':c_min_time, 'a_mean':a_mean, 'b_mean':b_mean, 'c_mean':c_mean, 'var_a_mean':var_a_mean, 'var_b_mean':var_b_mean, 'var_c_mean':var_c_mean, 'var_a_max':var_a_max, 'var_b_max':var_b_max, 'var_c_max':var_c_max, 'var_a_min':var_a_min, 'var_b_min':var_b_min, 'var_c_min':var_c_min}
    for key in parameters_dict.keys():
        if key[-4:] != 'time':
            parameters_dict[key] = round(parameters_dict[key], 2)
    return parameters_dict

def Form_Dataset(df_PA:pd.DataFrame, df_PB:pd.DataFrame, df_PC:pd.DataFrame, DataType:str):
    '''
    生成显示三相数据的数据集
    '''
    dataset = pd.concat([df_PA[['Time', DataType+'A']].set_index('Time'), df_PB[['Time', DataType+'B']].set_index('Time'), df_PC[['Time', DataType+'C']].set_index('Time')], axis=1).sort_index().reset_index()
    dataset['Time'] =  dataset['Time'].apply(lambda x:x[-8:])
    dataset1 = [['Time', DataType+'A',  DataType+'B',  DataType+'C']] + dataset.to_numpy().tolist()
    # 删除最后一个幽灵数据
    return dataset1[:-1]
def Curve_Display():
    dataset = None
    unit = {'功率':'(W)', '电压':'(V)', '电流':'(A)'}

    col1,col2 = st.columns([0.7,0.3])
    with col1:
        col11,col12,col13 = st.columns([0.4,0.4,0.2])
        with col11:
            # 日期选择器
            date = str(st.date_input('选择日期', value='today', min_value=None, max_value=None, key=None))
        with col12:
            # 模式选择：整点值，最大值，最小值，完整曲线，默认：整点值
            mode = st.selectbox('选择模式', ['整点值', '最大值', '最小值', '完整曲线'], index=3, help='选择展示每五分钟的整点值、最值或者完整曲线。')
        with col13:
            Data_Type = st.selectbox('选择数据类型', ['功率', '电压', '电流'], index=0, help='选择展示功率、电压或者电流数据。')
            DataType = {'功率':'P', '电压':'U', '电流':'I'}[Data_Type]

        PA_raw = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType=DataType+'A')
        PB_raw = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType= DataType+'B')
        PC_raw = ReadData.ReadData_Day(beeId=BeeID, mac=mac, time=date, PhoneNum=PhoneNum, password=password, DataType= DataType+'C')

        if mode == '完整曲线':
            # A相：红色，B相：黄色，C相：蓝色
            dataset = Form_Dataset(PA_raw, PB_raw, PC_raw, DataType)
            color = {
                'A':'#EF2C2B',
                'B':'#23B2E0',
                'C':'#000000'
            }
            figure = (
                Line()

                .add_dataset(dataset)
                .add_yaxis('A', y_axis=[], encode={'x':'Time', 'y':DataType+'A'}, is_connect_nones=True, itemstyle_opts=opts.ItemStyleOpts(color=color['A']), linestyle_opts=opts.LineStyleOpts(color=color['A'], width=2))
                .add_yaxis('B', y_axis=[], encode={'x':'Time', 'y':DataType+'B'}, is_connect_nones=True, itemstyle_opts=opts.ItemStyleOpts(color=color['B']), linestyle_opts=opts.LineStyleOpts(color=color['B'], width=2))
                .add_yaxis('C', y_axis=[], encode={'x':'Time', 'y':DataType+'C'}, is_connect_nones=True, itemstyle_opts=opts.ItemStyleOpts(color=color['C']), linestyle_opts=opts.LineStyleOpts(color=color['C'], width=2))

                .set_global_opts(
                    title_opts=opts.TitleOpts(title='三相'+Data_Type+'曲线'),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger_on='mousemove', trigger='axis', axis_pointer_type='cross'),
                    yaxis_opts=opts.AxisOpts(name=Data_Type+unit[Data_Type], axislabel_opts=opts.LabelOpts(formatter='{value}'), splitline_opts=opts.SplitLineOpts(is_show=False), type_='value'),
                    xaxis_opts=opts.AxisOpts(name='时间', axislabel_opts=opts.LabelOpts(interval=100,rotate=45), splitline_opts=opts.SplitLineOpts(is_show=False), type_='category') 
                )

                .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=False),
                )
            )
        else:
            st.write('目前仅支持在完整曲线模式下查看数据。')
        if 'figure' in locals():
            st_echarts(figure, height=height-130)
    with col2:
        if dataset:
            dataset = pd.DataFrame(dataset[1:], columns=dataset[0])
            # 计算三相参数
            parameters_dict = Calculate_Parameters(dataset, DataType)
            

            st.markdown(
                """
                <style>
                .card {
                    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                    transition: 0.3s;
                    width: 100%;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                    backdrop-filter: blur(10px);  /* 添加磨砂质感 */
                    margin-bottom: 20px;  /* 添加垂直方向的间隔 */
                }
            
                .card:hover {
                    box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
                }
            
                .container {
                    padding: 2px 16px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # 最值、均值、波动
            st.markdown(
                f"""
                <div class="card" style="background: #EEBEC0;">
                    <div class="container">
                        <h4 class="title">最值：</h4>
                        <p class="subtitle"><b>A相：</b>{parameters_dict['a_max']}/{parameters_dict['a_min']} {unit[Data_Type][1]}，<b>时间：</b>{parameters_dict['a_max_time']}/{parameters_dict['a_min_time']}</p>
                        <p class="subtitle"><b>B相：</b>{parameters_dict['b_max']}/{parameters_dict['b_min']} {unit[Data_Type][1]}，<b>时间：</b>{parameters_dict['b_max_time']}/{parameters_dict['b_min_time']}</p>
                        <p class="subtitle"><b>C相：</b>{parameters_dict['c_max']}/{parameters_dict['c_min']} {unit[Data_Type][1]}，<b>时间：</b>{parameters_dict['c_max_time']}/{parameters_dict['c_min_time']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="card" style="background: #A5CC5B;">
                    <div class="container">
                        <h4 class="title">均值：</h4>
                        <p class="subtitle"><b>A相/B相/C相：</b>{parameters_dict['a_mean']}/{parameters_dict['b_mean']}/{parameters_dict['c_mean']} {unit[Data_Type][1]}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="card" style="background: #C2C2C2;">
                    <div class="container">
                        <h4 class="title">波动：</h4>
                        <p class="subtitle"><b>A相：</b>{parameters_dict['var_a_mean']}/{parameters_dict['var_a_max']}/{parameters_dict['var_a_min']} {unit[Data_Type][1]}    <b>B相：</b>{parameters_dict['var_b_mean']}/{parameters_dict['var_b_max']}/{parameters_dict['var_b_min']} {unit[Data_Type][1]}</p>
                        <p class="subtitle"><b>C相：</b>{parameters_dict['var_c_mean']}/{parameters_dict['var_c_max']}/{parameters_dict['var_c_min']} {unit[Data_Type][1]}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == '__page__':
    # 参数
    PhoneNum = PHONE_NUM
    password = PASSWORD
    BeeID = '86200001187'
    mac = 'Mt3-M1-84f703120b64'
    height = 600

    st.title('总体用电分析')

    with st.container(border=True, height=height):
        Curve_Display()

