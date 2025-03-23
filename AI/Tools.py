from Globals import devices_lib
from langchain_core.tools import StructuredTool
from utils import ReadData
import datetime
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line
from streamlit_echarts import st_pyecharts as st_echarts


'''
定义工具函数
'''
tools = []

# 设备信息查询工具
def Get_Info(Device_Name:str)->dict:
    '''
    获取设备信息的，输入是设备的名字，输出是设备的信息字典，该字典包含两个键，分别是设备的beeID和mac，这二者可以用于查询数据
    '''
    return devices_lib[Device_Name] if Device_Name in devices_lib.keys() else None
Get_Device_Info_tool = StructuredTool.from_function(
    func=Get_Info,
    name='查询设备信息工具',
    description='传入设备名称，查询设备的beeID和mac。'
)

# 日期字符串构造工具
def Get_Date_Str(date:str='today', year:int=2025)->str:
    return str(datetime.datetime.now().date()) if date=='today' else f'{year}-{date[:2]}-{date[2:]}'
Get_Date_Str_tool = StructuredTool.from_function(
    func=Get_Date_Str,
    name='日期构造工具',
    description='传入日期和年份，构造符合格式要求的日期字符串。'
)

# 设备数据查询工具
def Read_Device_Data(Device_name:str, Date:str)->np.ndarray:
    beeID,mac = Get_Info(Device_name).values()
    df = ReadData.ReadData_Day(beeId=beeID, mac=mac, time=Date, PhoneNum='15528932507', password='123456', DataType='P')
    return df['P'].to_numpy()
Get_Device_Data_tool = StructuredTool.from_function(
    func=Read_Device_Data,
    name='查询设备数据工具',
    description='传入设备名称和日期，查询设备在指定日期的功率序列。'
)

# 图表工具
def Figure_Tool(data:str):
    '''
    画图工具
    '''
    def str_to_list(s):
        return [float(x) for x in s.strip('[]').split()]
    data = str_to_list(data)

    figure = (
        Line()
        .add_xaxis(list(range(len(data))))
        .add_yaxis('功率', data)
        .set_global_opts(title_opts=opts.TitleOpts(title='功率序列'))
    )
    st_echarts(figure)
    # 以字符串形式返回figure的html代码
    return figure.render_embed()



Get_Figure_tool = StructuredTool.from_function(
    func=Figure_Tool,
    name='图表工具',
    description='根据传入的数据画出对应的echarts图表，并以字符串的格式返回图表的html代码。'
)

def Get_Tools():
    tools = [
        Get_Device_Info_tool,
        Get_Device_Data_tool,
        Get_Date_Str_tool,
        Get_Figure_tool
    ]
    return tools


if __name__ == '__main__':
    pass

