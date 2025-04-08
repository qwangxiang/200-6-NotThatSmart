import pandas as pd
from pathlib import Path

class DataLoader:
    """专注负荷数据处理的核心加载器"""
    def __init__(self, data_path='./loaddata.csv'):
        self.data_path = data_path
        self.load_data = None  # 确保初始化时设置默认值
        
    def load_raw_data(self):
        """加载原始数据并设置索引"""
        self.load_data = pd.read_csv(
            self.data_path,
            sheet_name=0,
            parse_dates=['YMD'],
            index_col='YMD'
        )
        return self.load_data
    
    def preprocess(self):
        """核心预处理：生成连续时间序列"""
        # 确保数据已加载
        if self.load_data is None:
            self.load_raw_data()
            
        # 转换时间索引
        if not isinstance(self.load_data.index, pd.DatetimeIndex):
            self.load_data.index = pd.to_datetime(
                self.load_data.index.astype(str),
                format='%Y%m%d'
            )
            
        # 重塑为长格式时间序列
        time_series = self.load_data.stack().reset_index()
        time_series.columns = ['date', 'time', 'load']
        
        # 生成精确时间戳
        time_series['datetime'] = pd.to_datetime(
            time_series['date'].astype(str) + ' ' + 
            time_series['time'].str[1:3] + ':' + 
            time_series['time'].str[3:5]
        )
        
        return time_series.set_index('datetime')['load'].sort_index()