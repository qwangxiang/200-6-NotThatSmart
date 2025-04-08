from sklearn.preprocessing import StandardScaler
from iTransformer.data.data_loader import DataLoader
import numpy as np
import pandas as pd

class DatasetBuilder:
    """适配新数据结构的构建器"""
    def __init__(self, data_loader, seq_length=96, pred_length=1, standardize=False):
        self.data_loader = data_loader
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.scalers = {}
        self.standardize = standardize
        
    def build_for_date_range(self, start_date, end_date, model_type='keras', split=True):
        """核心构建方法"""
        series = self.data_loader.preprocess().loc[start_date:end_date]
        df = self._create_feature_dataset(series)

        if self.standardize:
            # 应用标准化
            data_for_sequences, _ = self._standardize_data(df, model_type)
        else:
            # 跳过标准化，直接使用原始数据
            data_for_sequences = df.values if isinstance(df, pd.DataFrame) else df

        # 调试：检查关键节点数据形状
        # print("预处理后 series 长度:", len(series))  # 应 >= 11 * 96 = 1056
        # print("特征工程后 df 形状:", df.shape)       # 应 (1056, 特征数)
        # print("标准化后 scaled_data 形状:", scaled_data.shape)  # 应 (1056, 特征数)

        X, y = self._create_sequences(data_for_sequences, model_type)

        
        # 根据split参数决定是否划分
        if split:
            return self._split_data(X, y, model_type)
        else:
            if model_type == 'torch':
                X = np.transpose(X, (0, 2, 1))
                y = y[:, 0] if y.ndim > 1 else y
            else:  # keras 模型
                if X.ndim == 2:
                    X = X.reshape(-1, self.seq_length, X.shape[1])
            return X, y  # 直接返回完整数据c

    def _create_feature_dataset(self, series):
        """创建含时间特征的DataFrame"""
        df = pd.DataFrame({'load': series})
        
        # 时间特征
        df['HOUR'] = df.index.hour
        df['DAY_OF_WEEK'] = df.index.dayofweek
        df['MONTH'] = df.index.month
        df['IS_WEEKEND'] = df.index.dayofweek // 5
        
        return df.dropna()

    def _standardize_data(self, df, model_type):
        """标准化处理"""
        if model_type not in self.scalers:
            self.scalers[model_type] = StandardScaler()
            scaled = self.scalers[model_type].fit_transform(df)
        else:
            scaled = self.scalers[model_type].transform(df)
        return scaled, self.scalers[model_type]

    def _create_sequences(self, data, model_type):
        """生成序列，允许数据不足时截断"""
        if isinstance(data, pd.DataFrame):
            data = data.values

        X, y = [], []
        max_possible = len(data) - self.seq_length - self.pred_length + 1
        
        # 至少生成一个样本（如果数据不足，截断 pred_length）
        for i in range(max(max_possible, 0)):
            end_idx = i + self.seq_length
            target_idx = end_idx + self.pred_length
            
            # 处理数据不足的情况
            if end_idx > len(data):
                break  # 或填充数据
            if target_idx > len(data):
                # 允许部分目标值
                y_seq = data[end_idx:, 0]
            else:
                y_seq = data[end_idx:target_idx, 0]
            
            X_seq = data[i:end_idx]
            if X_seq.ndim == 1:
                X_seq = X_seq.reshape(-1, 1)
                
            X.append(X_seq)
            y.append(y_seq)
        
        return np.array(X), np.array(y)

    def _split_data(self, X, y, model_type):
        """数据划分逻辑"""
        split_idx = int(0.8 * len(X))
        
        if model_type == 'keras':
            return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
        else:  # PyTorch格式转换
            X_torch = np.transpose(X, (0, 2, 1))
            y_torch = y[:, 0] if y.ndim > 1 else y
            return X_torch[:split_idx], y_torch[:split_idx], X_torch[split_idx:], y_torch[split_idx:]
        
        # 在DatasetBuilder中增强特征工程方法

def create_enhanced_features(self, series):
    """创建增强版时间特征的DataFrame"""
    df = pd.DataFrame({'load': series})
    
    # 基本时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_holiday'] = 0  # 这里可以添加节假日标记逻辑
    
    # 周期性特征（循环编码）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 负荷趋势特征
    df['load_shift_1'] = df['load'].shift(1)   # 前15分钟
    df['load_shift_4'] = df['load'].shift(4)   # 前1小时
    df['load_shift_96'] = df['load'].shift(96)  # 前一天同时刻
    
    # 滑动窗口特征
    df['load_rolling_mean_4'] = df['load'].rolling(window=4).mean()  # 1小时平均
    df['load_rolling_mean_12'] = df['load'].rolling(window=12).mean()  # 3小时平均
    df['load_rolling_std_12'] = df['load'].rolling(window=12).std()   # 3小时标准差
    
    # 日间模式
    for h in range(0, 24, 3):  # 每3小时一个区间
        df[f'hour_group_{h}'] = ((df['hour'] >= h) & (df['hour'] < h+3)).astype(int)
    
    # 一周内的模式
    for d in range(7):
        df[f'dow_{d}'] = (df['day_of_week'] == d).astype(int)
    
    # 特征互动
    df['weekend_hour'] = df['is_weekend'] * df['hour']
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# 改进的序列创建方法
def create_enhanced_sequences(self, data, model_type, add_future_mask=True):
    """生成带有未来掩码的增强序列"""
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    X, y = [], []
    n_samples = len(data) - self.seq_length - self.pred_length + 1
    
    # 至少要有一个样本
    if n_samples <= 0:
        # 如果数据不足，发出警告但尝试获取至少一个样本
        print(f"警告: 数据长度不足以创建完整序列。数据长度: {len(data)}, 需要至少: {self.seq_length + self.pred_length}")
        n_samples = 1
        # 调整序列长度或预测长度以适应数据
        adjusted_seq_length = min(self.seq_length, len(data) - 1)
        adjusted_pred_length = min(self.pred_length, len(data) - adjusted_seq_length)
    else:
        adjusted_seq_length = self.seq_length
        adjusted_pred_length = self.pred_length
    
    # 创建序列
    for i in range(n_samples):
        end_idx = i + adjusted_seq_length
        target_idx = min(end_idx + adjusted_pred_length, len(data))
        
        # 特征序列
        X_seq = data[i:end_idx]
        
        # 目标序列
        y_seq = data[end_idx:target_idx, 0]  # 仅使用第一列作为目标
        
        if len(y_seq) < adjusted_pred_length and len(y_seq) > 0:
            # 填充预测序列至期望长度
            y_seq = np.pad(y_seq, (0, adjusted_pred_length - len(y_seq)), 'edge')
            
        # 如果使用未来掩码
        if add_future_mask and model_type == 'torch':
            # 添加未来时间步的掩码特征
            future_mask = np.zeros((adjusted_seq_length, 1))
            X_seq = np.concatenate([X_seq, future_mask], axis=1)
        
        if X_seq.size > 0 and y_seq.size > 0:
            X.append(X_seq)
            y.append(y_seq)
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("无法创建有效的序列。请检查数据长度和序列/预测长度参数。")
    
    return np.array(X), np.array(y)

# 改进的数据标准化方法
def standardize_enhanced_data(self, df, model_type, return_scalers=True):
    """使用分组标准化处理不同类型的特征"""
    # 如果df是DataFrame，则保存列名
    if isinstance(df, pd.DataFrame):
        columns = df.columns
        df_values = df.values
    else:
        columns = None
        df_values = df
    
    # 根据特征类型将列分组
    if columns is not None:
        # 负荷相关特征
        load_cols = [i for i, col in enumerate(columns) if 'load' in col]
        # 时间特征
        time_cols = [i for i, col in enumerate(columns) if any(x in col for x in ['hour', 'day', 'month', 'weekend'])]
        # 周期性特征
        cyclic_cols = [i for i, col in enumerate(columns) if any(x in col for x in ['_sin', '_cos'])]
        # 二值特征
        binary_cols = [i for i, col in enumerate(columns) if any(x in col for x in ['is_', 'dow_', 'hour_group_'])]
    else:
        # 默认分组
        n_cols = df_values.shape[1]
        load_cols = [0]  # 假设第一列是负荷
        time_cols = list(range(1, min(5, n_cols)))  # 假设紧随其后的是时间特征
        cyclic_cols = []
        binary_cols = []
    
    # 确保每个特征只属于一个组
    all_assigned = set(load_cols + time_cols + cyclic_cols + binary_cols)
    other_cols = [i for i in range(df_values.shape[1]) if i not in all_assigned]
    
    # 初始化标准化器
    if model_type not in self.scalers:
        self.scalers[model_type] = {}
        
        # 负荷特征标准化器
        if load_cols:
            self.scalers[model_type]['load'] = StandardScaler()
            self.scalers[model_type]['load'].fit(df_values[:, load_cols])
            
        # 时间特征标准化器
        if time_cols:
            self.scalers[model_type]['time'] = StandardScaler()
            self.scalers[model_type]['time'].fit(df_values[:, time_cols])
            
        # 其他特征标准化器
        if other_cols:
            self.scalers[model_type]['other'] = StandardScaler()
            self.scalers[model_type]['other'].fit(df_values[:, other_cols])
            
        # 周期性和二值特征不需要标准化
    
    # 创建标准化结果数组
    scaled_values = df_values.copy()
    
    # 应用标准化
    if load_cols:
        scaled_values[:, load_cols] = self.scalers[model_type]['load'].transform(df_values[:, load_cols])
    if time_cols:
        scaled_values[:, time_cols] = self.scalers[model_type]['time'].transform(df_values[:, time_cols])
    if other_cols:
        scaled_values[:, other_cols] = self.scalers[model_type]['other'].transform(df_values[:, other_cols])
    
    # 始终返回两个值，保持与原始接口兼容
    return scaled_values, self.scalers[model_type]