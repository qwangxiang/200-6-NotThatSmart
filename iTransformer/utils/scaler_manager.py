import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class ScalerManager:
    """标准化管理器，用于管理模型的训练和预测中的数据缩放"""
    
    def __init__(self, scaler_path='models/scalers'):
        self.scaler_path = scaler_path
        self.scalers = {}
        # 确保目录存在
        os.makedirs(scaler_path, exist_ok=True)
    
    def fit(self, name, data):
        """拟合并保存标准化器"""
        # 确保数据为二维
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        scaler = StandardScaler()
        scaler.fit(data)
        
        # 保存到内部字典和文件
        self.scalers[name] = scaler
        return self.save_scaler(name, scaler)
    
    def transform(self, name, data):
        """使用指定的标准化器转换数据，增强型"""
        scaler = self.get_scaler(name)
        if scaler is None:
            raise ValueError(f"标准化器 '{name}' 不存在，请先训练")
        
        # 记录原始数据形状和维度
        original_shape = data.shape
        original_ndim = len(original_shape)
        # print(f"DEBUG: transform() - 原始数据形状: {original_shape}, 维度: {original_ndim}")
        
        # 将数据转换为2D (必须的，StandardScaler只能处理2D数据)
        if original_ndim == 1:
            # 1D数组转换为列向量
            data_2d = data.reshape(-1, 1)
            # print(f"DEBUG: transform() - 将1D数据转为2D: {data_2d.shape}")
        elif original_ndim >= 3:
            # 高维数据展平为2D (每行是一个样本)
            data_2d = data.reshape(original_shape[0], -1)
            # print(f"DEBUG: transform() - 将高维数据转为2D: {data_2d.shape}")
        else:
            # 已经是2D，直接使用
            data_2d = data
        
        # 应用缩放
        scaled_data_2d = scaler.transform(data_2d)
        # print(f"DEBUG: transform() - 标准化后2D数据形状: {scaled_data_2d.shape}")
        
        # 将缩放后的数据恢复为原始形状
        if original_ndim == 1:
            scaled_data = scaled_data_2d.flatten()
        elif original_ndim >= 3:
            scaled_data = scaled_data_2d.reshape(original_shape)
        else:
            scaled_data = scaled_data_2d
        
        # print(f"DEBUG: transform() - 恢复后数据形状: {scaled_data.shape}")
        return scaled_data

    def inverse_transform(self, name, data):
        """对标准化的数据进行反标准化，增强型"""
        scaler = self.get_scaler(name)
        if scaler is None:
            raise ValueError(f"标准化器 '{name}' 不存在，请先训练")
        
        # 记录原始数据形状和维度
        original_shape = data.shape
        original_ndim = len(original_shape)
        # print(f"DEBUG: inverse_transform() - 原始数据形状: {original_shape}, 维度: {original_ndim}")
        
        # 将数据转换为2D (必须的，StandardScaler只能处理2D数据)
        if original_ndim == 1:
            # 1D数组转换为列向量
            data_2d = data.reshape(-1, 1)
            # print(f"DEBUG: inverse_transform() - 将1D数据转为2D: {data_2d.shape}")
        elif original_ndim >= 3:
            # 高维数据展平为2D (每行是一个样本)
            data_2d = data.reshape(original_shape[0], -1)
            # print(f"DEBUG: inverse_transform() - 将高维数据转为2D: {data_2d.shape}")
        else:
            # 已经是2D，直接使用
            data_2d = data
        
        # 应用反缩放
        original_data_2d = scaler.inverse_transform(data_2d)
        # print(f"DEBUG: inverse_transform() - 反标准化后2D数据形状: {original_data_2d.shape}")
        
        # 将反缩放后的数据恢复为原始形状
        if original_ndim == 1:
            original_data = original_data_2d.flatten()
        elif original_ndim >= 3:
            original_data = original_data_2d.reshape(original_shape)
        else:
            original_data = original_data_2d
        
        # print(f"DEBUG: inverse_transform() - 恢复后数据形状: {original_data.shape}")
        return original_data

    def fit_transform(self, name, data):
        """拟合标准化器并立即转换数据"""
        # 确保数据为二维
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            was_1d = True
        else:
            was_1d = False
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 保存到内部字典和文件
        self.scalers[name] = scaler
        self.save_scaler(name, scaler)
        
        # 如果输入是一维，将输出恢复为一维
        if was_1d:
            scaled_data = scaled_data.flatten()
            
        return scaled_data
    
    def save_scaler(self, name, scaler):
        """保存标准化器到文件"""
        save_path = os.path.join(self.scaler_path, f'{name}_scaler.pkl')
        joblib.dump(scaler, save_path)
        print(f"已保存标准化器 '{name}' 到 {save_path}")
        return save_path
    
    def load_scaler(self, name):
        """从文件加载标准化器"""
        path = os.path.join(self.scaler_path, f'{name}_scaler.pkl')
        if Path(path).exists():
            scaler = joblib.load(path)
            self.scalers[name] = scaler
            print(f"从 {path} 加载标准化器 '{name}'")
            return scaler
        return None
    
    def get_scaler(self, name):
        """获取缓存的标准化器，如果不存在则尝试从文件加载"""
        if name in self.scalers:
            return self.scalers[name]
        
        return self.load_scaler(name)
    
    def has_scaler(self, name):
        """检查是否存在指定名称的标准化器"""
        if name in self.scalers:
            return True
            
        path = os.path.join(self.scaler_path, f'{name}_scaler.pkl')
        return Path(path).exists()
    
    def reset_scaler(self, name):
        """重置指定的标准化器"""
        if name in self.scalers:
            del self.scalers[name]
        
        scaler_path = os.path.join(self.scaler_path, f'{name}_scaler.pkl')
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            print(f"已删除标准化器文件: {scaler_path}")
        
        return not self.has_scaler(name)

    def reset_all_scalers(self):
        """重置所有标准化器"""
        scaler_names = list(self.scalers.keys())
        for name in scaler_names:
            self.reset_scaler(name)
        
        # 检查目录中的所有标准化器文件
        for file in os.listdir(self.scaler_path):
            if file.endswith('_scaler.pkl'):
                os.remove(os.path.join(self.scaler_path, file))
                print(f"已删除标准化器文件: {file}")
        
        return len(self.scalers) == 0
    
    def get_range(self, name):
        """获取标准化器的缩放范围"""
        scaler = self.get_scaler(name)
        if scaler is None:
            return None
        
        try:
            # 尝试获取StandardScaler的范围，这通常是通过mean_和scale_属性
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                # 计算近似可逆变换的范围
                # StandardScaler(X) = (X - mean) / scale
                # 如果输入值为min_val和max_val，则输出范围为
                min_out = (0 - scaler.mean_[0]) / scaler.scale_[0]  # 假设输入为0的输出
                max_out = (1000 - scaler.mean_[0]) / scaler.scale_[0]  # 假设输入为1000的输出
                return (min_out, max_out)
        except Exception as e:
            print(f"无法计算标准化器 '{name}' 的范围: {e}")
        
        return None