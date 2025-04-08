# torch_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import math
from iTransformer.models.abstract_model import AbstractTimeSeriesModel
import matplotlib.gridspec as gridspec

# 改进的学习率调度器
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    余弦退火学习率调度器，带预热阶段
    """
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 退火阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # 确保学习率不低于最小值
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class TemporalDataset(Dataset):
    """统一时序数据集类"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        # 确保targets的形状正确
        if len(targets.shape) == 1:
            # 如果targets是1D，将其重塑为[num_samples, 1]
            targets = targets.reshape(-1, 1)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, true):
        # 调整维度以确保兼容
        if pred.shape != true.shape:
            # 情况1: pred是[batch, pred_len]，true是[batch]
            if len(pred.shape) == 2 and len(true.shape) == 1:
                # 只使用第一个预测值进行比较，或者将true扩展为[batch, pred_len]
                # 方案A: 只用第一个预测值
                # pred = pred[:, 0]
                
                # 方案B: 将true扩展为[batch, pred_len]
                true = true.unsqueeze(1).expand(-1, pred.shape[1])
            
            # 情况2: pred是[batch]，true是[batch, pred_len]
            elif len(pred.shape) == 1 and len(true.shape) == 2:
                # 将pred扩展为[batch, pred_len]
                pred = pred.unsqueeze(1).expand(-1, true.shape[1])
        
        # 计算MSE损失
        mse_loss = self.mse(pred, true)
        
        # 计算MAPE损失，避免除以0
        epsilon = 1e-8
        mape_loss = torch.mean(torch.abs((true - pred) / (torch.abs(true) + epsilon)))
        
        return self.alpha * mse_loss + (1 - self.alpha) * mape_loss

class TemporalConvLSTM(nn.Module):
    """时空融合预测模型"""
    def __init__(self, input_size, seq_len=96, pred_len=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  # 添加BatchNorm
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # 时间特征提取
        self.temporal_lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # 预测头
        self.fc = nn.Sequential(
            nn.Linear(128 * (seq_len//2), 256),
            nn.BatchNorm1d(256),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len)
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, num_features)
        batch_size = x.size(0)
        
        # 空间特征提取
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = self.spatial_conv(x)  # (batch, 64, seq//2)
        
        # 时间特征提取
        x = x.permute(0, 2, 1)  # (batch, seq//2, 64)
        lstm_out, _ = self.temporal_lstm(x)  # (batch, seq//2, 128)
        
        # 特征融合
        x = lstm_out.contiguous().view(batch_size, -1)  # (batch, seq//2 * 128)
        
        # print(f"中间特征形状: {x.shape}")
        
        # 预测输出 - 注意我们不使用额外的激活函数，直接输出原始值
        return self.fc(x)

class TorchForecaster:
    """PyTorch预测器封装类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch 使用设备: {self.device}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            print(f"当前内存使用量: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def optimized_train(self, data_dict):
        """优化的训练流程"""
        # 初始化模型
        input_size = data_dict['train'][0].shape[-1]
        self.model = TemporalConvLSTM(
            input_size=input_size,
            seq_len=self.config['seq_length'],
            pred_len=self.config['pred_length']
        ).to(self.device)
        
        # 自动优化批量大小和学习率
        self.find_optimal_batch_size(data_dict)
        
        # # 使用torch.compile加速（如果可用）
        # if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        #     self.model = torch.compile(self.model, mode='reduce-overhead')
        #     print("已启用torch.compile加速")
        
        # 设置优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-4)
        
        # 余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['epochs'], eta_min=self.config['lr'] / 10
        )
        
        # 使用组合损失函数
        criterion = CombinedLoss(alpha=0.7)
        
        # 创建数据加载器
        train_dataset = TemporalDataset(*data_dict['train'])
        val_dataset = TemporalDataset(*data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # 初始化混合精度训练
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        
        # 训练监控变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        # 训练循环
        from time import time
        total_start = time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # 处理维度不匹配
                if batch_y.dim() == 1 and self.config['pred_length'] > 1:
                    batch_y = batch_y.unsqueeze(1).expand(-1, self.config['pred_length'])
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # 使用混合精度
                with autocast():
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                
                # 梯度缩放
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                
                train_loss += loss.item() * batch_x.size(0)
            
            # 计算平均训练损失
            train_loss /= len(train_loader.dataset)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    # 处理维度不匹配
                    if batch_y.dim() == 1 and self.config['pred_length'] > 1:
                        batch_y = batch_y.unsqueeze(1).expand(-1, self.config['pred_length'])
                    
                    # 对验证也使用混合精度
                    with autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_x.size(0)
            
            # 计算平均验证损失
            val_loss /= len(val_loader.dataset)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
                print(f"✓ 模型改进，已保存检查点")
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"⚠ {self.config['patience']}个epoch没有改进，提前停止训练")
                    break
            
            epoch_time = time() - epoch_start
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
                f"学习率: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"耗时: {epoch_time:.2f}秒")
            
            # 打印GPU内存使用情况
            if torch.cuda.is_available():
                used_mem = torch.cuda.memory_allocated() / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU内存使用: {used_mem:.2f}GB / {total_mem:.2f}GB")
        
        total_time = time() - total_start
        print(f"训练完成，总耗时: {total_time:.2f}秒，最佳验证损失: {best_val_loss:.4f}")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.model

    def find_optimal_batch_size(self, data_dict, initial_batch_size=32, max_batch_size=512):
        """查找最佳批量大小"""
        # 使用配置中的批量大小作为初始值
        if initial_batch_size is None:
            initial_batch_size = self.config['batch_size']
        
        batch_size = initial_batch_size
        original_lr = self.config['lr']
        
        # 获取输入形状
        if not hasattr(self, 'model') or self.model is None:
            print("警告：模型尚未初始化，无法进行批量大小优化")
            return batch_size
        
        print(f"开始寻找最佳批量大小（初始：{batch_size}）...")
        
        while batch_size < max_batch_size:
            try:
                # 尝试更大的批量大小
                next_batch_size = batch_size * 2
                
                # 创建一个测试批次
                dummy_input = torch.zeros((next_batch_size, self.config['seq_length'], 
                                        data_dict['train'][0].shape[-1])).to(self.device)
                
                # 测试前向传播
                self.model(dummy_input)
                
                # 如果成功，更新批量大小
                batch_size = next_batch_size
                torch.cuda.empty_cache()  # 清空缓存
                
                print(f"  测试批量大小 {batch_size} 成功")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  批量大小 {batch_size*2} 内存不足")
                    break
                else:
                    print(f"  出现非内存错误: {e}")
                    break
        
        # 根据批量大小调整学习率
        self.config['batch_size'] = batch_size
        self.config['lr'] = original_lr * (batch_size / initial_batch_size)  # 按比例调整学习率
        
        print(f"已优化批量大小为: {batch_size}, 学习率调整为: {self.config['lr']:.2e}")
        return batch_size

    def train(self, data_dict):
        """训练入口方法"""
        return self.optimized_train(data_dict)


    def train_backup(self, data_dict):
        """完整训练流程"""
        # 初始化模型
        input_size = data_dict['train'][0].shape[-1]
        self.model = TemporalConvLSTM(
            input_size=input_size,
            seq_len=self.config['seq_length'],
            pred_len=self.config['pred_length']
        ).to(self.device)
        
        print(f"模型已移至 {self.device} 设备")
        if torch.cuda.is_available():
            print(f"训练开始时 GPU 内存使用量: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
        
        # 设置优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-4)
        
        # 学习率调度 - 使用余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['epochs'],
            eta_min=self.config['lr'] / 10
        )
        
        # 使用组合损失函数
        criterion = CombinedLoss(alpha=0.7)
        
        # 创建数据加载器
        train_dataset = TemporalDataset(*data_dict['train'])
        val_dataset = TemporalDataset(*data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_x)
                
                # 确保形状匹配
                if outputs.shape != batch_y.shape:
                    print(f"警告: 输出形状 {outputs.shape} 与目标形状 {batch_y.shape} 不匹配")
                    # 可能需要调整形状
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪以避免梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    outputs = self.model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
            
            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 打印训练信息
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
            # 在每个epoch结束后打印GPU内存使用情况
            if torch.cuda.is_available():
                print(f"Epoch {epoch+1} GPU 内存使用量: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
        
        # 训练结束后绘制损失曲线
        os.makedirs('results', exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/training_loss.png')
        plt.close()
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

    def improved_train(self, data_dict):
        """改进的训练流程"""
        # 初始化模型 - 使用增强版模型
        input_size = data_dict['train'][0].shape[-1]
        self.model = EnhancedTemporalModel(
            input_size=input_size,
            seq_len=self.config['seq_length'],
            pred_len=self.config['pred_length']
        ).to(self.device)
        
        print(f"模型已移至 {self.device} 设备")
        
        # 设置优化器 - 使用RAdam优化器
        try:
            self.optimizer = optim.RAdam(
                self.model.parameters(), 
                lr=self.config['lr'], 
                weight_decay=1e-4
            )
        except AttributeError:
            # 如果RAdam不可用，回退到AdamW
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config['lr'], 
                weight_decay=1e-4
            )
        
        # 使用自动权重损失函数
        criterion = AutoWeightedLoss().to(self.device)
        
        # 创建数据加载器
        train_dataset = TemporalDataset(*data_dict['train'])
        val_dataset = TemporalDataset(*data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        # 使用改进的学习率调度器
        total_steps = self.config['epochs'] * len(train_loader)
        warmup_steps = total_steps // 10  # 10%的步骤用于预热
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练监控
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_x)
                
                # 确保形状匹配 - 处理维度问题
                if outputs.shape != batch_y.shape:
                    if len(outputs.shape) == 2 and len(batch_y.shape) == 2:
                        # 只有最后一个维度可能不匹配
                        min_dim = min(outputs.shape[1], batch_y.shape[1])
                        outputs = outputs[:, :min_dim]
                        batch_y = batch_y[:, :min_dim]
                    else:
                        print(f"警告: 输出形状 {outputs.shape} 与目标形状 {batch_y.shape} 不匹配")
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪以避免梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()  # 每批次更新学习率
                
                train_loss += loss.item() * batch_x.size(0)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_trues = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    outputs = self.model(batch_x)
                    
                    # 确保形状匹配
                    if outputs.shape != batch_y.shape:
                        if len(outputs.shape) == 2 and len(batch_y.shape) == 2:
                            min_dim = min(outputs.shape[1], batch_y.shape[1])
                            outputs = outputs[:, :min_dim]
                            batch_y = batch_y[:, :min_dim]
                    
                    # 收集预测和真实值用于计算指标
                    val_preds.append(outputs.cpu().numpy())
                    val_trues.append(batch_y.cpu().numpy())
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 计算性能指标
            val_preds = np.concatenate(val_preds, axis=0)
            val_trues = np.concatenate(val_trues, axis=0)
            
            val_mae = np.mean(np.abs(val_preds - val_trues))
            val_rmse = np.sqrt(np.mean((val_preds - val_trues) ** 2))
            
            # 打印训练信息
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | LR: {lr:.2e}")
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
                
                # 保存当前最佳模型的预测结果可视化
                self._plot_validation_results(val_trues, val_preds, epoch+1)
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 训练结束后绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/training_loss.png')
        plt.close()
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.model

    # 添加到TorchForecaster类的辅助方法
    def _plot_validation_results(self, y_true, y_pred, epoch):
        """绘制验证集预测结果"""
        # 扁平化数组便于绘图
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 确保长度一致
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # 计算指标
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-10))) * 100
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制预测对比
        plt.subplot(2, 1, 1)
        plt.plot(y_true_flat[:96], 'b-', label='真实值', linewidth=2)
        plt.plot(y_pred_flat[:96], 'r--', label='预测值', linewidth=2)
        plt.title(f'验证集预测结果 (Epoch {epoch}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')
        plt.xlabel('时间步 (15分钟间隔)')
        plt.ylabel('负荷')
        plt.legend()
        plt.grid(True)
        
        # 绘制误差
        plt.subplot(2, 1, 2)
        error = y_true_flat[:96] - y_pred_flat[:96]
        plt.bar(range(len(error)), error, color='g', alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('预测误差')
        plt.xlabel('时间步 (15分钟间隔)')
        plt.ylabel('误差')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/validation_epoch_{epoch}.png')
        plt.close()

    def evaluate(self, data_dict):
        """模型评估"""
        test_dataset = TemporalDataset(*data_dict['test'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        predictions = []  # 确保这是一个列表，而不是NumPy数组
        actuals = []      # 确保这是一个列表，而不是NumPy数组
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).detach().cpu().numpy()
                
                # 如果是多维输出，我们可能需要选择特定的维度
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    # 这里我们假设预测的是多个时间步，我们取所有时间步
                    for i in range(outputs.shape[0]):
                        predictions.extend(outputs[i])
                else:
                    # 单一预测值，直接添加
                    predictions.extend(outputs.flatten())
                
                # 同样处理实际值
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    for i in range(batch_y.shape[0]):
                        actuals.extend(batch_y[i].cpu().numpy())
                else:
                    actuals.extend(batch_y.cpu().numpy().flatten())
        
        # 转换为numpy数组（只在最后返回前转换）
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        # 计算指标
        mae = np.mean(np.abs(predictions_array - actuals_array))
        rmse = np.sqrt(np.mean((predictions_array - actuals_array)**2))
        mape = np.mean(np.abs((actuals_array - predictions_array) / (np.abs(actuals_array) + 1e-8))) * 100
        
        print(f"\nTest Metrics:")
        print(f"MAE: {mae:.2f} MW")
        print(f"RMSE: {rmse:.2f} MW")
        print(f"MAPE: {mape:.2f}%")
        
        # 可视化结果
        self._plot_results(actuals_array, predictions_array)
        
        return predictions_array, actuals_array
    
    def _plot_results(self, actuals, predictions, num_samples=96):
        """结果可视化"""
        plt.figure(figsize=(15, 6))
        
        # 选择最近24小时数据
        plot_len = min(len(actuals), num_samples)
        plot_actual = actuals[:plot_len]
        plot_pred = predictions[:plot_len]
        
        plt.plot(plot_actual, label='实际负荷', alpha=0.8)
        plt.plot(plot_pred, '--', label='预测负荷')
        
        plt.title('负荷预测结果 (24小时)')
        plt.xlabel('时间步 (15分钟间隔)')
        plt.ylabel('电力负荷 (MW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/forecast_results.png')
        plt.close()

    def enhanced_evaluate(self, data_dict):
        """增强版评估方法"""
        test_dataset = TemporalDataset(*data_dict['test'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).detach().cpu().numpy()
                
                # 保存结果
                predictions.append(outputs)
                actuals.append(batch_y.numpy())
        
        # 合并结果
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # 确保形状一致性
        if predictions.shape != actuals.shape:
            min_shape = [min(p, a) for p, a in zip(predictions.shape, actuals.shape)]
            predictions = predictions[:min_shape[0], :min_shape[1]] if len(predictions.shape) > 1 else predictions[:min_shape[0]]
            actuals = actuals[:min_shape[0], :min_shape[1]] if len(actuals.shape) > 1 else actuals[:min_shape[0]]
        
        # 计算指标
        metrics = self._calculate_metrics(actuals, predictions)
        
        # 打印指标
        print("\n==== 测试集评估结果 ====")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # 可视化结果
        self._enhanced_plot_results(actuals, predictions)
        
        # 返回结果，便于进一步分析
        return predictions, actuals, metrics

    def _calculate_metrics(self, y_true, y_pred):
        """计算多种评估指标"""
        # 确保是一维数组
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 截取至相同长度
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # 均方误差（MSE）
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        
        # 均方根误差（RMSE）
        rmse = np.sqrt(mse)
        
        # 平均绝对误差（MAE）
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        
        # 平均绝对百分比误差（MAPE）
        # 避免除以零
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + epsilon))) * 100
        
        # 对称平均绝对百分比误差（SMAPE）
        # 更稳健的指标
        smape = 200 * np.mean(np.abs(y_pred_flat - y_true_flat) / (np.abs(y_pred_flat) + np.abs(y_true_flat) + epsilon))
        
        # R^2 (确定系数)
        if np.var(y_true_flat) == 0:
            r2 = 0  # 避免除以零
        else:
            r2 = 1 - (np.sum((y_true_flat - y_pred_flat) ** 2) / (np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)))
        
        # 返回所有指标
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape,
            'R^2': r2
        }

    def _enhanced_plot_results(self, actuals, predictions, num_samples=192):
        """增强版可视化结果，带有多图表布局和更详细的指标"""
        # 创建子图布局
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # 扁平化数组以便于绘图
        y_true_flat = actuals.flatten()
        y_pred_flat = predictions.flatten()
        
        # 确保长度一致
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # 计算误差
        errors = y_true_flat - y_pred_flat
        
        # 选择要显示的数据点数量
        plot_len = min(len(y_true_flat), num_samples)
        
        # 1. 整体对比图
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(range(plot_len), y_true_flat[:plot_len], 'b-', label='真实值', linewidth=2)
        ax1.plot(range(plot_len), y_pred_flat[:plot_len], 'r--', label='预测值', linewidth=2)
        
        # 添加误差区域
        ax1.fill_between(range(plot_len), 
                        y_true_flat[:plot_len], 
                        y_pred_flat[:plot_len], 
                        color='gray', alpha=0.3, label='误差')
        
        # 添加图例和标签
        ax1.set_title('负荷预测结果对比（实际值 vs 预测值）', fontsize=14)
        ax1.set_xlabel('时间步（15分钟间隔）', fontsize=12)
        ax1.set_ylabel('负荷（标准化）', fontsize=12)
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(True)
        
        # 添加垂直线标记完整天数
        for day in range(1, plot_len // 96 + 1):
            ax1.axvline(x=day*96, color='gray', linestyle='--', alpha=0.5)
            
        # 2. 误差分布图
        ax2 = plt.subplot(gs[1, 0])
        ax2.hist(errors, bins=50, alpha=0.75, color='steelblue')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title('预测误差分布', fontsize=14)
        ax2.set_xlabel('误差值', fontsize=12)
        ax2.set_ylabel('频率', fontsize=12)
        ax2.grid(True)
        
        # 3. 误差时间序列图
        ax3 = plt.subplot(gs[1, 1])
        ax3.plot(range(plot_len), errors[:plot_len], 'g-', linewidth=1)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('预测误差随时间变化', fontsize=14)
        ax3.set_xlabel('时间步（15分钟间隔）', fontsize=12)
        ax3.set_ylabel('误差值', fontsize=12)
        ax3.grid(True)
        
        # 4. 相关性散点图
        ax4 = plt.subplot(gs[2, 0])
        ax4.scatter(y_true_flat, y_pred_flat, alpha=0.5, color='darkblue')
        
        # 添加理想的对角线
        min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
        max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax4.set_title('预测值 vs 实际值相关性', fontsize=14)
        ax4.set_xlabel('实际值', fontsize=12)
        ax4.set_ylabel('预测值', fontsize=12)
        ax4.grid(True)
        
        # 5. 性能指标显示
        ax5 = plt.subplot(gs[2, 1])
        ax5.axis('off')  # 不显示坐标轴
        
        # 计算指标
        metrics = self._calculate_metrics(y_true_flat, y_pred_flat)
        
        # 创建指标文本
        metrics_text = "\n".join([
            f"MSE: {metrics['MSE']:.4f}",
            f"RMSE: {metrics['RMSE']:.4f}",
            f"MAE: {metrics['MAE']:.4f}",
            f"MAPE: {metrics['MAPE']:.2f}%",
            f"SMAPE: {metrics['SMAPE']:.2f}%",
            f"R²: {metrics['R^2']:.4f}"
        ])
        
        # 在图中添加指标文本
        ax5.text(0.5, 0.5, metrics_text, 
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax5.transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig('results/enhanced_forecast_results.png', dpi=300, bbox_inches='tight')
        
        # 为了进一步分析，再创建每日负荷对比图
        if plot_len >= 96:  # 至少有一天的数据
            plt.figure(figsize=(15, 8))
            
            for day in range(min(plot_len // 96, 3)):  # 最多显示3天
                start_idx = day * 96
                end_idx = start_idx + 96
                
                plt.subplot(3, 1, day+1)
                plt.plot(range(96), y_true_flat[start_idx:end_idx], 'b-', label='真实值', linewidth=2)
                plt.plot(range(96), y_pred_flat[start_idx:end_idx], 'r--', label='预测值', linewidth=2)
                
                # 添加时间标记（每4小时）
                hour_ticks = list(range(0, 96, 16))  # 每16个点为4小时
                hour_labels = [f'{h//4}:00' for h in range(0, 24, 4)]
                plt.xticks(hour_ticks, hour_labels)
                
                plt.title(f'第{day+1}天负荷曲线对比', fontsize=14)
                plt.xlabel('时间', fontsize=12)
                plt.ylabel('负荷', fontsize=12)
                plt.legend(loc='best')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('results/daily_forecast_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')


class TorchConvTransformer(AbstractTimeSeriesModel):
    """包装TorchForecaster，使其符合AbstractTimeSeriesModel接口"""
    model_type = 'convtrans'
    
    def __init__(self, input_shape=None, **kwargs):
        super().__init__()
        # 配置默认值
        self.config = {
            'seq_length': 96,
            'pred_length': 1,
            'batch_size': 32,
            'lr': 5e-4,
            'epochs': 20,
            'patience': 10
        }
        
        # 更新配置
        if kwargs:
            self.config.update(kwargs)
            
        # 保存输入形状
        self._input_shape = input_shape
        
        # 如果提供了input_shape，设置seq_length
        if input_shape is not None:
            self.config['seq_length'] = input_shape[0]
            
        # 创建实际的模型
        self.forecaster = TorchForecaster(self.config)
        
    @property
    def input_shape(self):
        """实现抽象方法：返回模型输入形状"""
        return self._input_shape
    
    @property
    def output_shape(self):
        """实现抽象方法：返回模型输出形状"""
        return (self.config['pred_length'],)
        
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None, save_dir=None, callbacks=None):
        """训练模型"""
        if epochs:
            self.config['epochs'] = epochs
        if batch_size:
            self.config['batch_size'] = batch_size
            
        # 准备数据字典
        data_dict = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_val, y_val)  # 暂时用验证集代替测试集
        }
        
        # 训练模型
        self.forecaster.train(data_dict)
        
        if save_dir:
            self.save(save_dir)
            
        return self.forecaster.model
    
    def predict(self, X):
        """进行预测"""
        # 检查模型是否已训练
        if self.forecaster.model is None:
            raise ValueError("模型尚未训练")
            
        # 准备数据
        if isinstance(X, np.ndarray):
            # 转换为数据集
            dataset = TemporalDataset(X, np.zeros((len(X), self.config['pred_length'])))
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # 预测
            self.forecaster.model.eval()
            predictions = []
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.forecaster.device)
                    outputs = self.forecaster.model(batch_x).detach().cpu().numpy()
                    predictions.append(outputs)
                    
            return np.vstack(predictions)
        else:
            raise ValueError("输入必须是numpy数组")
    
    def save(self, save_dir='models/convtrans'):
        """保存模型"""
        import os
        import json
        
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型参数
        model_path = os.path.join(save_dir, f'{self.model_type}_model.pth')
        torch.save(self.forecaster.model.state_dict(), model_path)
        
        # 保存配置
        config_path = os.path.join(save_dir, f'{self.model_type}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
            
        # 保存输入形状
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        with open(input_shape_path, 'w') as f:
            json.dump(list(self._input_shape) if self._input_shape else None, f)
            
        print(f"模型已保存到目录: {save_dir}")
    
    @classmethod
    def load(cls, save_dir='models/convtrans'):
        """加载模型"""
        import os
        import json
        
        # 检查文件是否存在
        model_path = os.path.join(save_dir, f'{cls.model_type}_model.pth')
        config_path = os.path.join(save_dir, f'{cls.model_type}_config.json')
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        
        if not all(os.path.exists(p) for p in [model_path, config_path, input_shape_path]):
            raise FileNotFoundError(f"模型文件缺失，无法加载模型")
            
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 加载输入形状
        with open(input_shape_path, 'r') as f:
            input_shape = json.load(f)
            if input_shape:
                input_shape = tuple(input_shape)
        
        # 创建模型实例
        model_instance = cls(input_shape, **config)
        
        # 初始化模型架构
        input_size = input_shape[1] if input_shape else 1
        model_instance.forecaster.model = TemporalConvLSTM(
            input_size=input_size,
            seq_len=config['seq_length'],
            pred_len=config['pred_length']
        ).to(model_instance.forecaster.device)
        
        # 加载模型参数
        model_instance.forecaster.model.load_state_dict(
            torch.load(model_path, map_location=model_instance.forecaster.device, weights_only=False)
        )
        
        return model_instance

        # 在torch_models.py中添加改进后的模型

class EnhancedTemporalModel(nn.Module):
    """增强版时空融合预测模型，添加注意力机制和残差连接"""
    def __init__(self, input_size, seq_len=96, pred_len=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 多尺度空间特征提取 - 使用不同卷积核大小捕捉不同模式
        self.spatial_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.spatial_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 双向LSTM层，捕捉长期依赖关系
        self.bi_lstm = nn.LSTM(
            input_size=64,  # 两个卷积层的输出通道之和
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # 双向LSTM
        )
        
        # 自注意力机制 - 学习序列内部的关联
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1)
        
        # 输出层 - 预测头
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, pred_len)
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, features)
        batch_size = x.size(0)
        
        # 空间特征提取 - 转换为卷积层所需的格式
        x_conv = x.permute(0, 2, 1)  # (batch, features, seq)
        
        # 多尺度卷积处理
        conv1_out = self.spatial_conv1(x_conv)  # (batch, 32, seq)
        conv2_out = self.spatial_conv2(x_conv)  # (batch, 32, seq)
        
        # 连接不同尺度的特征
        conv_concat = torch.cat([conv1_out, conv2_out], dim=1)  # (batch, 64, seq)
        
        # 转回序列格式
        x_seq = conv_concat.permute(0, 2, 1)  # (batch, seq, 64)
        
        # 双向LSTM处理
        lstm_out, _ = self.bi_lstm(x_seq)  # (batch, seq, 256)
        
        # 自注意力机制
        # 准备注意力层输入格式：(seq_len, batch, features)
        attn_input = lstm_out.permute(1, 0, 2)  
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        
        # 返回原始格式：(batch, seq, features)
        attn_output = attn_output.permute(1, 0, 2)
        
        # 获取序列最后一个时间步的特征作为预测基础
        final_features = attn_output[:, -1, :]
        
        # 通过全连接层生成预测
        output = self.fc_layers(final_features)
        
        return output
    

class AutoWeightedLoss(nn.Module):
    """自动学习权重的多任务损失函数"""
    def __init__(self):
        super(AutoWeightedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # 可学习的权重参数
        self.mse_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.mae_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.mape_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        
    def forward(self, pred, true):
        # 计算各损失
        mse = self.mse_loss(pred, true)
        mae = self.mae_loss(pred, true)
        if pred.shape != true.shape:
            if len(pred.shape) == 2 and len(true.shape) == 1:
                # 方案1：取预测的第一个时间步
                pred = pred[:, 0]
                
        # MAPE损失
        epsilon = 1e-8
        mape = torch.mean(torch.abs((true - pred) / (torch.abs(true) + epsilon)))
        
        # 动态调整权重
        # 权重的平方确保非负性，并除以和以确保归一化
        weights_sum = self.mse_weight**2 + self.mae_weight**2 + self.mape_weight**2 + 1e-8
        
        mse_weight = (self.mse_weight**2) / weights_sum
        mae_weight = (self.mae_weight**2) / weights_sum
        mape_weight = (self.mape_weight**2) / weights_sum
        
        # 组合损失
        loss = mse_weight * mse + mae_weight * mae + mape_weight * mape
        return loss