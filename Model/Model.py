import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    # 此处为attention层，进行特征提取
    def attention_net(self, x, query, mask=None):  # query作为query, x视为key，也视为value，不用mask
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / np.sqrt(d_k)  # 得到query和x的点积，然后除以np.sqrt(d_k)进行缩放

        alpha_n = F.softmax(scores, dim=-1)  # 在行上做softmax，分配权重

        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)  # 经过注意力加权的上下文向量

        return context, alpha_n

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        query = self.dropout(x)
        att, alpha = self.attention_net(x, query)  # 注意力机制特征提取
        att = att.view(-1, h)  # 改形状
        # x = x.view(-1, h)
        x1 = self.fc(att)
        # x1 = self.fc(x) # 临时看看lstm
        x1 = x1.view(s, b, -1)  # 把形状改回来
        return x1
























