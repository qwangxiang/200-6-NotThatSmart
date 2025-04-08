# from .keras_models import KerasGRU, KerasLSTM
from .torch_models import TorchForecaster
from iTransformer.data.dataset_builder import DatasetBuilder
from iTransformer.data.data_loader import DataLoader

__all__ = ['KerasGRU', 'KerasLSTM', 'TorchForecaster', 'DatasetBuilder']

# 配置默认参数
DEFAULT_SEQ_LENGTH = 96
DEFAULT_PRED_LENGTH = 4
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3

# 全局数据加载器
def load_data():
    return DataLoader('附件1-数据.xlsx')

# 全局数据预处理器
def prepare_dataset(seq_length=DEFAULT_SEQ_LENGTH, pred_length=DEFAULT_PRED_LENGTH):
    data_loader = load_data()
    dataset_builder = DatasetBuilder(data_loader, seq_length, pred_length)
    return dataset_builder.build()