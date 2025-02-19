# 全局变量存储文件

# 时间间隔，单位分钟
TIME_INTERVAL = 15

# 存储电话号码
PHONE_NUM = '15528932507'

# 存储密码
PASSWORD = '123456'

# 判断开始结束的点数
NUM_POINTS = 2
CHANGE_LOWER = 400
CHANGE_UPPER = 500

# 人体感应器档案
Inductions = {
    '展示区人体感应1':{
        'beeID': '86200001187',
        'mac': 'Irs-M1-84f703112028'
    },
    '展示区人体感应2':{
        'beeID': '86200001187',
        'mac': 'Irs-M1-84f703122288'
    },
    '小会议室人体感应':{
        'beeID': '86200001187',
        'mac': 'Irs-M1-84f7031218b4'
    },
    '办公室B人体感应':{
        'beeID': '86200001183',
        'mac': 'Irs-M1-84f70310d0f4'
    },
    '办公室C人体感应':{
        'beeID': '86200001183',
        'mac': 'Irs-M1-7cdfa1b85e50'
    },
    '办公室D人体感应':{
        'beeID': '86200001183',
        'mac': 'Irs-M1-7cdfa1b84cd8'
    },
    '大会议室人体感应':{
        'beeID': '86200001183',
        'mac': 'Irs-M1-84f703101f5c'
    },
    '学生办公区人体感应1':{
        'beeID': '86200001289',
        'mac': 'Irs-M1-7cdfa1b85e28'
    },
    '学生办公区人体感应2':{
        'beeID': '86200001289',
        'mac': 'Irs-M1-7cdfa1b84cb4'
    },
}

# 设备档案
devices_lib = {
    '空调':{
        'beeID': '86200001289',
        'mac': 'Asm-M1-58cf790d4cc0'
    },
    '打印机':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-84f703126e44'
    },
    '子路由器':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-7cdfa1b608c8'
    },
    '冰箱':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-48ca43e5f64c'
    },
    '网络设备':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-7cdfa1b89d58'
    },
    '咖啡机':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-9c9e6e186d3c'
    },
    '烧水壶':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-9c9e6e186f14'
    },
    '微波炉':{
        'beeID': '86200001187',
        'mac': 'Sck-M1-9c9e6e159b7c'
    },
}

# 工位档案
workstation_lib = {
    1:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e159fd0', 'Sck-M1-9c9e6e1518e4', 'Sck-M1-48ca43e5f388']
    },
    2:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e15a098', 'Sck-M1-48ca43e5f598', 'Sck-M1-9c9e6e15997c']
    },
    3:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e177494', 'Sck-M1-9c9e6e159950', 'Sck-M1-9c9e6e156e58']
    },
    4:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-48ca43e608c4', 'Sck-M1-9c9e6e156ec8', 'Sck-M1-48ca43e5c918']
    },
    5:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e18716c', 'Sck-M1-48ca43e5f38c', 'Sck-M1-9c9e6e1857c8']
    },
    6:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-48ca43e5f58c', 'Sck-M1-9c9e6e156dd0', 'Sck-M1-9c9e6e158e04']
    },
    7:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-48ca43e59004', 'Sck-M1-9c9e6e156e40', 'Sck-M1-9c9e6e15a0bc']
    },
    8:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e156ea0', 'Sck-M1-48ca43e6088c', 'Sck-M1-48ca43e5c43c']
    },
    9:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e156eec', 'Sck-M1-9c9e6e185870', 'Sck-M1-9c9e6e1519f8']
    },
    10:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e18706c', 'Sck-M1-9c9e6e156f08', 'Sck-M1-48ca43e5f5b4']
    },
    11:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e158f3c', 'Sck-M1-9c9e6e1690b0', 'Sck-M1-9c9e6e186fcc']
    },
    12:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-48ca43e5f3b0', 'Sck-M1-48ca43e5c90c', 'Sck-M1-48ca43e60868']
    },
    13:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-48ca43e5f638', 'Sck-M1-9c9e6e15a280', 'Sck-M1-9c9e6e185910']
    },
    14:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e15a014', 'Sck-M1-9c9e6e1870c8', 'Sck-M1-48ca43e5f630']
    },
    15:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e159db4', 'Sck-M1-9c9e6e156f20', 'Sck-M1-48ca43e5cbd0']
    },
    16:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e164d54', 'Sck-M1-48ca43e5f5c0', 'Sck-M1-9c9e6e158718']
    },
    17:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e187110', 'Sck-M1-9c9e6e151c84', 'Sck-M1-9c9e6e152c1c']
    },
    18:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e159ae8', 'Sck-M1-9c9e6e156f88', 'Sck-M1-9c9e6e158de4']
    },
    19:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-48ca43e5c920', 'Sck-M1-9c9e6e16701c', 'Sck-M1-48ca43e5ca64']
    },
    20:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e159af4', 'Sck-M1-9c9e6e158e10', 'Sck-M1-9c9e6e159dd4']
    },
    21:{
        'beeID': '86200001289',
        'mac': ['Sck-M1-9c9e6e18711c', 'Sck-M1-9c9e6e159abc', 'Sck-M1-9c9e6e16937c']
    },
    22:{
        'beeID': '86200001187',
        'mac': ['Sck-M1-9c9e6e159b2c', 'Sck-M1-9c9e6e187178']
    },
    23:{
        'beeID': '86200001187',
        'mac': ['Sck-M1-9c9e6e159ff8', 'Sck-M1-9c9e6e17bf74']
    },
    24:{
        'beeID': '86200001187',
        'mac': ['Sck-M1-48ca43e5c8ac', 'Sck-M1-48ca43e5c4d8']
    },
}









