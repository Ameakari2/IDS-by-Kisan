"""
读取数据、
缺失值处理、
编码、
归一化（只对训练集fit）、
构建binary和multi数据
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

"""
def load_data():
    data = pd.read_csv(config.DATA_PATH)
    features = pd.read_csv(config.FEATURE_PATH)
    return data, features
"""

def clean_data(data):
    """
    data['service'].replace('-', np.nan, inplace=True)
    data.dropna(inplace=True)
    """
    data = data.copy()
    # 避免 chained assignment
    data['service'] = data['service'].replace('-', np.nan)
    # 删除缺失值
    data = data.dropna()
    return data

def encode_categorical(train_df, test_df):
    categorical_columns = ['proto', 'service', 'state']
    combined = pd.concat([train_df, test_df], axis=0)
    combined = pd.get_dummies(combined, columns=categorical_columns)
    train_df = combined.iloc[:len(train_df)]
    test_df = combined.iloc[len(train_df):]
    return train_df, test_df

"""
def encode_categorical(data):
    # UNSW-NB15 的类别特征
    categorical_columns = ['proto', 'service', 'state'] # 关键的三个特征
    # One-Hot 编码
    data = pd.get_dummies(data, columns=categorical_columns)
    return data
"""

def normalize(train_df, test_df):
    scaler = MinMaxScaler()
    feature_cols = train_df.columns.difference(['label']) # 这里！或许不需要drop
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df


def prepare_binary(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    # 1. 定义想要删除的列，建议加上 'id'以及之前讨论过的低贡献列
    cols_to_drop = ['id', 'attack_cat', 'is_sm_ips_ports', 'ct_ftp_cmd']
    # 2. 统一删除（只删除 DataFrame 中确实存在的列，防止报错），这行代码非常实用！！！
    train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
    test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)

    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 0 else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 0 else 1)
    """
    if 'attack_cat' in train_df.columns:
        train_df = train_df.drop(columns=['attack_cat'])
    if 'attack_cat' in test_df.columns:
        test_df = test_df.drop(columns=['attack_cat'])
    """
    return train_df, test_df


def prepare_multi(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    # 加上 .strip()可以防止 LabelEncoder 把它们识别成两个不同的类
    train_df['attack_cat'] = train_df['attack_cat'].fillna("Normal").strip()
    test_df['attack_cat'] = test_df['attack_cat'].fillna("Normal").strip()
    # 标签编码，在训练集上学习所有的攻击类别
    le = LabelEncoder()
    le.fit(train_df['attack_cat'])

    # 生成多分类的目标标签列（我们依然叫它 'label'，方便后续统一处理）
    # 注意：这里会覆盖掉原本的 0/1 二分类标签？？？
    train_df['label'] = le.transform(train_df['attack_cat'])
    test_df['label'] = le.transform(test_df['attack_cat'])

    cols_to_drop = ['id', 'attack_cat', 'is_sm_ips_ports', 'ct_ftp_cmd']
    train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
    test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)

    return train_df, test_df, le
