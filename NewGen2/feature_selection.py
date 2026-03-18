import numpy as np
import config

"""
def select_features(df):
    corr = df.corr() # 在整个数据集上计算相关系数
    corr_y = abs(corr['label']) # 使用整个数据集的标签列
    selected = corr_y[corr_y > config.CORRELATION_THRESHOLD].index
    return df[selected]
"""

def select_features(X, y, threshold=config.CORRELATION_THRESHOLD):
    """
    X: DataFrame, 特征数据
    y: Series, 标签
    threshold: 相关系数阈值
    return: 选中的特征列名列表
    """
    df = X.copy()
    df['label'] = y
    corr = df.corr()
    corr_y = abs(corr['label']).drop('label')  # 排除自身
    selected_cols = corr_y[corr_y > threshold].index.tolist()
    return selected_cols

# ChatGPT给的高级版本
"""
def select_features(X, y, threshold=0.1):
    df = X.copy()
    df["label"] = y
    corr = df.corr()
    corr_y = abs(corr["label"]).drop("label")
    selected = corr_y[corr_y > threshold].index.tolist()
    print("Selected features:", len(selected))
    return selected
"""
