import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config

def load_dataset():
    # 1. 读取数据
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # 2. 分类特征编码
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # 3. 标签
    y_train = train_df['new_label'].values
    y_test = test_df['new_label'].values

    # 4. 删除标签列
    X_train = train_df.drop(columns=['new_label'])
    X_test = test_df.drop(columns=['new_label'])

    # 5. 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. ===== 特征分组（关键）=====
    columns = train_df.drop(columns=['new_label']).columns.tolist()

    # 静态特征（统计 + 基础）
    static_features = [
        'spkts','dpkts','sbytes','dbytes','rate',
        'sttl','dttl','sload','dload','sloss','dloss',
        'smean','dmean','ct_srv_src','ct_srv_dst',
        'ct_dst_ltm','ct_src_ltm'
    ]

    # 动态特征（时间 + 行为）
    dynamic_features = [
        'dur','sinpkt','dinpkt','sjit','djit',
        'tcprtt','synack','ackdat',
        'trans_depth','response_body_len',
        'ct_state_ttl','ct_dst_src_ltm'
    ]

    # 剩余未分配特征 → 默认给 static
    remaining = [col for col in columns if col not in static_features + dynamic_features]
    static_features += remaining

    # 获取索引
    static_idx = [columns.index(col) for col in static_features]
    dynamic_idx = [columns.index(col) for col in dynamic_features]

    # 分组数据
    X_train_static = X_train[:, static_idx]
    X_train_dynamic = X_train[:, dynamic_idx]

    X_test_static = X_test[:, static_idx]
    X_test_dynamic = X_test[:, dynamic_idx]

    return (X_train_static, X_train_dynamic,
            X_test_static, X_test_dynamic,
            y_train, y_test)
