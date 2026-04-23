import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config
import numpy as np

def merge_rare_classes(y):
    y = y.copy()
    rare_classes = [6, 7, 8, 9] # 设定要合并的类别
    target_class = 6  # 合并到6
    for rc in rare_classes:
        y[y == rc] = target_class
    return y

def to_binary_labels(y):
    y = y.copy()
    y[y != 0] = 1
    return y

def remove_classes(X, y, remove_list):
    mask = ~np.isin(y, remove_list)   # 保留不在 remove_list 的
    X = X[mask]
    y = y[mask]
    return X, y

def semantic_grouping(columns):
    group1 = []  # 流量统计
    group2 = []  # 时间行为
    for col in columns:
        if any(k in col for k in ['bytes', 'pkts', 'rate']):
            group1.append(col)
        elif any(k in col for k in ['dur', 'jit', 'pkt', 'tt']):
            group2.append(col)
        else:
            group1.append(col)
    return group1, group2

def reorder_features(df):
    feature_cols = [c for c in df.columns if c != 'new_label']

    flow = [c for c in feature_cols if any(k in c for k in ['bytes', 'pkts', 'rate'])]
    time = [c for c in feature_cols if any(k in c for k in ['dur', 'jit', 'pkt'])]
    tcp  = [c for c in feature_cols if any(k in c for k in ['tcp', 'ack', 'syn'])]
    stat = [c for c in feature_cols if 'ct_' in c]
    ordered = []
    for group in [flow, time, tcp, stat]:
        for c in group:
            if c in feature_cols and c not in ordered:
                ordered.append(c)
    # 剩余特征补上
    remaining = [c for c in feature_cols if c not in ordered]
    ordered += remaining

    return df[ordered + ['new_label']]

def load_dataset_baseline():
    # 1. 读取数据
    if config.MODE == "multi":
        train_df = pd.read_csv(config.TRAIN_PATH)
        test_df = pd.read_csv(config.TEST_PATH)
    if config.MODE == "binary":
        train_df = pd.read_csv(config.TRAIN_BINARY)
        test_df = pd.read_csv(config.TEST_BINARY)
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
    if config.MODE == "multi":
        y_train = merge_rare_classes(y_train)
        y_test = merge_rare_classes(y_test)
    if config.MODE == "binary":
        y_train = to_binary_labels(y_train)
        y_test = to_binary_labels(y_test)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 4. 删除标签列
    X_train = train_df.drop(columns=['new_label'])
    X_test = test_df.drop(columns=['new_label'])

    # 5. 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_dataset():
    # 1. 读取数据
    if config.MODE == "multi":
        train_df = pd.read_csv(config.TRAIN_PATH)
        test_df = pd.read_csv(config.TEST_PATH)
    if config.MODE == "binary":
        train_df = pd.read_csv(config.TRAIN_BINARY)
        test_df = pd.read_csv(config.TEST_BINARY)
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
    if config.MODE == "multi":
        y_train = merge_rare_classes(y_train)
        y_test = merge_rare_classes(y_test)
    if config.MODE == "binary":
        y_train = to_binary_labels(y_train)
        y_test = to_binary_labels(y_test)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 4. 删除标签列
    X_train = train_df.drop(columns=['new_label'])
    X_test = test_df.drop(columns=['new_label'])

    # 5. 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. ===== 特征分组（关键）=====
    columns = train_df.drop(columns=['new_label']).columns.tolist()
    """
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
    """
    # 语义分组======
    static_features, dynamic_features = semantic_grouping(columns)

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

def load_dataset_remove():
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # 标签
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)

        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    
    # 3. 提取 X 和 y
    y_train = train_df['new_label'].values
    y_test = test_df['new_label'].values

    X_train = train_df.drop(columns=['new_label']).values
    X_test = test_df.drop(columns=['new_label']).values

    # 4. ⭐ 删除类别
    X_train, y_train = remove_classes(X_train, y_train, [6,7,8,9])
    X_test, y_test = remove_classes(X_test, y_test, [6,7,8,9])

    # 5. ⭐ 标签重编码
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 6. ⭐ 标准化（现在才可以）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    columns = train_df.drop(columns=['new_label']).columns.tolist()

    static_features, dynamic_features = semantic_grouping(columns)

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

def load_dataset_multiscale():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import config

    # 1. 读取数据
    train_df = pd.read_csv(config.TRAIN_BINARY)
    test_df = pd.read_csv(config.TEST_BINARY)

    # 2. 分类特征编码（必须在最前）
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)

        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    
    train_df = reorder_features(train_df)
    test_df = reorder_features(test_df)
    # 3. 标签
    y_train = train_df['new_label'].values
    y_test = test_df['new_label'].values

    # ⭐ 二分类支持
    if config.MODE == "binary":
        y_train = (y_train != 0).astype(int)
        y_test = (y_test != 0).astype(int)

    # ⭐ 多分类可选（保留你原逻辑）
    if config.MODE == "multi":
        y_train = merge_rare_classes(y_train)
        y_test = merge_rare_classes(y_test)

    # ⭐ 重新编码标签
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 4. 特征
    X_train = train_df.drop(columns=['new_label']).values
    X_test = test_df.drop(columns=['new_label']).values

    # ⭐ 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# 仅用于单独测试 0 和 3 
def balance_sample(df, target_col, classes, n_limit):
    """
     辅助函数：对指定 DataFrame 进行按类别采样
      """
    sampled_list = []
    for cls in classes:
        cls_df = df[df[target_col] == cls]
          # 如果现有数量超过限制，则采样；否则保留全部
        if len(cls_df) > n_limit:
            cls_df = cls_df.sample(n=n_limit, random_state=42)
        sampled_list.append(cls_df)
    return pd.concat(sampled_list).sample(frac=1, random_state=42) # 合并并打乱

def load_dataset_filtered():
    # 1. 读取原始数据
    train_df = pd.read_csv(config.TRAIN_BINARY)
    test_df = pd.read_csv(config.TEST_BINARY)

    # 2. 类别过滤与采样 (每个类别最高 5000 条)
    target_classes = [0, 3]
    limit_per_class = 5000
    
    train_df = balance_sample(train_df, 'new_label', target_classes, limit_per_class)
    test_df = balance_sample(test_df, 'new_label', target_classes, limit_per_class)

    # 3. 分类特征编码（基于采样后的数据，确保效率）
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    
    # 假设 reorder_features 是你已定义的逻辑
    train_df = reorder_features(train_df)
    test_df = reorder_features(test_df)

    # 4. 提取特征与标签
    X_train = train_df.drop(columns=['new_label']).values
    y_train = train_df['new_label'].values
    X_test = test_df.drop(columns=['new_label']).values
    y_test = test_df['new_label'].values

    # 5. 标签重编码 (将 [0, 3] 映射为 [0, 1])
    le_label = LabelEncoder()
    y_train = le_label.fit_transform(y_train)
    y_test = le_label.transform(y_test)
    # 7. 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

