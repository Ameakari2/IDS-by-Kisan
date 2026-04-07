import pandas as pd
import config
import feature_selection
import preprocess

def load_dataset():
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # 清洗数据
    train_df = preprocess.clean_data(train_df)
    test_df = preprocess.clean_data(test_df)

    # one-hot 编码类别特征
    train_df, test_df = preprocess.encode_categorical(train_df, test_df)

    if config.MODE == "binary":
        train_df, test_df = preprocess.prepare_binary(train_df, test_df)
    elif config.MODE == "multi":
        train_df, test_df, _ = preprocess.prepare_multi(train_df, test_df)
    else:
        raise ValueError("mode must be 'binary' or 'multi'")
    
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()
    test_rows = set(tuple(row) for row in test_df.to_numpy())
    before_len = len(train_df)
    train_df = train_df[train_df.apply(lambda row: tuple(row) not in test_rows, axis=1)]
    after_len = len(train_df)
    print(f"\n从训练集删除重复样本数量：{before_len - after_len}")
    print(f"最终训练集: {train_df.shape} | 测试集: {test_df.shape}")
    
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    # 归一化，与 feature selection 交换位置更好
    X_train, X_test = preprocess.normalize(X_train, X_test)
    """
    # Feature selection (只在训练集上计算)
    selected_features = feature_selection.select_features(X_train, y_train)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    """
    return X_train, X_test, y_train, y_test

