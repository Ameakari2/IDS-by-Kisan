import pandas as pd
import numpy as np
import config

# 1. 加载数据集
train_df = pd.read_csv(config.TRAIN_DATA)
test_df = pd.read_csv(config.TEST_DATA)

def clean_and_format(df, name="数据集"):
    print(f"--- 正在处理 {name} ---")
    
    # --- 步骤 1: 删除含空格的行 ---
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna()

    # --- 步骤 2: 删除自身重复的数据 (新增) ---
    # keep='first' 表示保留重复项中的第一条，删除其余的
    before_dup = len(df)
    df = df.drop_duplicates()
    print(f"删除了 {before_dup - len(df)} 条自身重复的行")

    # --- 步骤 3: 处理 service 列 ---
    df['service'] = df['service'].replace('-', 'unknown')

    # --- 步骤 4: 动态检查并删除无关列 ---
    cols_to_remove = [
        'id', 'label', 'stime', 'ltime', 
        'srcip', 'dstip', 'is_ftp_login', 'ct_ftp_cmd'
    ]
    existing_drop_cols = [c for c in cols_to_remove if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    
    # 4. 【核心修正】在删除 ID 类列后，立即执行全表去重
    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"在特征维度上删除了 {before_dup - len(df)} 条重复行")
    
    # --- 步骤 5: 标签重映射 (0-9) ---
    label_mapping = {
        'Normal': 0, 'Generic': 1, 'Exploits': 2, 'Fuzzers': 3, 
        'DoS': 4, 'Reconnaissance': 5, 'Analysis': 6, 
        'Backdoor': 7, 'Shellcode': 8, 'Worms': 9
    }
    
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].str.strip()
        df['new_label'] = df['attack_cat'].map(label_mapping)
        df = df.dropna(subset=['new_label'])
        df = df.drop(columns=['attack_cat'])
    
    return df

# 执行初步清洗（包含自身去重）
train_processed = clean_and_format(train_df, "训练集")
test_processed = clean_and_format(test_df, "测试集")

# --- 步骤 6: 跨数据集去重 (防止测试集泄露) ---
# 此时 train 和 test 内部已经没有重复了，现在确保 test 里没有 train 的数据
original_test_size = len(test_processed)
test_processed = pd.merge(test_processed, train_processed, how='left', indicator=True) \
                   .query("_merge == 'left_only'") \
                   .drop(columns=['_merge'])
print(f"\n跨数据集去重：从测试集中删除了 {original_test_size - len(test_processed)} 条在训练集中出现过的样本")

# --- 步骤 7: 采样上限处理 (15000条) ---
def sample_limit(df):
    # 使用 groupby 确保每个类别最多 15000，不足则全部保留
    return df.groupby('new_label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 15000), random_state=42)
    ).reset_index(drop=True)

def sample_limit2(df):
    return df.groupby('new_label', group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), 20000 if x.name == 0 else 5000),
            random_state=42)
    ).reset_index(drop=True)

final_train = sample_limit2(train_processed)
final_test = sample_limit2(test_processed)

# 最终导出
final_train.to_csv('train_binary.csv', index=False)
final_test.to_csv('test_binary.csv', index=False)
print("\n所有预处理操作已完成，文件已保存。")

