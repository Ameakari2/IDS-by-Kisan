import pandas as pd

def review_dataset(file_path, name="数据集"):
    print(f"{'='*20} {name} 检查报告 {'='*20}")
    
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 1. 检查数据形状
        print(f"数据总行数: {df.shape[0]}")
        print(f"数据总列数: {df.shape[1]}")
        
        # 2. 检查列名
        print("\n当前所有列名:")
        print(list(df.columns))
        
        # 3. 检查标签分布 (new_label)
        if 'new_label' in df.columns:
            print("\n标签分布情况 (new_label):")
            # 统计数量和占比
            counts = df['new_label'].value_counts().sort_index()
            percentages = df['new_label'].value_counts(normalize=True).sort_index() * 100
            
            # 建立标签名对照表（方便阅读）
            mapping = {
                0: 'Normal', 1: 'Generic', 2: 'Exploits', 3: 'Fuzzers', 
                4: 'DoS', 5: 'Reconnaissance', 6: 'Analysis', 
                7: 'Backdoor', 8: 'Shellcode', 9: 'Worms'
            }
            
            dist_df = pd.DataFrame({
                '攻击类型': [mapping.get(i, f"未知{i}") for i in counts.index],
                '样本数量': counts.values,
                '占比 (%)': percentages.values.round(2)
            })
            print(dist_df.to_string(index=False))
        else:
            print("\n警告: 未找到 'new_label' 列，请检查预处理是否成功。")
            
        # 4. 检查是否有缺失值
        nan_count = df.isnull().sum().sum()
        print(f"\n全表缺失值总量: {nan_count}")
        
        # 5. 检查是否存在重复行
        dup_count = df.duplicated().sum()
        print(f"表内重复行数量: {dup_count}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}，请确保预处理脚本已运行并生成了该文件。")
    print("="*55 + "\n")

if __name__ == "__main__":
    # 检查你生成的两个新文件
    review_dataset('train_processed.csv', name="训练集")
    review_dataset('test_processed.csv', name="测试集")

