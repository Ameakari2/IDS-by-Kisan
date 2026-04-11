import config
import dataset_loader
import train2, train3, train4
import federated
from sklearn.model_selection import train_test_split

def run():
    X_train, X_test, y_train, y_test = dataset_loader.load_dataset()
    # 检查是否开启了联邦学习模式
    use_fl = getattr(config, "USE_FEDERATED", False)
    if use_fl:
        print(f"\nRunning Federated Learning ({config.MODE} mode)...")
        federated.run_federated(X_train.values, X_test.values, 
                                y_train.values, y_test.values, config.MODE)
    else:
        if config.MODE == "binary":
            train2.train_c(X_train.values, X_test.values, y_train.values, y_test.values, "binary")
        else:
            train2.train_c(X_train.values, X_test.values, y_train.values, y_test.values, "multi")

def run2():
    # 1. 加载数据 [cite: 2]
    X_train_full, X_test, y_train_full, y_test = dataset_loader.load_dataset()

    # 检查是否开启了联邦学习模式 
    use_fl = getattr(config, "USE_FEDERATED", False)
    if use_fl:
        print(f"\nRunning Federated Learning ({config.MODE} mode)... ")
        # 注意：如果后续要实现联邦增量学习，federated.run_federated 也需要进行类似修改
        import federated
        federated.run_federated(X_train_full.values, X_test.values, 
                                y_train_full.values, y_test.values, config.MODE)
    else:
        print(f"\nTraining without Federated Learning... ")
        # === 模拟增量学习场景 ===
        # 我们将原始训练集分为两部分：Task 1 (初始知识) 和 Task 2 (新出现的攻击/变体)
        X_task1, X_task2, y_task1, y_task2 = train_test_split(
            X_train_full.values, y_train_full.values, 
            test_size=0.8, random_state=config.RANDOM_STATE
        )

        # --- 阶段 1: 初始训练 (Base Training) ---
        # 此时 is_incremental=False，模型会同时训练静态和动态分支
        print("\n>>> Starting Phase 1: Base Training (Task 1)")
        model_v1, ewc_v1 = train3.train_cnn(
            X_task1, X_test.values, y_task1, y_test.values, 
            task=config.MODE, is_incremental=False
        )

        # --- 阶段 2: 增量学习 (Incremental Fine-tuning) ---
        # 此时 is_incremental=True，模型会冻结静态分支，并启用 EWC 和 LwF 约束
        print("\n>>> Starting Phase 2: Incremental Training (Task 2)")
        model_v2, ewc_v2 = train3.train_cnn(
            X_task2, X_test.values, y_task2, y_test.values, 
            task=config.MODE, 
            is_incremental=True, 
            old_model=model_v1, 
            ewc_instance=ewc_v1
        )
        print("\nIncremental Learning process completed.")

def run3():
    X_train_full, X_test, y_train_full, y_test = dataset_loader.load_dataset()
    # 检查是否开启了联邦学习模式 
    use_fl = getattr(config, "USE_FEDERATED", False)
    if use_fl:
        print(f"\nRunning Federated Learning ({config.MODE} mode)... ")
        # 注意：如果后续要实现联邦增量学习，federated.run_federated 也需要进行类似修改
        import federated
        federated.run_federated(X_train_full.values, X_test.values, 
                                y_train_full.values, y_test.values, config.MODE)
    else:
        print(f"\nTraining without Federated Learning... ")
        # === 模拟增量学习场景 ===
        # 我们将原始训练集分为两部分：Task 1 (初始知识) 和 Task 2 (新出现的攻击/变体)
        X_task1, X_task2, y_task1, y_task2 = train_test_split(
            X_train_full.values, y_train_full.values, 
            test_size=0.8, random_state=config.RANDOM_STATE
        )
        # --- 阶段 1: 初始训练 (Base Training) ---
        # 此时 is_incremental=False，模型会同时训练静态和动态分支
        print("\n>>> Starting Phase 1: Base Training (Task 1)")
        model_v1, ewc_v1 = train4.train_cnn(
            X_task1, X_test.values, y_task1, y_test.values, 
            task=config.MODE, is_incremental=False
        )
        # --- 阶段 2: 增量学习 (Incremental Fine-tuning) ---
        # 此时 is_incremental=True，模型会冻结静态分支，并启用 EWC 和 LwF 约束
        print("\n>>> Starting Phase 2: Incremental Training (Task 2)")
        model_v2, ewc_v2 = train4.train_cnn(
            X_task2, X_test.values, y_task2, y_test.values, 
            task=config.MODE, 
            is_incremental=True, 
            old_model=model_v1, 
            ewc_instance=ewc_v1
        )
        print("\nIncremental Learning process completed.")

if __name__ == "__main__":
    if config.MODE == "binary":
        print("\nRunning Binary Classification...")
    else:
        print("\nRunning Multi Classification...")
    run3()
