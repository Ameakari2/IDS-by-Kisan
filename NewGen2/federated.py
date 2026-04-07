"""
版本说明：3-19v1，简单 CNN 结构
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config
from train2 import CNNModel, evaluate_model

def get_client_dataloaders(X_train, y_train, num_clients, batch_size):
    """
    将集中式数据均匀划分给多个客户端 (IID 划分)
    """
    # 模拟数据被打乱并切分
    indices = np.random.permutation(len(X_train))
    client_data_loaders = []
    
    # 计算每个客户端的数据量
    split_size = len(X_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_clients - 1 else len(X_train)
        
        client_indices = indices[start_idx:end_idx]
        client_X = X_train[client_indices]
        client_y = y_train[client_indices]
        
        dataset = TensorDataset(client_X, client_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_data_loaders.append(loader)
        
    return client_data_loaders

def client_update(global_model, client_loader, epochs, lr, device):
    """
    客户端本地训练
    """
    # 客户端复制一份全局模型进行本地更新
    model = copy.deepcopy(global_model)
    model.train()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for X_batch, y_batch in client_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def fedavg(weights_list, client_data_sizes):
    """
    服务器端进行 FedAvg 聚合计算
    """
    total_samples = sum(client_data_sizes)
    avg_weights = copy.deepcopy(weights_list[0])
    
    for key in avg_weights.keys():
        # 按数据量加权
        avg_weights[key] = avg_weights[key] * (client_data_sizes[0] / total_samples)
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key] * (client_data_sizes[i] / total_samples)
            
    return avg_weights

def run_federated(X_train, X_test, y_train, y_test, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Federated Learning on {device}...")

    # 1. 数据预处理 (与 train2.py 保持一致)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 确定输出维度
    if task == "binary":
        output_dim = 2
    elif task == "multi":
        output_dim = len(torch.unique(y_train))
    else: 
        raise ValueError("mode must be 'binary' or 'multi'")

    # 2. 获取联邦参数
    num_clients = getattr(config, "FL_NUM_CLIENTS", 5)
    fl_rounds = getattr(config, "FL_ROUNDS", 20)
    local_epochs = getattr(config, "FL_LOCAL_EPOCHS", 3)
    
    # 3. 划分客户端数据
    client_loaders = get_client_dataloaders(X_train, y_train, num_clients, config.BATCH_SIZE)
    client_data_sizes = [len(loader.dataset) for loader in client_loaders]

    # 4. 初始化全局模型
    global_model = CNNModel(output_dim=output_dim).to(device)
    
    # 5. 联邦学习轮次循环
    for round_idx in range(fl_rounds):
        print(f"\n--- Global Round {round_idx + 1}/{fl_rounds} ---")
        
        local_weights = []
        # 遍历所有客户端进行本地训练
        for client_idx in range(num_clients):
            print(f"  Training Client {client_idx + 1}...")
            client_w = client_update(
                global_model=global_model,
                client_loader=client_loaders[client_idx],
                epochs=local_epochs,
                lr=config.LEARNING_RATE,
                device=device
            )
            local_weights.append(client_w)
            
        # 服务器聚合权重
        print("  Aggregating weights (FedAvg)...")
        global_weights = fedavg(local_weights, client_data_sizes)
        global_model.load_state_dict(global_weights)
        
        # 可选：每轮结束后在全局测试集上验证性能
        evaluate_model(
            global_model, X_test, y_test, 
            batch_size=config.BATCH_SIZE, device=device, 
            task_name=f"Round {round_idx+1}", print_report=False
        )

    # 6. 最终模型评估
    print("\n=== Final Global Model Evaluation ===")
    evaluate_model(
        global_model, X_test, y_test, 
        batch_size=config.BATCH_SIZE, device=device, 
        task_name=task, print_report=True, return_results=False
    )
    
    # 保存全局模型
    import os
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_PATH, f"fl_cnn_{task}.pt")
    torch.save(global_model.state_dict(), model_path)
    print(f"Global FL Model saved to {model_path}")
    