import os
import config
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from incremental import BalancedReplayBuffer, EWC

def evaluate_model(model, X, y, batch_size=64, device=None, task_name="",
                   print_report=True, return_results=True, test_size=None,
                   random_state=None, criterion=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """
    # 如果需要从原始数据中划分测试集
    if test_size is not None:
        from sklearn.model_selection import train_test_split
        # 注意：这里 (X, y) 是全部数据，我们只取测试部分
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X, y = X_test, y_test
    """

    # 转换为 Tensor（如果尚未是 Tensor）
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    # 创建 DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)  # 恢复为样本损失和
            sample_count += batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    # avg_loss = total_loss / len(dataset)
    avg_loss = total_loss / sample_count

    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)

    if print_report:
        print(f"\n=== Evaluation Results {task_name} ===")
        print(f"Test Loss: {avg_loss:.6f}")
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Precision (weighted): {precision:.6f}")
        print(f"Recall (weighted): {recall:.6f}")
        print(f"F1-score (weighted): {f1:.6f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        print("\nConfusion Matrix:")
        print(cm)

    if return_results:
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report_dict,
            'y_true': all_labels,
            'y_pred': all_preds
        }


class CNNModel(nn.Module):
    def __init__(self, output_dim, input_channels=1):
        super().__init__()
               
        self.conv = nn.Sequential(
            # nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1), # 留一个输入通道数？
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train_cnn(X_train, X_test, y_train, y_test, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reshape for CNN (1D) 
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    if task == "binary":
        output_dim = 2
    elif task == "multi":
        output_dim = len(torch.unique(y_train))
    else: 
        raise ValueError("mode must be 'binary' or 'multi'")

    model = CNNModel(output_dim=output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epochs = config.CNN_EPOCHS if hasattr(config, "CNN_EPOCHS") else 20
    print("\nTraining CNN...")

    # 1. 检查标签分布
    print("y_train unique:", torch.unique(y_train))
    print("y_train counts:", torch.bincount(y_train))
    # 2. 检查输入数据统计
    print("X_train mean:", X_train.mean().item(), "std:", X_train.std().item())

    buffer = BalancedReplayBuffer(num_classes=output_dim, samples_per_class=200) 
    # balanced version
    ewc = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if any(len(v) > 0 for v in buffer.buffer.values()):
                old_x, old_y = buffer.sample(16)
                old_x = old_x.to(device)
                old_y = old_y.to(device)

                X_batch = torch.cat([X_batch, old_x], dim=0)
                y_batch = torch.cat([y_batch, old_y], dim=0)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            # EWC 正则
            if ewc is not None:
                loss += 0.4 * ewc.penalty(model)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            for i in range(len(X_batch)):    # 只存新数据
                if i < 64:   # 原始 batch size, 设备移动问题？
                    buffer.add(X_batch[i].detach(), y_batch[i].detach()) 
            
        """
        # 确保 buffer 中数据在CPU, 避免 GPU 显存占用。建议修改
        batch_size = y_batch.shape[0]
        for i in range(batch_size):
            buffer.add(
                X_batch[i].detach().cpu(),
                y_batch[i].detach().cpu()
            )
        """

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    """
    # evaluation
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)

    print(f"\nCNN Accuracy ({task}):", acc)
    print(classification_report(y_true, y_pred, zero_division=0))
    """
    print("\nEvaluating on test set...")
    evaluate_model(
        model,
        X_test.cpu().numpy(),   # 传入 NumPy 数组
        y_test.cpu().numpy(),
        batch_size=64,
        device=device,
        task_name=task,
        print_report=True,
        return_results=False    # 只需打印，不返回结果
    )
    """
    # Save model
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_PATH, f"cnn_{task}_v2.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # 说明：PyTorch保存的为.pt文件，传统机器学习（Sklearn）保存的.pkl文件
    """
    print("Computing Fisher Information for EWC...")
    ewc = EWC(model, train_loader, device)
    
def train_c(X_train, X_test, y_train, y_test, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    if task == "binary":
        output_dim = 2
    elif task == "multi":
        output_dim = len(torch.unique(y_train))
    else: 
        raise ValueError("mode must be 'binary' or 'multi'")

    model = CNNModel(output_dim=output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epochs = config.CNN_EPOCHS if hasattr(config, "CNN_EPOCHS") else 20
    print("\nTraining CNN...")

    # 1. 检查标签分布
    print("y_train unique:", torch.unique(y_train))
    print("y_train counts:", torch.bincount(y_train))
    # 2. 检查输入数据统计
    print("X_train mean:", X_train.mean().item(), "std:", X_train.std().item())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
 
    print("\nEvaluating on test set...")
    evaluate_model(
        model,
        X_test.cpu().numpy(),   # 传入 NumPy 数组
        y_test.cpu().numpy(),
        batch_size=64,
        device=device,
        task_name=task,
        print_report=True,
        return_results=False    # 只需打印，不返回结果
    )

