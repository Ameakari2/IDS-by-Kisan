import os
import config
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from incremental import BalancedReplayBuffer, EWC

class CNNModel(nn.Module):
    # def __init__(self):
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
        """
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 输出: 32*3*3

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 输出: 64*1*1

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1) # 输出: 128*1*1
        )"""
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
    """
    # reshape for 2D CNN
    # X_train: (num_samples, 49) -> (num_samples, 1, 7, 7)
    X_train = X_train.reshape(-1, 1, 7, 7)
    X_test = X_test.reshape(-1, 1, 7, 7)
    """

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

    # model = CNNModel().to(device)
    model = CNNModel(output_dim=output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epochs = config.CNN_EPOCHS if hasattr(config, "CNN_EPOCHS") else 20
    print("\nTraining CNN...")
    
    """
    buffer = ReplayBuffer(capacity=2000)
    """
    buffer = BalancedReplayBuffer(num_classes=output_dim, samples_per_class=200) 
    # balanced version
    ewc = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            """
            # 从 replay buffer 采样旧数据
            if len(buffer.buffer) > 0:
                old_x, old_y = buffer.sample(32) # 这里可以从 32 改成 16
                old_x = old_x.to(device)
                old_y = old_y.to(device)
                X_batch = torch.cat([X_batch, old_x])
                y_batch = torch.cat([y_batch, old_y])
            # replay buffer 和 optimizer 在 batch 内, buffer 要 detach

            """
            # 更安全版本：（要判断 任意类别有数据）
            if any(len(v) > 0 for v in buffer.buffer.values()):
                old_x, old_y = buffer.sample(16)
                old_x = old_x.to(device)
                old_y = old_y.to(device)
                """
                # 保存原始新数据用于添加 buffer
                original_X = X_batch  # 在拼接前保存
                original_y = y_batch
                """
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
            # deepsuck 卡住的版本
            """
            total_loss = 0.0
            total_samples = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)   # 此时 loss 是批次总损失（标量）
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_samples += inputs.size(0)
            """

        """
        # 存入 replay buffer
        for i in range(len(X_batch)):
            buffer.add(X_batch[i], y_batch[i])    # X_batch 会包含 旧数据+新数据
        """
        for i in range(len(X_batch)):    # 只存新数据
            if i < 64:   # 原始 batch size, 设备移动问题？
                buffer.add(X_batch[i].detach(), y_batch[i].detach()) 
        """
        # 添加原始新数据到 buffer（deepsuck修改意见, original）
        for i in range(len(original_X)):
            buffer.add(original_X[i].detach(), original_y[i].detach())
        
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
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        # print(outputs.shape)  # 检查 outputs 的形状
        # print(len(train_loader))
    
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
    # Save model
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_PATH, f"cnn_ids_{task}_v2.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # 说明：PyTorch保存的为.pt文件，传统机器学习（Sklearn）保存的.pkl文件

    print("Computing Fisher Information for EWC...")
    ewc = EWC(model, train_loader, device)

