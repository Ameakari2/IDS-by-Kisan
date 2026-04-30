import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix)
import numpy as np
import math

import config

def evaluate_baseline(model, X, y, device):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)

            outputs = model(x)

            _, preds = torch.max(outputs, 1)
            # probs = torch.softmax(outputs, dim=1)[:, 1]
            # preds = (probs > 0.5).long() # 阈值

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== Baseline Evaluation ===")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("Recall:", recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("F1:", f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def evaluate_baseline_universal(model, X, y, device):
    """
    通用评价函数：不再进行 reshape，只负责推理和指标计算
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            outputs = model(x)

            # _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long() # 阈值

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\n=== {type(model).__name__} Evaluation ===")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("Recall (Macro):", recall_score(all_labels, all_preds, average='macro', zero_division=0))
    print("F1:", f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM 层
        # batch_first=True 表示输入形状为 (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        # 初始化隐藏状态和细胞状态 (可选，不传则默认为全0)
        # x shape: (batch, seq_len, input_dim)
        
        # out shape: (batch, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)
        # 我们通常取最后一个时间步的输出作为特征
        # 或者也可以取 h_n[-1]
        last_time_step = out[:, -1, :] 
        
        logits = self.fc(last_time_step)
        return logits

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x) 
        # 取最后一个时间步的输出
        return self.fc(out[:, -1, :])

class ECA1D(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA1D, self).__init__()
        # 1. 根据通道数自适应计算卷积核大小 (必须是奇数)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        # 2. 全局平均池化，将特征压缩到 [Batch, Channels, 1]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 3. 1D卷积，用于跨通道信息交互。注意这里 in_channels=1, out_channels=1
        # 这里的 padding 保证了卷积前后通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # 4. Sigmoid 激活函数，生成权重
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x 的输入形状应为: [Batch, Channels, Length]
        # 特征压缩 -> [Batch, Channels, 1]
        y = self.avg_pool(x)
        # 为了让 1D 卷积在通道维度上滑动，需要转置
        # 转置后形状 -> [Batch, 1, Channels]
        y = y.transpose(-1, -2)
        # 经过 1D 卷积交互通道信息
        y = self.conv(y)
        # 转置回原来的形状 -> [Batch, Channels, 1]
        y = y.transpose(-1, -2)
        # 生成通道注意力权重
        y = self.sigmoid(y)
        # 将权重与原特征相乘
        # expand_as 确保了形状匹配，即使 length 维度不为 1 也能正确广播
        return x * y.expand_as(x)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # self.eca = ECA1D(128)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        x = self.conv(x)
        # x = self.eca(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

def train_cnn(X_train, X_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    # ✅ 拼接特征（关键区别）
    X_train = np.concatenate([Xs_train, Xd_train], axis=1)
    X_test = np.concatenate([Xs_test, Xd_test], axis=1)
    """
    # reshape
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    output_dim = len(np.unique(y_train.numpy()))
    input_dim = X_train.shape[-1]

    model = SimpleCNN(input_dim, output_dim).to(device)

    # 👉 baseline建议：先不用加权
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("\nTraining Baseline CNN...")

    for epoch in range(config.CNN_EPOCHS):
        model.train()
        total_loss = 0

        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Baseline] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # ===== 测试 =====
    print("\nEvaluating Baseline...")
    evaluate_baseline(model, X_test, y_test, device)

def train_sequence(X_train, X_test, y_train, y_test):
    """
    model_type: 'LSTM' 或 'RNN'
    """
    model_type = config.MODEL_TYPE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ===== 1. 维度适配 (关键修改) =====
    # 假设特征平铺，我们将其 reshape 为 (Batch, Seq_Len, Input_Dim)
    seq_len = 10 
    input_dim = X_train.shape[-1] // seq_len
    
    # Reshape 训练集和测试集
    X_train_seq = X_train.reshape(-1, seq_len, input_dim)
    X_test_seq = X_test.reshape(-1, seq_len, input_dim)

    # 转换为 Tensor
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    # 测试集也预先转好，方便最后 evaluate 调用
    X_test_tensor = X_test_seq 
    y_test_tensor = y_test

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # ===== 2. 初始化模型 =====
    output_dim = len(np.unique(y_train))
    
    if model_type.upper() == 'LSTM':
        model = SimpleLSTM(input_dim, 128, output_dim).to(device)
    else:
        model = SimpleRNN(input_dim, 128, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\n--- Training {model_type} Baseline ---")

    # ===== 3. 训练循环 =====
    for epoch in range(config.CNN_EPOCHS):
        model.train()
        total_loss = 0
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪 (针对 RNN 系模型非常重要)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        print(f"[{model_type}] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # ===== 4. 自动评价 (Evaluation) =====
    print(f"\nEvaluating {model_type}...")
    # 这里直接调用你之前的评估逻辑，但传入的是已经 reshape 好的测试数据
    evaluate_baseline_universal(model, X_test_tensor, y_test_tensor, device)


