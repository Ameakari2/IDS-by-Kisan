import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import math
import numpy as np
from sklearn.metrics import f1_score
import config

def evaluate_threshold(model, X, y, device):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    y = np.array(y)
    dataset = TensorDataset(X, torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_probs = []
    all_labels = []
    # =========================
    # 1️⃣ 先收集预测概率
    # =========================
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    # =========================
    # 2️⃣ 找最优阈值（关键修复）
    # =========================

    best_th = 0.5
    best_f1 = 0
    for th in np.linspace(0.1, 0.9, 81):
        preds = (all_probs > th).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    print(f"\nBest Threshold: {best_th:.3f}, Best F1: {best_f1:.4f}")
    # =========================
    # 3️⃣ 用最优阈值重新预测
    # =========================
    all_preds = (all_probs > best_th).astype(int)
    # =========================
    # 4️⃣ 评估
    # =========================
    print("\n=== Baseline Evaluation ===")

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("Recall:", recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("F1:", f1_score(all_labels, all_preds, average='weighted', zero_division=0))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

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

def evaluate_model(model, X, y, batch_size=64, device=None, task_name="",
                   print_report=True, return_results=True, test_size=None,
                   random_state=None, criterion=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,in_channels,
            kernel_size=kernel_size,padding=padding,
            groups=in_channels, bias=False   # ⭐关键
        )
        self.pointwise = nn.Conv1d(
            in_channels,out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class ResDepthwiseBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            in_c, out_c, kernel_size=k, padding=k//2
        )
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        res = self.shortcut(x)
        return self.relu(out + res)
    
class MultiScaleBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # 两个不同感受野
        self.branch_k3 = ResDepthwiseBlock(32, 64, 3)
        self.branch_k5 = ResDepthwiseBlock(32, 64, 5)
        # ⭐新增融合层
        self.fuse = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
    def forward(self, x):
        f1 = self.branch_k3(x)
        f2 = self.branch_k5(x)
        # concat
        return torch.cat([f1, f2], dim=1)  # (B,128,L)
    
class SharedStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
    def forward(self, x):
        return self.stem(x)
    
class MultiScaleCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.stem = SharedStem()
        self.multi_scale = MultiScaleBranch()
        self.eca = ECA1D(128)
        self.pool = nn.AdaptiveAvgPool1d(4) # 改成保留一点结构信息
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        f = self.multi_scale(x)
        f = self.eca(f)
        f = self.pool(f)
        out = self.fc(f)
        return out
    
def train_multiscale(X_train, X_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reshape
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    output_dim = len(np.unique(y_train.numpy()))

    model = MultiScaleCNN(output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

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

        print(f"[MultiScale] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    evaluate_baseline(model, X_test, y_test, device)


