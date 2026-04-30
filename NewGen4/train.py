import config
from train3 import evaluate_threshold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

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

def evaluate_model(model, Xs, Xd, y, device):
    model.eval()

    Xs = torch.tensor(Xs, dtype=torch.float32)
    Xd = torch.tensor(Xd, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(Xs, Xd, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xs, xd, labels in loader:
            xs = xs.to(device)
            xd = xd.to(device)

            outputs, _, _ = model(xs, xd)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== Evaluation ===")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("Recall:", recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    print("F1:", f1_score(all_labels, all_preds, average='weighted', zero_division=0))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels   # ⭐关键
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ECA1D(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA1D, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

class LightConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            in_c, out_c, kernel_size=k, padding=k//2
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

class DualBranchCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        """
        # Static Branch（稳定特征）
        self.static_branch = nn.Sequential(
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

        # Dynamic Branch（变化特征）
        self.dynamic_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # 更大感受野
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)  # 强化动态性
        )
        """
        # 4-21v1pro======
        # ⭐ 静态分支（语义更稳定 → 小卷积核）
        self.static_branch = nn.Sequential(
            LightConvBlock(1, 32, 3),
            LightConvBlock(32, 64, 3),
            DepthwiseSeparableConv1d(64, 128, 3, padding=1)
        )
        # ⭐ 动态分支（行为 → 大感受野）
        self.dynamic_branch = nn.Sequential(
            LightConvBlock(1, 32, 5),
            LightConvBlock(32, 64, 5),
            DepthwiseSeparableConv1d(64, 128, 3, padding=1)
        )
        
        # self.gate_fc = nn.Linear(256, 128)
        self.eca = ECA1D(256)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ====== Gated Fuion 的 fc 需要改小1/2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, x_static, x_dynamic):
        
        # ECA 纯净版
        f_s = self.static_branch(x_static)
        f_d = self.dynamic_branch(x_dynamic)
        f_s = F.adaptive_avg_pool1d(f_s, 1) # 两个分支统一到固定长度
        f_d = F.adaptive_avg_pool1d(f_d, 1)
        f = torch.cat([f_s, f_d], dim=1)  # concat（保留结构信息）, (B,256,L)
        f = self.eca(f)
        f = self.pool(f)  # (B,256,1)
        out = self.fc(f)
        """
        # Gated Fuion Version
        f_s = self.static_branch(x_static)
        f_d = self.dynamic_branch(x_dynamic)
        f_s = F.adaptive_avg_pool1d(f_s, 1).squeeze(-1)  # (B,128)
        f_d = F.adaptive_avg_pool1d(f_d, 1).squeeze(-1)  # (B,128)
        f_cat = torch.cat([f_s, f_d], dim=1)  # (B,256)
        gate = torch.sigmoid(self.gate_fc(f_cat))  # (B,128)
        f = gate * f_s + (1 - gate) * f_d  # (B,128)
        out = self.fc(f)
        """
        return out, f_s, f_d

def train_dual(Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reshape
    Xs_train = np.expand_dims(Xs_train, axis=1)
    Xd_train = np.expand_dims(Xd_train, axis=1)
    Xs_test = np.expand_dims(Xs_test, axis=1)
    Xd_test = np.expand_dims(Xd_test, axis=1)

    # tensor
    Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
    Xd_train = torch.tensor(Xd_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # dataloader
    dataset = TensorDataset(Xs_train, Xd_train, y_train)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # output dim
    output_dim = len(np.unique(y_train.numpy()))

    model = DualBranchCNN(output_dim).to(device)

    # 4-17v2======
    classes = np.unique(y_train.numpy())
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train.numpy()
    )
    weights = np.sqrt(weights)
    if len(weights) > 3:
        weights[3] *= 0.5 # 抑制类别3过强
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # criterion = FocalLoss(alpha=weights, gamma=2)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Class weights:", weights)
    # 4-17v2======
    
    # criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epochs = config.CNN_EPOCHS
    print("\nTraining Dual-Branch CNN...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xs, xd, labels in loader:
            xs = xs.to(device)
            xd = xd.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, f_s, f_d = model(xs, xd)
            # 分类损失
            loss_cls = criterion(outputs, labels)

            # ===== 解耦损失 =====
            f_s_flat = f_s.view(f_s.size(0), -1)
            f_d_flat = f_d.view(f_d.size(0), -1)

            cos_sim = F.cosine_similarity(f_s_flat, f_d_flat, dim=1)
            loss_diff = torch.mean(cos_sim)

            # ===== 两阶段训练 =====
            if epoch < epochs * 0.3:
                loss = loss_cls
            else:
                loss = loss_cls + 0.05 * loss_diff

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    print("\nEvaluating...")
    evaluate_model(model, Xs_test, Xd_test, y_test, device)

def train_binary(Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reshape
    Xs_train = np.expand_dims(Xs_train, axis=1)
    Xd_train = np.expand_dims(Xd_train, axis=1)
    Xs_test = np.expand_dims(Xs_test, axis=1)
    Xd_test = np.expand_dims(Xd_test, axis=1)
    # tensor
    Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
    Xd_train = torch.tensor(Xd_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(Xs_train, Xd_train, y_train)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    # 二分类 → output_dim = 2
    model = DualBranchCNN(output_dim=2).to(device)
    # 不再使用 class_weight
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epochs = config.CNN_EPOCHS
    print("\nTraining Binary Dual-Branch CNN...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xs, xd, labels in loader:
            xs = xs.to(device)
            xd = xd.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, f_s, f_d = model(xs, xd)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    print("\nEvaluating...")
    evaluate_model(model, Xs_test, Xd_test, y_test, device)

