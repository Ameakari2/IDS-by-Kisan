"""
版本说明：
4.2v3 的 code 更改了新的模型构建，从 train2 升级为 train3 
从CNN+EWC变为了双分支CNN+EWC+LwF+CBAM
未使用联邦
"""
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from incremental import BalancedReplayBuffer, EWC, lwf_distillation_loss

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

# ==========================================
# 1D-CBAM 注意力机制
# ==========================================
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM1D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# ==========================================
# 双分支 CNN 架构
# ==========================================
class DualBranchCNN(nn.Module):
    def __init__(self, output_dim, input_channels=1):
        super().__init__()
        
        # 1. 静态通用分支 (Static Branch) - 捕捉通用协议特征
        self.static_branch = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
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
        
        # 2. 动态进化分支 (Dynamic Branch) - 捕捉新数据变体
        self.dynamic_branch = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
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

        # 3. 融合模块：两个分支 (128+128=256) 拼接后经过 CBAM
        self.cbam = CBAM1D(in_planes=256)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # 双分支特征提取
        feat_static = self.static_branch(x)
        feat_dynamic = self.dynamic_branch(x)
        
        # 特征拼接 (Batch, Channels, Length) -> 维度变为 256
        feat_concat = torch.cat([feat_static, feat_dynamic], dim=1)
        
        # 经过 CBAM 注意力加权
        feat_attended = self.cbam(feat_concat)
        
        # 全局池化与分类
        out = self.pool(feat_attended)
        out = self.fc(out)
        return out

    def freeze_static(self):
        """冻结静态分支的权重，用于微调阶段"""
        for param in self.static_branch.parameters():
            param.requires_grad = False
        print("Static branch is now frozen.")

# ==========================================
# 更新后的训练逻辑 (支持双约束)
# ==========================================
def train_cnn(X_train, X_test, y_train, y_test, task, 
              is_incremental=False, old_model=None, ewc_instance=None):
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

    # 初始化新模型
    model = DualBranchCNN(output_dim=output_dim).to(device)

    # 如果是增量学习阶段
    if is_incremental:
        if old_model is not None:
            # 继承旧模型参数
            model.load_state_dict(old_model.state_dict())
            old_model.eval() # 确保旧模型在 eval 模式
        
        # 冻结静态分支
        model.freeze_static()

    criterion = nn.CrossEntropyLoss()
    # 优化器会自动忽略 requires_grad=False 的参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    
    epochs = config.CNN_EPOCHS if hasattr(config, "CNN_EPOCHS") else 20
    print(f"\nTraining DualBranch CNN (Incremental Mode: {is_incremental})...")

    buffer = BalancedReplayBuffer(num_classes=output_dim, samples_per_class=200)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 经验回放
            if any(len(v) > 0 for v in buffer.buffer.values()):
                old_x, old_y = buffer.sample(16)
                old_x = old_x.to(device)
                old_y = old_y.to(device)
                X_batch = torch.cat([X_batch, old_x], dim=0)
                y_batch = torch.cat([y_batch, old_y], dim=0)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # --- 双约束微调 ---
            if is_incremental:
                # 约束 1: EWC (限制参数空间)
                if ewc_instance is not None:
                    loss += 0.4 * ewc_instance.penalty(model)
                
                # 约束 2: LwF (限制输出空间)
                if old_model is not None:
                    with torch.no_grad():
                        old_outputs = old_model(X_batch)
                    loss += 0.5 * lwf_distillation_loss(outputs, old_outputs, T=2.0)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 将新数据加入 Buffer
            for i in range(len(X_batch)):    
                if i < 64:  
                    buffer.add(X_batch[i].detach(), y_batch[i].detach()) 

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    print("\nEvaluating on test set...")
    evaluate_model(
        model,
        X_test.cpu().numpy(),
        y_test.cpu().numpy(),
        batch_size=64,
        device=device,
        task_name=task,
        print_report=True,
        return_results=False
    )
    """
    # Save model
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_PATH, f"dual_{task}_v1.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    """
    print("Computing Fisher Information for EWC...")
    # 这里计算的 Fisher 矩阵将只会基于未被冻结的 dynamic_branch 和 fc 层
    new_ewc = EWC(model, train_loader, device)
    
    return model, new_ewc
