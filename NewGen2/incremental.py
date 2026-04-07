import torch
import torch.nn as nn
import torch.nn.functional as F
import random
"""
原理：
EWC 核心类（最重要部分）：
实现保存旧权重、计算费雪矩阵、计算正则损失三个核心功能
"""

# version 1, 存储旧任务样本, Replay 中大多数是 Normal 
"""
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []

    def add(self, x, y):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((x.cpu(), y.cpu()))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)
"""
# version 2, class → sample list, 实现 Class Balanced Replay
class BalancedReplayBuffer:
    def __init__(self, num_classes, samples_per_class=200):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.buffer = {i: [] for i in range(num_classes)}
    def add(self, x, y):
        label = int(y.item())
        if len(self.buffer[label]) < self.samples_per_class:
            self.buffer[label].append((x.cpu(), y.cpu()))
    def sample(self, batch_size):
        xs = []
        ys = []
        per_class = max(1, batch_size // self.num_classes)
        for c in self.buffer:
            samples = self.buffer[c]
            if len(samples) == 0:
                continue
            selected = random.sample(samples, min(per_class, len(samples)))
            for x, y in selected:
                xs.append(x)
                ys.append(y)
        return torch.stack(xs), torch.stack(ys)
    
# Elastic Weight Consolidation
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self.compute_fisher(dataloader)

    def compute_fisher(self, dataloader):
        # 计算费雪矩阵：用旧任务数据，梯度平方的平均值
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        # 保存当前模型参数 然后计算 Fisher Information
        self.model.eval()
        criterion = nn.CrossEntropyLoss() # 多分类常用
        # criterion = nn.BCEWithLogitsLoss() # 二分类常用 / 直接不用
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.model.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

"""
注：你原来的 EWC 类写得很好，
由于在获取需要求梯度的参数时使用了 if p.requires_grad 
这完美契合了我们后续要冻结静态分支的设计，所以 EWC 部分无需改动。
"""
def lwf_distillation_loss(y_pred, y_old_pred, T=2.0):
    """
    LwF 知识蒸馏损失 (响应蒸馏)
    通过软标签让新模型记住旧模型的输出分布
    y_pred: 新模型的输出 logits
    y_old_pred: 旧模型的输出 logits (需保持无梯度状态)
    T: 温度系数 (Temperature)，通常设置为 2.0 左右
    """
    # 新模型的输出使用 log_softmax
    p_s = F.log_softmax(y_pred / T, dim=1)
    # 旧模型的输出作为目标，使用 softmax
    p_t = F.softmax(y_old_pred / T, dim=1)   
    # 计算 KL 散度
    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T ** 2)
    return loss
