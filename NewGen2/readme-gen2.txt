main：run和run2，分别对应简单CNN，Dual-branch CNN，支持联邦（双分支的联邦未修改）

dataset_loader：数据清洗、one-hot编码；删除两个数据集本身的重复数据；删除训练集中与测试集重复的数据

preprocess：binary和multi两种模式；删除缺省值；proto, service, state 关键的三个特征；删除几个列；normalize

feature_selection：线性删除特征，没啥用

train2：简单1D CNN结构；binary和multi两种模式；检查标签分布、输入数据统计；未使用增量学习，ewc = None；

train3：从CNN+EWC变为了双分支CNN+EWC+LwF+CBAM，双分支特征提取--特征拼接--经过CBAM注意力加权--全局池化与分类；1D-CBAM 注意力机制；

drift-detection：漂移检测——如果检测到漂移。触发增量学习，没啥用

incremental：回放（Class Balanced Replay）&EWC（弹性权重巩固）；LwF 知识蒸馏损失（响应蒸馏）

federated：将集中式数据均匀划分给多个客户端；客户端复制一份全局模型进行本地更新；服务器端进行 FedAvg 聚合计算


