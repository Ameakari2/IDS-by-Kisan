main：run和run2，分别对应简单CNN，Dual-branch CNN，支持联邦（双分支的联邦未修改）

dataset_loader：数据清洗、one-hot编码；删除两个数据集本身的重复数据；删除训练集中与测试集重复的数据

preprocess：binary和multi两种模式；删除缺省值；proto, service, state 关键的三个特征；删除几个列；normalize

train2：简单1D CNN结构；binary和multi两种模式；检查标签分布、输入数据统计；train_cnn增量学习使用类别平衡回放，未使用ewc；新增train_c，只有CNN作为基线模型

train3：从CNN+EWC变为了双分支CNN+EWC+LwF+CBAM，双分支特征提取--特征拼接--经过CBAM注意力加权--全局池化与分类；①1D-CBAM 注意力机制分为channel和spatial；②注意力机制从CBAM改为ECA；③双分支为静态+动态

train4：从 Dual Branch CNN 改为 格拉姆角场 (GAF) + 轻量级 2D-CNN；深度可分离卷积块（轻量化）。增量学习：类别平衡回放+EWC和LwF双约束。训练过程：1D到2D转换；X为image而不再是tensor，y还是tensor

incremental：类别平衡回放（Class Balanced Replay）+ EWC（弹性权重巩固）；LwF 知识蒸馏损失（响应蒸馏）Learning without Forgetting

federated：将集中式数据均匀划分给多个客户端；客户端复制一份全局模型进行本地更新；服务器端进行 FedAvg 聚合计算

feature_selection：线性删除特征，没啥用

drift-detection：漂移检测——如果检测到漂移。触发增量学习，没啥用
