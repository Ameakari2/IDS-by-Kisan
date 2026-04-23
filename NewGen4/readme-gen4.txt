【instructions】
この前作ったIDSシステムは全部うんこのようなものだから4月17日で斬新なバージョンを提出してきたんだ
brand-new preprocess


【へんか】
4-17v2：重新设置加权交叉熵，一定程度上抑制类别3，调小解耦 loss（0.05到0.01）

【dual-branch CNN】
Gated Fusion（融合方式），从concat改来
Focal Loss——不行

【4-23v2pro】**关键设计**
train3.py
深度可分离卷积+残差网络+ECA
细节设计：一维自适应平均池化层+共享stem层

train：baseline对比组的普通CNN；从multi转为binary

