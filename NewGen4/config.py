# --- Path Settings ---
MODEL_SAVE_PATH = "D:/ProgramTest/SystemMaster/NewGen3"

TRAIN_DATA = "D:/ProgramTest/SystemMaster/datasets/UNSW_NB15_training-set.csv"
TEST_DATA = "D:/ProgramTest/SystemMaster/datasets/UNSW_NB15_testing-set.csv"

TRAIN_PATH = "train_processed.csv"
TEST_PATH = "test_processed.csv"

# --- Mode Setting ---
MODE = "multi" # binary 和 multi 两种模式（小写）

# --- CNN Settings ---
LEARNING_RATE=0.001 # 合适的学习率
CNN_EPOCHS = 20
BATCH_SIZE = 64

# --- Feature Selection ---
CORRELATION_THRESHOLD = 0.1 # 从0.3改为0.1

# --- Original Training Files ---
TEST_SIZE_BINARY = 0.2
TEST_SIZE_MULTI = 0.3
RANDOM_STATE = 42

# --- FL Settings ---
USE_FEDERATED = False    # 是否启用联邦学习
FL_NUM_CLIENTS = 5       # 模拟的客户端数量
FL_ROUNDS = 20           # 服务器聚合的全局轮次 (Communication Rounds)
FL_LOCAL_EPOCHS = 3      # 每个客户端在本地更新的 Epoch 数
