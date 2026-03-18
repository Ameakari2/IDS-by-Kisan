from river.drift import ADWIN
# pip install river

class DriftDetector:
    def __init__(self):
        self.adwin = ADWIN()
    def update(self, error):

        """
        输入当前预测是否错误
        0 = 正确
        1 = 错误
        """
        self.adwin.update(error)
        if self.adwin.drift_detected:
            return True
        return False
    
"""
# 在train.py中使用 需修改 evaluation 部分
model.eval()
detector = DriftDetector()
drift_triggered = False
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        y_true_batch = y_batch.numpy()

        for pred, true in zip(preds, y_true_batch):
            error = 0 if pred == true else 1
            drift = detector.update(error)
            if drift:
                drift_triggered = True
                print("Concept Drift Detected!")
                break
"""

# 如果检测到漂移：触发增量学习
"""
if drift_triggered:
    print("Trigger Incremental Learning...")
    incremental_update(model, new_data_loader)
"""
