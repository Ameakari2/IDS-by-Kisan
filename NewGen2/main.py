import feature_selection
import dataset_loader
import train2
import config
from sklearn.model_selection import train_test_split
import pandas as pd

def run():
    X_train, X_test, y_train, y_test = dataset_loader.load_dataset()
    if config.MODE == "binary":
        train2.train_cnn(X_train.values, X_test.values, y_train.values, y_test.values, "binary")
    else:
        train2.train_cnn(X_train.values, X_test.values, y_train.values, y_test.values, "multi")

if __name__ == "__main__":
    if config.MODE == "binary":
        print("Running Binary Classification...")
    else:
        print("Running Multi Classification...")
    run()
