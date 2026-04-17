import config
import dataset_loader
import train

def run():
    """
    X_train, X_test, y_train, y_test = dataset_loader.load_dataset()
    if config.MODE == "binary":
        train.train_c(X_train, X_test, y_train, y_test, "binary")
    else:
        train.train_dual(X_train, X_test, y_train, y_test, "multi")
    """
    Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test = dataset_loader.load_dataset()
    train.train_dual(Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test, "multi")
   
if __name__ == "__main__":
    if config.MODE == "binary":
        print("\nRunning Binary Classification...")
    else:
        print("\nRunning Multi Classification...")
    run()
