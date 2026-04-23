import config
import dataset_loader
import train, train3

def run():
    Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test = dataset_loader.load_dataset()
    if config.MODE == "multi":
        train.train_dual(Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test)
    if config.MODE == "binary":
        train.train_binary(Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test)
    
def run_baseline():
    # Xs_train, Xd_train, Xs_test, Xd_test, y_train, y_test = dataset_loader.load_dataset()
    X_train, X_test, y_train, y_test = dataset_loader.load_dataset_baseline()
    train.train_baseline(X_train, X_test, y_train, y_test)

def run3():
    X_train, X_test, y_train, y_test = dataset_loader.load_dataset_multiscale()
    if config.MODE == "multi":
        train3.train_multiscale(X_train, X_test, y_train, y_test)
    if config.MODE == "binary":
        train3.train_multiscale(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    if config.MODE == "binary":
        print("\nRunning Binary Classification...")
    elif config.MODE == "multi":
        print("\nRunning Multi Classification...")
    else:
        raise ValueError("mode must be 'binary' or 'multi'")
    # run3()
    run_baseline()
