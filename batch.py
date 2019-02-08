from data_preparation import get_data
from stock_prediction import predict, train
import sys

if __name__ == '__main__':
    argv = sys.argv
    action = "predict"
    if len(argv) > 1:
        action = argv[1]
    if action == "predict":
        print("===================predict====================")
        get_data(False)
        predict()
    elif action == "train":
        print("===================train====================")
        get_data(True)
        train()

