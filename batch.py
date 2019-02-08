from data_preparation import get_data
from stock_prediction import predict, train
# one time running, comment them after the models are trained.
get_data(True)
train()
# daily running
get_data(False)
predict()