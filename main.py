import matplotlib.pyplot as plt
import numpy as np
import asyncio

from utils import shift_dataset, data_preparation, strategy
from model import FAInance
from binance_utils import connect, download

# # Uncomment to redownload locally the whole available dataset
# download("BTCUSDT", "2017-01-08", "2023-07-16")

# # Uncomment (and modify connect) to retrieve real time data
# coroutine = connect()
# t = asyncio.run(coroutine)

# # THIS HAS NO MEANING TO EXIST ANYMORE, DON'T UNCOMMENT, DON'T USE
# # Uncomment to clean separator for investing datasets (first run!)
# from utils import clean_separator
# clean_separator()
#
# # Uncomment to get rid of not common dates (first run!)
# from utils import fill_dates
# fill_dates()
#
# # Uncomment to rebuild dataset (first run!)
# from utils import get_dataset
# get_dataset()

df = shift_dataset('data/binance_data/clean_1MIN.csv', days=1)

sub_df, X_train, y_train, X_val, y_val, X_test, y_test,train_X_reshaped, valid_X_reshaped, test_X_reshaped \
    = data_preparation(df, start_date='2014-01-01', end_date='2023-07-17')

# # Uncomment to train a new model
# FAInance_model = FAInance(df)
#
# FAInance_model.train(train_X_reshaped, y_train, valid_X_reshaped, y_val, epochs=100, lr=0.01)
#
# # FAInance_model.plot_history()
#
# FAInance_model.save()

FAInance_model = FAInance.load(path="models/1MIN_15step_btc_vol.h5")

# make predictions on test set
y_pred = np.array([x[0] for x in FAInance_model.predict(test_X_reshaped)])

# print(y_pred)

strategy(y_test, y_pred)

# FAInance_model.plot_results(y_pred[-15:], y_test[-15:])

