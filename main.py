import matplotlib.pyplot as plt
import numpy as np

from utils import shift_dataset, data_preparation
from model import FAInance


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

df = shift_dataset('data/full.csv', days=1)

sub_df, X_train, y_train, X_val, y_val, X_test, y_test, train_X_reshaped, valid_X_reshaped, test_X_reshaped = data_preparation(df)
#
FAInance_model = FAInance(df)

FAInance_model.train(train_X_reshaped, y_train, valid_X_reshaped, y_val, epochs=100, lr=0.01)

# FAInance_model.plot_history()

FAInance_model.save()    # default to models/try.h5

FAInance_model = FAInance.load(path="models/best_try.h5")

# make predictions on test set
y_pred = np.array([x[0] for x in FAInance_model.predict(test_X_reshaped)])

# print(y_pred)

# for i in range(1, len(y_pred)):
#     print(sub_df['Date'].tolist()[-len(y_pred) + i])
#     print("Rispetto al giorno prima:", "sale" if y_pred[i] > y_pred [i-1] else "scende")

FAInance_model.plot_results(y_pred, y_test)


# this strategy is bullshit but run it if u want
def strategy(y_pred, y_test, budget):
    portfolio = budget
    bought = 0
    txs = 0
    perse = 0

    for i in range(len(y_test) - 1):
        if y_test[i]: break

        predicted_price = y_pred[i + 1]
        my_current_price = y_pred[i]
        current_price = y_test[i]
        tomorrow_price = y_test[i + 1]

        print("Il mio budget attuale è:", portfolio, "$")
        print("Ho fatto", txs, "transazioni")
        print("Il prezzo di oggi è:", current_price, "BTC")
        print("Il prezzo di domani sarà:", predicted_price, "BTC")
        print("Il prezzo di domani è più", "basso" if predicted_price < current_price else "alto")
        if my_current_price < predicted_price:
            spesa = budget * 10/100
            print("Spendo il 10% del mio budget:", spesa)
            bought += spesa/current_price
            print("Il valore di acquisto è", spesa, "diviso", current_price, "BTC")
            portfolio -= spesa
            print("Sottraggo dal mio budget la spesa.")
            if tomorrow_price < current_price:
                print("Il prezzo di domani era in realtà più basso:", tomorrow_price)
                perse += 1

        else:
            vendita = bought * current_price
            print("Vendo al prezzo attuale tutto ciò che ho acquistato finora")
            print("Vendo a prezzo pieno:", vendita)
            portfolio += vendita
            bought = 0
            if tomorrow_price > current_price:
                print("Il prezzo di domani era in realtà più alto:", tomorrow_price)
                perse += 1

        txs += 1

    return txs, perse, portfolio + bought * current_price


# txs, perse, portfolio = strategy(y_pred, y_test, 100)
#
# print("Numero di transazioni:", txs)
# print("Numero di transazioni perse:", perse)
# print("Budget finale:", portfolio)
