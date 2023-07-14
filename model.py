import keras.models
import pandas as pd
import tensorflow as tf
import pickle

from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from matplotlib import pyplot as plt


class FAInance:
    def __init__(self, df):
        initializer = tf.keras.initializers.GlorotNormal(seed=35)
        self.df = df

        # Define the Keras model
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(1, 2), return_sequences=True))
        self.model.add(LSTM(32, return_sequences=True))
        self.model.add(LSTM(16))
        self.model.add(Dense(128, activation='relu', kernel_initializer=initializer,
                        bias_initializer='zeros',
                        kernel_regularizer=l1(0.01),
                        bias_regularizer=l2(0.01)))
        self.model.add(Dense(256, activation='relu', kernel_initializer=initializer,
                        bias_initializer='zeros',
                        kernel_regularizer=l1(0.02),
                        bias_regularizer=l2(0.02)))
        self.model.add(Dense(64, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer,
                        bias_initializer='zeros',
                        kernel_regularizer=l1(0.01),
                        bias_regularizer=l2(0.01)))
        self.model.add(Dense(32, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer,
                        bias_initializer='zeros',
                        kernel_regularizer=l1(0.01),
                        bias_regularizer=l2(0.01)))
        self.model.add(Dense(16, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer,
                        bias_initializer='zeros',
                        kernel_regularizer=l1(0.01),
                        bias_regularizer=l2(0.01)))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

    def train(self, train_X_reshaped, y_train, valid_X_reshaped, y_val,
              epochs=400, batch_size=128, lr=0.001, loss='mean_squared_error'):
        # compile the model
        self.model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=['mae'])

        # saves the best model out of the result of each epoch (not the last!)
        checkpoint = ModelCheckpoint('models/best_try.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0)

        # after <patience> epochs in which loss is stable (doesn't decrease) it stops the training
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        # learning reate dynamic tweaks to make the model gradient descent smoother
        learning_rate_scheduler = LearningRateScheduler(self.lr_scheduler)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0)

        # train the model
        self.history = self.model.fit(train_X_reshaped, y_train, validation_data=(valid_X_reshaped, y_val),
                                      epochs=epochs, batch_size=batch_size,
                                      callbacks=[checkpoint, early_stopping, learning_rate_scheduler, reduce_lr])

        return self.history

    def lr_scheduler(self, epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.001)

    def evaluate(self, test_X_reshaped, y_test): return self.model.evaluate(test_X_reshaped, y_test)

    def predict(self, test_X_reshaped): return self.model.predict(test_X_reshaped)

    def get_history(self): return self.history

    def plot_history(self):
        # Access the loss and MAE values from the training history
        loss = self.history.history['loss']
        mae = self.history.history['mae']

        # Access the validation loss and MAE values from the training history
        val_loss = self.history.history['val_loss']
        val_mae = self.history.history['val_mae']

        # Plot the loss
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        # Plot the MAE
        plt.plot(mae, label='Training MAE')
        plt.plot(val_mae, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.show()

    def plot_results(self, y_pred, y_test):

        # Plotting the values
        xpositions = range(len(self.df['Date'][-len(y_pred):]))
        xlabels = self.df['Date'][-len(y_pred):]
        plt.plot(self.df['Date'][-len(y_pred):], y_pred)
        plt.xticks(xpositions[::2], xlabels[::2], rotation='vertical')
        # Adding labels and title
        plt.xlabel('Day')
        plt.ylabel('Value')
        plt.title('Predicted BTC-USD Trend')
        plt.tight_layout()

        plt.show()

        # Plotting the values
        xpositions = range(len(self.df['Date'][-len(y_test):]))
        xlabels = self.df['Date'][-len(y_test):]
        plt.plot(self.df['Date'][-len(y_test):], y_test)
        plt.xticks(xpositions[::2], xlabels[::2], rotation='vertical')
        # Adding labels and title
        plt.xlabel('Day')
        plt.ylabel('Value')
        plt.title('Real BTC-USD Trend')
        plt.tight_layout()

        plt.show()

    # This basically doesn't have so much sense because the best model is already saved by using
    # ModelCheckpoint before fitting.
    def save(self, path="models/try.h5"): self.model.save(path)

    # Loads a model
    @staticmethod
    def load(path="models/try.h5"):
        f = FAInance(pd.read_csv("data/clean.csv"))
        f.model = keras.models.load_model(path)
        return f
