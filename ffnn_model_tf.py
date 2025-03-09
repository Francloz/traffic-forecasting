import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tf.keras import layers, models


class FFNN():
    def __init__(self):
        pass

    def create_dataset(data, look_back, future_steps):
        X, y = [], []
        for i in range(len(data) - look_back - future_steps):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back:i + look_back + future_steps])
        return np.array(X), np.array(y)

    def training_model(self, model, X_train, y_train):
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        return model, X_train, y_train, history

    def evaluating(self, model, X_test, y_test):
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Loss: {test_loss:.4f}')

    def predicting(self, model, X_test):
        predictions = model.predict(X_test)
        pred_5 = predictions[:, 0]
        pred_15 = predictions[:, 1]
        pred_30 = predictions[:, 2]
        return predictions, pred_5, pred_15, pred_30

    def performance_metrics(self, predictions, y_test):
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    def visualizing(self, history):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def main(self):
        data_normalized = []
        look_back = 12  # Past 12 time steps
        future_steps = 3  # Predict next 3 steps (5, 15, 30 min)
        X, y = self.create_dataset(data_normalized, look_back, future_steps)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the model
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(look_back,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(future_steps)  # Output layer for multi-step prediction
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

        # Training the model
        model, X_train, y_train, history = self.training_model(model, X_train, y_train)

        # Evaluating the model
        self.evaluating(model, X_test, y_test)

        # Predicting future values (Predict traffic at 5, 15, and 30 minutes)
        predictions, pred_5, pred_15, pred_30 = self.predicting(model, X_test)
        print(f'pred_5:\n {pred_5}')
        print(f'pred_15:\n {pred_15}')
        print(f'pred_30:\n {pred_30}')

        # Obtaining performance metrics
        self.performance_metrics(predictions, y_test)

        self.visualizing(history)
