import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def create_dataset(data, look_back, future_steps):
        X, y = [], []
        for i in range(len(data) - look_back - future_steps):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back:i + look_back + future_steps])
        return np.array(X), np.array(y)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_model(self, model, X_train, y_train):
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        return model, X_train, y_train

    def predicting(self, model, X_train):
        model.eval()
        with torch.no_grad():
            predictions = model(X_train).numpy()
        return predictions

    def evaluating(self, predictions, y_train):
        mae = mean_absolute_error(y_train.numpy(), predictions)
        rmse = mean_squared_error(y_train.numpy(), predictions, squared=False)

        print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')

        pred_5 = predictions[:, 0]  # First step (5 minutes)
        pred_15 = predictions[:, 1]  # Second step (15 minutes)
        pred_30 = predictions[:, 2]  # Third step (30 minutes)
        return pred_5, pred_15, pred_30

    def main(self):
        data_normalized = []
        look_back = 12  # Past 12 time steps
        future_steps = 3  # Predict next 3 steps (5, 15, 30 min)

        X, y = self.create_dataset(data_normalized, look_back, future_steps)
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)

        # Initialize the model
        input_dim = 12  # Example: 12 time steps of past data
        hidden_dim = 64
        output_dim = 3  # Predict 5, 15, 30 minutes into the future
        model = FFNN(input_dim, hidden_dim, output_dim)

        # Training the model
        model, X_train, y_train = self.training_model(model, X_train, y_train)

        # Predicting values
        predictions = self.predicting(model, X_train)

        # Evaluating predictions
        pred_5, pred_15, pred_30 = self.evaluating(predictions, y_train)
        print(f'pred_5:\n {pred_5}')
        print(f'pred_15:\n {pred_15}')
        print(f'pred_30:\n {pred_30}')
