# Cargar librerías necesarias
library(readr)
library(dplyr)
library(ggplot2)
library(forecast)
library(cluster)
library(solitude)  # Alternativa para Isolation Forest
library(tidyverse)
library(zoo)

# 1. Cargar y explorar el dataset
data <- read_csv("./METR-LA.csv")  # Asegúrate de que el archivo esté en el mismo directorio

print(head(data))
print(summary(data))

# Verificar los nombres de las columnas
print(colnames(data))

# 2. Preprocesamiento de datos
# Si la primera columna es el timestamp, renombrarla
colnames(data)[1] <- "Timestamp"

# Convertir la columna de timestamp a un objeto datetime
data$Timestamp <- as.POSIXct(data$Timestamp, format = "%Y-%m-%d %H:%M:%S")
data[data == 0.0] <- NA

data 

data <- data %>%
  mutate(across(-Timestamp, ~ na.approx(., na.rm = FALSE)))

data

# Extraer características temporales
data <- data %>%
  mutate(Hour = as.numeric(format(Timestamp, "%H")),
         DayOfWeek = weekdays(Timestamp),
         IsWeekend = ifelse(DayOfWeek %in% c("Saturday", "Sunday"), 1, 0))

# 3. Modelado de series temporales (ARIMA)
# Seleccionar un sensor específico para el pronóstico
sensor_data <- data$`773869`

# Convertir a un objeto de serie temporal
ts_data <- ts(sensor_data, frequency = 288)  # 288 intervalos de 5 minutos en un día
ts_data
# Load required package
library(forecast)

# Assume ts_data is already defined as a time series, e.g.:
# ts_data <- ts(c(64.375, 62.667, 64.000, ...), frequency = 288, start = c(1, 1))

# Set the initial training window size
initial_train <- 200   # adjust this value as needed

# Number of iterations to perform (only 10 iterations)
num_iterations <- 1000

# Forecast horizon: forecast one time step ahead
h <- 1

# Total number of observations in ts_data
n <- length(ts_data)
if (initial_train + num_iterations > n) {
  stop("Not enough observations for the specified number of iterations.")
}

# Vectors to store forecasts and actual values
predictions <- numeric(num_iterations)
actuals <- numeric(num_iterations)

# Rolling forward (expanding window) validation for 10 iterations
for(i in 1:num_iterations) {
  # Create training set from the start to the current point
  train_data <- ts(ts_data[1:(initial_train + i - 1)],
                   frequency = frequency(ts_data),
                   start = start(ts_data))
  
  # Fit the ARIMA(2,0,0) model to the training data
  model <- Arima(train_data, order = c(2, 0, 0), method = "ML")
  
  # Forecast the next observation
  fc <- forecast(model, h = h)
  
  # Save the forecast and the actual observation
  predictions[i] <- fc$mean[h]
  actuals[i] <- ts_data[initial_train + i]
}

# Calculate forecast error metric (e.g., Mean Absolute Percentage Error)
errors <- actuals - predictions
mape <- mean(abs(errors) / abs(actuals)) * 100

cat("MAPE:", mape, "\n")

# Optionally, plot the actual data and the forecasts for the iterations
plot(ts_data, main = "Rolling Forecast (10 iterations) using ARIMA(2,0,0)",
     xlab = "Time", ylab = "Value", col = "black")
points((initial_train + 1):(initial_train + num_iterations), predictions,
       col = "red", type = "l", lwd = 2)
legend("topright", legend = c("Actual", "Forecast"),
       col = c("black", "red"), lty = 1, lwd = 2)

