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
# Ajustar un modelo ARIMA
arima_model <- auto.arima(ts_data)
print(summary(arima_model))




# Pronosticar las próximas 24 horas (288 intervalos)
forecast_result <- forecast(arima_model, h = 288)
plot(forecast_result, main = "Pronóstico ARIMA para el Sensor 773869")


##ahí me quedé porque mi laptop se me atora cuando carga esa parte. haha
#pendiente de analizar

# 4. Clustering (K-Means)
# Seleccionar un subconjunto de sensores para clustering
sensor_subset <- data[, 2:11]  # Ejemplo: primeros 10 sensores

# Realizar clustering con K-Means
set.seed(123)
kmeans_result <- kmeans(sensor_subset, centers = 3)  # 3 clusters

# Añadir etiquetas de cluster al dataset
sensor_subset$Cluster <- as.factor(kmeans_result$cluster)

# Visualizar los clusters
ggplot(sensor_subset, aes(x = `773869`, y = `767541`, color = Cluster)) +
  geom_point() +
  labs(title = "Clustering K-Means de Sensores de Tráfico",
       x = "Sensor 773869",
       y = "Sensor 767541")

# 5. Detección de anomalías (Isolation Forest con solitude)
# Seleccionar un sensor específico para detección de anomalías
sensor_data <- data$`773869`

# Crear un dataframe para solitude
anomaly_data <- data.frame(Speed = sensor_data)

# Ajustar el modelo Isolation Forest
iso_model <- isolationForest$new()
iso_model$fit(anomaly_data)

# Predecir anomalías
anomaly_scores <- iso_model$predict(anomaly_data)
data$Anomaly <- ifelse(anomaly_scores$anomaly_score > 0.65, 1, 0)  # Umbral para anomalías

# Visualizar anomalías
ggplot(data, aes(x = Timestamp, y = `773869`, color = as.factor(Anomaly))) +
  geom_point() +
  labs(title = "Detección de Anomalías para el Sensor 773869",
       x = "Tiempo",
       y = "Velocidad (mph)",
       color = "Anomalía")

# 6. Evaluación del modelo ARIMA
# Dividir los datos en entrenamiento y prueba
train <- ts_data[1:(length(ts_data) - 288)]
test <- ts_data[(length(ts_data) - 287):length(ts_data)]

# Ajustar ARIMA en los datos de entrenamiento
arima_model <- auto.arima(train)

# Pronosticar en los datos de prueba
forecast_result <- forecast(arima_model, h = 288)

# Calcular métricas de evaluación
print(accuracy(forecast_result, test))

# 7. Evaluación del clustering (Silhouette Score)
# Calcular el índice de silueta
silhouette_score <- silhouette(kmeans_result$cluster, dist(sensor_subset[, 1:10]))
print(mean(silhouette_score[, 3]))

# 8. Visualización de resultados combinados
ggplot(data, aes(x = Timestamp)) +
  geom_line(aes(y = `773869`, color = "Velocidad Real")) +
  geom_line(aes(y = forecast_result$mean, color = "Velocidad Pronosticada")) +
  geom_point(aes(y = ifelse(Anomaly == 1, `773869`, NA), color = "Anomalía"), size = 2) +
  labs(title = "Pronóstico de Velocidad y Detección de Anomalías",
       x = "Tiempo",
       y = "Velocidad (mph)",
       color = "Leyenda")

