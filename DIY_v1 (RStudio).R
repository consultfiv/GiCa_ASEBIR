#------------------------------------------------------------------------------------------
# SECCIÓN 1: Instalar y/o cargar liberías ---
# Función para instalar y cargar paquetes si no están instalados
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}
# Lista de paquetes necesarios
paquetes <- c("caret", "mlbench", "pROC", "randomForest", "OptimalCutpoints", "ggplot2")
# Aplicar la función a cada paquete
sapply(paquetes, install_if_missing)

#------------------------------------------------------------------------------------------
# SECCIÓN 2: Cargar dataset ---
data(PimaIndiansDiabetes)
df <- PimaIndiansDiabetes


#------------------------------------------------------------------------------------------
# SECCIÓN 3: Definir parámetros generales ---
OUTPUT <- "diabetes"  # Variable de salida (ajustar si cambia el dataset)
LEVEL1 <- "pos"       # Clase positiva (evento de interés)
LEVEL0 <- "neg"       # Clase negativa



#------------------------------------------------------------------------------------------
# SECCIÓN 4: Preparación de datos ---
# Convertir la variable de salida en factor con los niveles definidos
df[[OUTPUT]] <- factor(df[[OUTPUT]], levels = c(LEVEL0, LEVEL1))


# Dividir en conjunto de entrenamiento y prueba
set.seed(123)
trainIndex <- createDataPartition(df[[OUTPUT]], p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

#------------------------------------------------------------------------------------------
# SECCIÓN 5: Entrenamiento del modelo ---
# Definir control de entrenamiento con validación cruzada
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Ajustar modelo (Random Forest)
tune_grid <- expand.grid(mtry = c(2, 3, 4, 5))  # Número de variables en cada división

modelo <- train(as.formula(paste(OUTPUT, "~ .")), data = trainData, method = "rf", 
                trControl = train_control, metric = "ROC",
                tuneGrid = tune_grid,
                ntree = 1000)  # Ajusta el número de árboles

# Predicciones en test
pred <- predict(modelo, testData, type = "prob")

# Convertir en factores según umbral 0.5
pred_class <- factor(ifelse(pred[[LEVEL1]] > 0.5, LEVEL1, LEVEL0), levels = c(LEVEL0, LEVEL1))

# Matriz de confusión
conf_mat <- confusionMatrix(pred_class, testData[[OUTPUT]],
                            positive = LEVEL1)

# Extraer las métricas
metrics_total <- data.frame(
  F1Score = conf_mat$byClass["F1"],
  Sens = conf_mat$byClass["Sensitivity"],
  Spec = conf_mat$byClass["Specificity"],
  PPV = conf_mat$byClass["Pos Pred Value"],
  NPV = conf_mat$byClass["Neg Pred Value"]
)


#------------------------------------------------------------------------------------------
# SECCIÓN 6: Análisis de los resultados ---
# Calcular curva ROC y AUC
roc_curve <- roc(testData[[OUTPUT]], pred[[LEVEL1]])  # Usar la probabilidad de la clase positiva
AUC_value <- auc(roc_curve)
print(AUC_value)
# Añadir AUC al data frame de métricas
metrics_total$AUC <- AUC_value
# Extraer los datos de la curva ROC para graficar
roc_data <- data.frame(
  Sensitivity = roc_curve$sensitivities,
  Specificity = 1 - roc_curve$specificities
)
# Crear el gráfico con ggplot2
curvaROC <- ggplot(roc_data, aes(x = Specificity, y = Sensitivity)) +
  geom_line(color = "#008CBD", linewidth = 1.2) +  # Línea de la curva ROC
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +  # Línea de referencia (diagonal)
  labs(
    title = "Curva ROC",
    subtitle = paste("AUC =", round(AUC_value, 3)),
    x = "1 - Especificidad",
    y = "Sensibilidad"
  ) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1)) +  # Asegurar que el eje x comience en 0
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1.1)) +  # Asegurar que el eje y comience en 0
  theme_bw(base_size = 14) +  # Usar tema con bordes
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Centrar título
    plot.subtitle = element_text(hjust = 0.5, size = 14, face = "italic")  # Centrar subtítulo
  )
#------------------------------------------------------------------------------------------
# Añadir las probabilidades al dataset de test con el nombre correcto
testData$predicho <- pred[[LEVEL1]]
# Calcular el umbral óptimo basado en Sensibilidad y Especificidad (Youden)
opt_cut <- optimal.cutpoints(X = "predicho", status = OUTPUT, 
                             tag.healthy = LEVEL0, tag.disease = LEVEL1, 
                             methods = "Youden", data = testData)
#------------------------------------------------------------------------------------------
# Importancia de variables con caret
var_imp <- varImp(modelo)


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


cat("\014")  # Simula Ctrl + L para limpiar la consola

# VISUALIZAR RESULTADOS
print(conf_mat)

# Mostrar el gráfico
print(curvaROC)

# Mostrar métricas finales
print(metrics_total)

# Mostrar resultados cut-off points
summary(opt_cut)

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#var importance
print(var_imp)
plot(var_imp)

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------



