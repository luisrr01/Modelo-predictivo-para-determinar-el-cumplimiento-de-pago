#---------------------------------#
# PROYECTO FINAL MINERÍA DE DATOS #
#---------------------------------#

# Para limpiar el workspace, por si hubiera algún dataset 
# o información cargada
rm(list = ls())

# Para limpiar el área de gráficos
graphics.off()

# Otras opciones
cat("\014")
options(scipen = 999)
options(digits = 4)

# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# Paquetes
library(pacman)
p_load(foreign, DataExplorer, recipes, MLmetrics, caret, 
       caTools, yardstick)


#-----------------------------------------#
#   CASO DEFAULT OF CREDIT CARD CLIENTS   #
#-----------------------------------------#

# Lectura de datos --------------------------------------------

library(foreign)
datos <-read.table("default of credit card clients.csv",
                   sep = ";", dec = ",",
                   header = T, stringsAsFactors = T)

attr(datos, "variable.labels") <- NULL

str(datos)

# No considerar la variable de identificación ID
datos$ID <- NULL 
str(datos)

table(datos$DEFPAY)
datos$DEFPAY <- factor(datos$DEFPAY, 
                       levels = c(0, 1),
                       labels = c("No_paga", "Si_paga"))
datos$SEX <- factor(datos$SEX, 
                    levels = c(1, 2),
                    labels = c("Masculino", "Femenino"))
datos$EDUCATION <- factor(datos$EDUCATION, 
                          levels = c(1, 2, 3, 4),
                          labels = c("Posgrado", "Universidad", "Secundaria", "Otros"))
datos$MARRIAGE <- factor(datos$MARRIAGE, 
                         levels = c(1, 2, 3),
                         labels = c("Casado", "Soltero", "Otros"))

datos$PAY_0 <- factor(datos$PAY_0,
                      levels = c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                      labels = c("Pago_debido", "Pago_Ret1M",
                                 "Pago_Ret2M", "Pago_Ret3M",
                                 "Pago_Ret4M", "Pago_Ret5M",
                                 "Pago_Ret6M", "Pago_Ret7M",
                                 "Pago_Ret8M", "Pago_Ret9M"))
datos$PAY_2 <- factor(datos$PAY_2,
                      levels = c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                      labels = c("Pago_debido", "Pago_Ret1M",
                                 "Pago_Ret2M", "Pago_Ret3M",
                                 "Pago_Ret4M", "Pago_Ret5M",
                                 "Pago_Ret6M", "Pago_Ret7M",
                                 "Pago_Ret8M", "Pago_Ret9M"))
datos$PAY_3 <- factor(datos$PAY_3,
                      levels = c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                      labels = c("Pago_debido", "Pago_Ret1M",
                                 "Pago_Ret2M", "Pago_Ret3M",
                                 "Pago_Ret4M", "Pago_Ret5M",
                                 "Pago_Ret6M", "Pago_Ret7M",
                                 "Pago_Ret8M", "Pago_Ret9M"))
datos$PAY_4 <- factor(datos$PAY_4,
                      levels = c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                      labels = c("Pago_debido", "Pago_Ret1M",
                                 "Pago_Ret2M", "Pago_Ret3M",
                                 "Pago_Ret4M", "Pago_Ret5M",
                                 "Pago_Ret6M", "Pago_Ret7M",
                                 "Pago_Ret8M", "Pago_Ret9M"))
datos$PAY_5 <- factor(datos$PAY_5,
                      levels = c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                      labels = c("Pago_debido", "Pago_Ret1M",
                                 "Pago_Ret2M", "Pago_Ret3M",
                                 "Pago_Ret4M", "Pago_Ret5M",
                                 "Pago_Ret6M", "Pago_Ret7M",
                                 "Pago_Ret8M", "Pago_Ret9M"))
datos$PAY_6 <- factor(datos$PAY_6,
                      levels = c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                      labels = c("Pago_debido", "Pago_Ret1M",
                                 "Pago_Ret2M", "Pago_Ret3M",
                                 "Pago_Ret4M", "Pago_Ret5M",
                                 "Pago_Ret6M", "Pago_Ret7M",
                                 "Pago_Ret8M", "Pago_Ret9M"))

str(datos)

contrasts(datos$DEFPAY)

prop.table(table(datos$DEFPAY)) * 100

plot_missing(datos)
# Imputación de missing values de forma manual ----
datos <- knnImputation(datos,
                       scale = TRUE, 
                       k = 3)
sum(is.na(datos))
plot_missing(datos)

# Seleccion de variables usando el algoritmo Boruta ----
library(Boruta)
set.seed(2021)
boruta.data <- Boruta(DEFPAY ~ ., data = datos, doTrace = 2)

print(boruta.data)

plot(boruta.data, cex.axis = 0.5)

plotImpHistory(boruta.data, lty = 1)

# Eliminación de variables no importantes

datos$EDUCATION  <- NULL
datos$SEX  <- NULL

# Selección de muestra de entrenamiento (70%) y evaluación (30%) ----
library(caret)
set.seed(2021) 

index      <- createDataPartition(datos$DEFPAY, 
                                  p = 0.7, 
                                  list = FALSE)

data.train <- datos[ index, ]            # 21001 datos trainig             
data.test  <- datos[-index, ]            # 8999 datos testing

addmargins(table(datos$DEFPAY))
round(prop.table(table(datos$DEFPAY)) * 100, 2)

addmargins(table(data.train$DEFPAY))
round(prop.table(table(data.train$DEFPAY)) * 100, 2)

addmargins(table(data.test$DEFPAY))
round(prop.table(table(data.test$DEFPAY)) * 100, 2)

# Pre-procesamiento de los datos con el paquete recipes ----
library(recipes)
set.seed(2021)
objeto_recipe <- recipe(DEFPAY~  .,
                        data =  data.train) %>%
  step_corr(all_numeric(), - all_outcomes(), threshold = 0.5) %>%
  step_nzv(all_predictors()) %>%
  step_range(all_numeric()) %>%   # Min-Max [0,1]
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep()

objeto_recipe

train <- bake(objeto_recipe, new_data = NULL)
test  <- bake(objeto_recipe, new_data = data.test)

train <- as.data.frame(train)
test  <- as.data.frame(test)

train <- train %>% select(1:9, 11:66, 10)
test <- test %>% select(1:9, 11:66, 10)
accuracy_
# Balanceo de datos con Smote (smote_train) ----
library(DMwR) 
set.seed(2021)
smote_train <- SMOTE(DEFPAY ~ ., 
                     data = train, 
                     perc.over  = 200,  # SMOTE
                     perc.under = 150)  # Undersampling                      

addmargins(table(train$DEFPAY))
addmargins(table(smote_train$DEFPAY))

target     <- "DEFPAY"
predictores <- setdiff(names(smote_train), target)
predictores

# Entrenamiento de los modelos con Validación Cruzada y usando el accuracy ----
ctrl <- trainControl(method="cv", number=10, classProbs =  TRUE)

# Modelado con algoritmos de clasificación ----
library(caret)

# 3.1 GLM ----------------------
set.seed(2021)
modelo_glm <- train(DEFPAY ~ ., 
                    data = smote_train,
                    method = "glm", family = "binomial", 
                    trControl = ctrl, 
                    tuneLength = 5,
                    metric = "Accuracy")

modelo_glm

modelo_glm$bestTune

#No cuenta con hiperparametro

plot(modelo_glm)

varImp(modelo_glm)
plot(varImp(modelo_glm))

PROBA.GLM <- predict(modelo_glm, test, type = "prob")
head(PROBA.GLM)
PROBA.GLM <- PROBA.GLM[, 2]

CLASE.GLM <- predict(modelo_glm, test) # umbral = 0.5
head(CLASE.GLM)

# Evaluando la performance de la Regresion Lineal 
# Matriz de Confusión
tabla1  <- table(Predicho = CLASE.GLM, Real = test$DEFPAY)
addmargins(tabla1)

library(yardstick)
autoplot(conf_mat(tabla1), type = "heatmap")  
summary(conf_mat(tabla1), event_level = "second") 

# Calcular el accuracy
accuracy_GLM <- mean(test$DEFPAY == CLASE.GLM); accuracy_GLM

# Calcular el error de mala clasificación
error <- mean(test$DEFPAY != CLASE.GLM) ; error

# Matriz de Confusión usando el paquete caret
library(caret)
cm_GLM <- caret::confusionMatrix(CLASE.GLM,
                                 test$DEFPAY,
                                 positive = "Si_paga")
cm_GLM

cm_GLM$table

cm_GLM$byClass["Sensitivity"] 
cm_GLM$byClass["Specificity"] 
cm_GLM$overall["Accuracy"]
cm_GLM$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.GLM, test$DEFPAY, plotROC = TRUE) -> auc_GLM
abline(0, 1, col = "red")
auc_GLM

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.GLM, real) -> logloss_GLM
logloss_GLM


test  <- bake(objeto_recipe, new_data = data.test)

train <- as.data.frame(train)
test  <- as.data.frame(test)

train <- train %>% select(1:9, 11:66, 10)
test <- test %>% select(1:9, 11:66, 10)

# Dividiendo usando el punto de corte óptimo con AUC
library(pROC)
roc <- roc(test[, target], PROBA.GLM)

umbral_glm <- pROC::coords(roc, "best")$threshold
umbral_glm

test$CLASE.GLM2 <- as.factor(ifelse(PROBA.GLM > umbral_glm,
                                    "Si_paga", 
                                    "No_paga"))

cm_GLM2 <- caret::confusionMatrix(test$CLASE.GLM2, 
                                  test[, target],
                                  positive = "Si_paga")

# Calcular el accuracy
accuracy_GLM2 <- mean(test$DEFPAY == test$CLASE.GLM2) ; accuracy_GLM2

cm_GLM2$byClass["Sensitivity"] 
cm_GLM2$byClass["Specificity"] 
cm_GLM2$overall["Accuracy"]
cm_GLM2$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.GLM, test$DEFPAY, plotROC = TRUE) -> auc_GLM2
abline(0, 1, col = "red")
auc_GLM2

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.GLM, real) -> logloss_GLM2
logloss_GLM2


# 3.2 KNN ----------------------
set.seed(2021)
modelo_knn <- train(DEFPAY ~ ., 
                    data = smote_train, 
                    method = "knn", 
                    trControl = ctrl,
                    tuneGrid = expand.grid(k=seq(1,7,2)),
                    metric = "Accuracy")

modelo_knn

modelo_knn$bestTune
#k = 1

plot(modelo_knn)

varImp(modelo_knn)
plot(varImp(modelo_knn))

PROBA.KNN <- predict(modelo_knn, test, type = "prob")
head(PROBA.KNN)
PROBA.KNN <- PROBA.KNN[, 2]

CLASE.KNN <- predict(modelo_knn, test) # 0.5
head(CLASE.KNN)

# Evaluando la performance del KNN Radial
# Matriz de Confusión
tabla2  <- table(Predicho = CLASE.KNN, Real = test$DEFPAY)
addmargins(tabla2)

library(yardstick)
autoplot(conf_mat(tabla2), type = "heatmap")  
summary(conf_mat(tabla2), event_level = "second") 

# Calcular el accuracy
accuracy_KNN <- mean(test$DEFPAY == CLASE.KNN); accuracy_KNN

# Calcular el error de mala clasificación
error <- mean(test$DEFPAY != CLASE.KNN) ; error

# Matriz de Confusión usando el paquete caret
library(caret)
cm_KNN <- caret::confusionMatrix(CLASE.KNN,
                                 test$DEFPAY,
                                 positive = "Si_paga")
cm_KNN

cm_KNN$table

cm_KNN$byClass["Sensitivity"] 
cm_KNN$byClass["Specificity"] 
cm_KNN$overall["Accuracy"]
cm_KNN$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.KNN, test$DEFPAY, plotROC = TRUE) -> auc_KNN
abline(0, 1, col = "red")
auc_KNN

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.KNN, real) -> logloss_KNN
logloss_KNN


test  <- bake(objeto_recipe, new_data = data.test)

train <- as.data.frame(train)
test  <- as.data.frame(test)

train <- train %>% select(1:9, 11:66, 10)
test <- test %>% select(1:9, 11:66, 10)

# Dividiendo usando el punto de corte óptimo con AUC
library(pROC)
roc <- roc(test[, target], PROBA.KNN)

umbral_knn <- pROC::coords(roc, "best")$threshold
umbral_knn

test$CLASE.KNN2 <- as.factor(ifelse(PROBA.KNN > umbral_knn,
                                    "Si_paga", 
                                    "No_paga"))

cm_KNN2 <- caret::confusionMatrix(test$CLASE.KNN2, 
                                  test[, target],
                                  positive = "Si_paga")

# Calcular el accuracy
accuracy_KNN2 <- mean(test$DEFPAY == test$CLASE.KNN2) ; accuracy_KNN2

cm_KNN2$byClass["Sensitivity"] 
cm_KNN2$byClass["Specificity"] 
cm_KNN2$overall["Accuracy"]
cm_KNN2$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.KNN, test$DEFPAY, plotROC = TRUE) -> auc_KNN2
abline(0, 1, col = "red")
auc_KNN2

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.KNN, real) -> logloss_KNN2
logloss_KNN2


# 3.3 Naive Bayes ----------------------
set.seed(2021)
modelo_nb <- train(DEFPAY ~ ., 
                   data = smote_train,
                   method = "nb", 
                   trControl = ctrl,
                   tuneGrid = expand.grid(fL=0,usekernel=T,
                                          adjust=c(0.1,0.5,1)),
                   metric = "Accuracy" )

modelo_nb 

modelo_nb$bestTune

#  fL usekernel adjust
#1  0      TRUE    0.1

plot(modelo_nb)

varImp(modelo_nb)

plot(varImp(modelo_nb))

PROBA.NB <- predict(modelo_nb, newdata = test, 
                    type = "prob")
head(PROBA.NB)
PROBA.NB <- PROBA.NB[, 2]

CLASE.NB <- predict(modelo_nb, newdata = test)  # 0.5 umbral
head(CLASE.NB)

head(cbind(test, PROBA.NB, CLASE.NB))

# Evaluando la performance del modelo Naive Bayes
# Matriz de Confusión
tabla3  <- table(Predicho = CLASE.NB, Real = test$DEFPAY)
addmargins(tabla3)

library(yardstick)
autoplot(conf_mat(tabla3), type = "heatmap")  
summary(conf_mat(tabla3), event_level = "second") 

# Calcular el accuracy
accuracy_nb <- mean(test$DEFPAY == CLASE.NB) ; accuracy_nb

# Calcular el error de mala clasificación
error <- mean(test$DEFPAY != CLASE.NB) ; error

# Matriz de Confusión usando paquete caret
library(caret)
cm_nb <- caret::confusionMatrix(CLASE.NB,
                                test$DEFPAY,
                                positive = "Si_paga")
cm_nb

cm_nb$table

cm_nb$byClass["Sensitivity"] 
cm_nb$byClass["Specificity"] 
cm_nb$overall["Accuracy"]
cm_nb$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.NB, test$DEFPAY, plotROC = TRUE) -> auc_nb
abline(0, 1, col = "red")
auc_nb

# Calcular el LogLoss
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.NB, real) -> logloss_nb
logloss_nb


test  <- bake(objeto_recipe, new_data = data.test)

train <- as.data.frame(train)
test  <- as.data.frame(test)

train <- train %>% select(1:9, 11:66, 10)
test <- test %>% select(1:9, 11:66, 10)

# Dividiendo usando el punto de corte óptimo con AUC
library(pROC)
roc <- roc(test[, target], PROBA.NB)

umbral_nb <- pROC::coords(roc, "best")$threshold
umbral_nb

test$CLASE.NB2 <- as.factor(ifelse(PROBA.NB > umbral_nb,
                                   "Si_paga", 
                                   "No_paga"))

cm_nb2 <- caret::confusionMatrix(test$CLASE.NB2, 
                                 test[, target],
                                 positive = "Si_paga")

# Calcular el accuracy
accuracy_nb2 <- mean(test$DEFPAY == test$CLASE.NB2) ; accuracy_nb2

cm_nb2$byClass["Sensitivity"] 
cm_nb2$byClass["Specificity"] 
cm_nb2$overall["Accuracy"]
cm_nb2$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.NB, test$DEFPAY, plotROC = TRUE) -> auc_nb2
abline(0, 1, col = "red")
auc_nb2

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.NB, real) -> logloss_nb2
logloss_nb2


# 3.4 SVM ----------------------
set.seed(2021)
modelo_svm <- train(DEFPAY ~ ., 
                    data = smote_train, 
                    method = "svmRadial", 
                    trControl = ctrl, 
                    #tunelength = 10,
                    tuneGrid = expand.grid(C=seq(0,2,length=10),
                                           sigma=0.5509),
                    metric = "Accuracy")

#The final values used for the model were sigma = 0.5509 and C = 1.

modelo_svm

modelo_svm$bestTune
#> modelo_svm$bestTune
#     sigma C
#10 0.05539 2

plot(modelo_svm)

varImp(modelo_svm)
plot(varImp(modelo_svm))

PROBA.SVM <- predict(modelo_svm, test, type = "prob")
head(PROBA.SVM)
PROBA.SVM <- PROBA.SVM[, 2]

CLASE.SVM <- predict(modelo_svm, test) # 0.5
head(CLASE.SVM)

# Evaluando la performance del SVM Radial
# Matriz de Confusión
tabla4  <- table(Predicho = CLASE.SVM, Real = test$DEFPAY)
addmargins(tabla4)

library(yardstick)
autoplot(conf_mat(tabla4), type = "heatmap")  
summary(conf_mat(tabla4), event_level = "second") 

# Calcular el accuracy
accuracy_svm <- mean(test$DEFPAY == CLASE.SVM); accuracy_svm

# Calcular el error de mala clasificación
error <- mean(test$DEFPAY != CLASE.SVM) ; error

# Matriz de Confusión usando el paquete caret
library(caret)
cm_svm <- caret::confusionMatrix(CLASE.SVM,
                                 test$DEFPAY,
                                 positive = "Si_paga")
cm_svm

cm_svm$table

cm_svm$byClass["Sensitivity"] 
cm_svm$byClass["Specificity"] 
cm_svm$overall["Accuracy"]
cm_svm$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.SVM, test$DEFPAY, plotROC = TRUE) -> auc_svm
abline(0, 1, col = "red")
auc_svm

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.SVM, real) -> logloss_svm
logloss_svm


test  <- bake(objeto_recipe, new_data = data.test)

train <- as.data.frame(train)
test  <- as.data.frame(test)

train <- train %>% select(1:9, 11:66, 10)
test <- test %>% select(1:9, 11:66, 10)

# Dividiendo usando el punto de corte óptimo con AUC
library(pROC)
roc <- roc(test[, target], PROBA.SVM)

umbral_svm <- pROC::coords(roc, "best")$threshold
umbral_svm

test$CLASE.SVM2 <- as.factor(ifelse(PROBA.SVM > umbral_svm,
                                    "Si_paga", 
                                    "No_paga"))

cm_svm2 <- caret::confusionMatrix(test$CLASE.SVM2, 
                                  test[, target],
                                  positive = "Si_paga")

# Calcular el accuracy
accuracy_svm2 <- mean(test$DEFPAY == test$CLASE.SVM2) ; accuracy_svm2

cm_svm2$byClass["Sensitivity"] 
cm_svm2$byClass["Specificity"] 
cm_svm2$overall["Accuracy"]
cm_svm2$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.SVM, test$DEFPAY, plotROC = TRUE) -> auc_svm2
abline(0, 1, col = "red")
auc_svm2

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.SVM, real) -> logloss_svm2
logloss_svm2


# 3.5 ARBOL C5.0 CON CARET ----------------------

set.seed(2021)
modelo_C5    <- train(DEFPAY ~ ., 
                      data = smote_train, 
                      method = "C5.0", 
                      trControl = ctrl, 
                      tuneLength = 5,
                      metric = "Accuracy")

modelo_C5

modelo_C5$bestTune
#> modelo_C5$bestTune
#   trials model winnow
#5      40 tree  FALSE

plot(modelo_C5)
varImp(modelo_C5)

plot(varImp(modelo_C5))

# Predicción de la clase y probabilidad 
PROBA.C5 <- predict(modelo_C5, newdata = test, type = "prob")
head(PROBA.C5)
PROBA.C5 <- PROBA.C5[, 2]

CLASE.C5 <- predict(modelo_C5, newdata = test )
head(CLASE.C5)

head(cbind(test, PROBA.C5, CLASE.C5))

# Evaluando la performance del modelo C5.0
# Matriz de Confusión
tabla5  <- table(Predicho = CLASE.C5, Real = test$DEFPAY)
addmargins(tabla5)

library(yardstick)
autoplot(conf_mat(tabla5), type = "heatmap")  
summary(conf_mat(tabla5), event_level = "second") 

# Calcular el accuracy
accuracy_c5 <- mean(test$DEFPAY == CLASE.C5) ; accuracy_c5

# Calcular el error de mala clasificación
error <- mean(test$DEFPAY != CLASE.C5) ; error

cm_c5 <- caret::confusionMatrix(CLASE.C5,
                                test$DEFPAY,
                                positive = "Si_paga")

cm_c5
cm_c5$byClass["Sensitivity"] 
cm_c5$byClass["Specificity"] 
cm_c5$overall["Accuracy"]
cm_c5$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.C5, test$DEFPAY, plotROC = TRUE) -> auc_c5
abline(0, 1, col = "red")
auc_c5

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.C5, real) -> logloss_c5
logloss_c5


test  <- bake(objeto_recipe, new_data = data.test)

train <- as.data.frame(train)
test  <- as.data.frame(test)

train <- train %>% select(1:9, 11:66, 10)
test <- test %>% select(1:9, 11:66, 10)

# Dividiendo usando el punto de corte óptimo con AUC
library(pROC)
roc <- roc(test[, target], PROBA.C5)

umbral_c5 <- pROC::coords(roc, "best")$threshold
umbral_c5

test$CLASE.C52 <- as.factor(ifelse(PROBA.C5 > umbral_c5,
                                   "Si_paga", 
                                   "No_paga"))

cm_c52 <- caret::confusionMatrix(test$CLASE.C52, 
                                 test[, target],
                                 positive = "Si_paga")

# Calcular el accuracy
accuracy_c52 <- mean(test$DEFPAY == test$CLASE.C52) ; accuracy_c52

cm_c52$byClass["Sensitivity"] 
cm_c52$byClass["Specificity"] 
cm_c52$overall["Accuracy"]
cm_c52$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.C5, test$DEFPAY, plotROC = TRUE) -> auc_c52
abline(0, 1, col = "red")
auc_c52

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(PROBA.C5, real) -> logloss_c52
logloss_c52


# Grafico de densidad de los modelos ----

# Se decide visualizar el grafico de densidad de los modelos para usar los que tengan los 3
# mejores graficos de densidad en el modelo stacking
modelos  <- list(GLM        = modelo_glm,
                 KNN        = modelo_knn,
                 NB        = modelo_nb,
                 SVM        = modelo_svm,
                 C5         = modelo_C5)

comparacion_modelos <- resamples(modelos)
summary(comparacion_modelos)

dotplot(comparacion_modelos)

bwplot(comparacion_modelos)

densityplot(comparacion_modelos, 
            metric = "Accuracy",
            auto.key = TRUE)

library(psych)
corPlot(modelCor(comparacion_modelos), cex = 1.2, main = "Matriz de correlación")

# Se selecciona C5, KNN, y SVM

#3.6 Ensamble y stacking ----

# Ensamble
fitControl   <- trainControl(method = "cv",
                             number = 10,
                             savePredictions = 'final',
                             #summaryFunction = twoClassSummary,
                             classProbs = T)

modelos = list(
  C5.0  =  caretModelSpec(method = "C5.0",
                          tuneLength = 5),
  knn       =  caretModelSpec(method = "knn", 
                              tuneGrid = data.frame(k = 1)),
  svmRadial      =  caretModelSpec(method = "svmRadial",
                                   tuneGrid = data.frame(C = 2,sigma = 0.5509))
)


library(caretEnsemble)
set.seed(2021) 
modelo_ensamble <- caretList(DEFPAY ~ ., 
                             data = smote_train,
                             tuneList = modelos,
                             metric = "Accuracy",
                             # methodList = algorithmList,
                             trControl = fitControl)

# Vista global de los resultados de todos los modelos
modelo_ensamble

# Stacking de modelos usando CART (rpart)
stackControl <- trainControl(method = "cv",
                             number = 10,
                             savePredictions = 'final', # To save out of fold predictions for best parameter combinantions
                             classProbs = T             # To save the class probabilities of the out of fold predictions
)

set.seed(2021)
stack.cart <-  caretStack(modelo_ensamble,
                          method = "rpart",      
                          tuneLength = 10,
                          metric = "Accuracy",  
                          trControl = stackControl)

plot(stack.cart)

summary(stack.cart)

test$proba_stack  <- predict(object = stack.cart,
                             test[, predictores],
                             type = 'prob')

head(test$proba_stack)
test$proba_stack <- 1 - test$proba_stack

test$clase_stack  <- predict(object = stack.cart,
                             test[, predictores])

head(test$clase_stack)

cm_stacking       <- caret::confusionMatrix(test$clase_stack,
                                            test[, target],
                                            positive = "Si_paga")

# Calcular el accuracy
accuracy_stacking <- mean(test$DEFPAY == test$clase_stack) ; accuracy_stacking

cm_stacking$byClass["Sensitivity"] 
cm_stacking$byClass["Specificity"] 
cm_stacking$overall["Accuracy"]
cm_stacking$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(test$proba_stack, test$DEFPAY, plotROC = TRUE) -> auc_stacking
abline(0, 1, col = "red")
auc_stacking

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(test$proba_stack, real) -> logloss_stacking
logloss_stacking

# Dividiendo usando el punto de corte óptimo con AUC
library(pROC)
roc <- roc(test[, target], test$proba_stack)

umbral_stacking <- pROC::coords(roc, "best")$threshold
umbral_stacking

test$clase_stack2 <- as.factor(ifelse(test$proba_stack > umbral_stacking,
                                      "Si_paga", 
                                      "No_paga"))

cm_stacking2 <- caret::confusionMatrix(test$clase_stack2, 
                                       test[, target],
                                       positive = "Si_paga")

# Calcular el accuracy
accuracy_stacking2 <- mean(test$DEFPAY == test$clase_stack2) ; accuracy_stacking2

cm_stacking2$byClass["Sensitivity"] 
cm_stacking2$byClass["Specificity"] 
cm_stacking2$overall["Accuracy"]
cm_stacking2$byClass["Balanced Accuracy"]

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(test$proba_stack, test$DEFPAY, plotROC = TRUE) -> auc_stacking2
abline(0, 1, col = "red")
auc_stacking2

# Log-Loss
library(MLmetrics)
real <- as.numeric(test$DEFPAY)
real <- ifelse(real == 2, 1, 0)
LogLoss(test$proba_stack, real) -> logloss_stacking2
logloss_stacking2

# Cuadro Comparativo de los modelos ----
algoritmos       <- c("Reg. Lineal",
                      "Reg. Lineal con corte optimo",
                      "KNN",
                      "KNN con corte optimo",
                      "Naive Bayes",
                      "Naive Bayes con corte optimo",
                      "SVM",
                      "SVM con corte optimo",
                      "Arbol C5.0",
                      "Arbol C5.0 con corte optimo",
                      "Stacking con Cart",
                      "Stacking con Cart con umbral optimo")

sensibilidad  <- c(cm_GLM$byClass["Sensitivity"],
                   cm_GLM2$byClass["Sensitivity"],
                   cm_KNN$byClass["Sensitivity"],
                   cm_KNN2$byClass["Sensitivity"],
                   cm_nb$byClass["Sensitivity"],
                   cm_nb2$byClass["Sensitivity"],
                   cm_svm$byClass["Sensitivity"],
                   cm_svm2$byClass["Sensitivity"],
                   cm_c5$byClass["Sensitivity"],
                   cm_c5$byClass["Sensitivity"],
                   cm_stacking$byClass["Sensitivity"],
                   cm_stacking2$byClass["Sensitivity"])

especificidad <- c(cm_GLM$byClass["Specificity"],
                   cm_GLM2$byClass["Specificity"],
                   cm_KNN$byClass["Specificity"],
                   cm_KNN2$byClass["Specificity"],
                   cm_nb$byClass["Specificity"],
                   cm_nb2$byClass["Specificity"],
                   cm_svm$byClass["Specificity"],
                   cm_svm2$byClass["Specificity"],
                   cm_c5$byClass["Specificity"],
                   cm_c52$byClass["Specificity"],
                   cm_stacking$byClass["Specificity"],
                   cm_stacking2$byClass["Specificity"])

accuracy_bal <- c( cm_GLM$byClass["Balanced Accuracy"],
                   cm_GLM2$byClass["Balanced Accuracy"],
                   cm_KNN$byClass["Balanced Accuracy"],
                   cm_KNN2$byClass["Balanced Accuracy"],
                   cm_nb$byClass["Balanced Accuracy"],
                   cm_nb2$byClass["Balanced Accuracy"],
                   cm_svm$byClass["Balanced Accuracy"],
                   cm_svm2$byClass["Balanced Accuracy"],
                   cm_c5$byClass["Balanced Accuracy"],
                   cm_c52$byClass["Balanced Accuracy"],
                   cm_stacking$byClass["Balanced Accuracy"],
                   cm_stacking2$byClass["Balanced Accuracy"])

accuracy      <- c(accuracy_GLM,
                   accuracy_GLM2,
                   accuracy_KNN,
                   accuracy_KNN2,
                   accuracy_nb,
                   accuracy_nb2,
                   accuracy_svm,
                   accuracy_svm2,
                   accuracy_c5,
                   accuracy_c52,
                   accuracy_stacking,
                   accuracy_stacking2)

area_roc      <- c(auc_GLM,
                   auc_GLM2,
                   auc_KNN,
                   auc_KNN2,
                   auc_nb,
                   auc_nb2,
                   auc_svm,
                   auc_svm2,
                   auc_c5,
                   auc_c52,
                   auc_stacking,
                   auc_stacking2)

comparacion <- data.frame(algoritmos,
                          sensibilidad, 
                          especificidad,
                          accuracy_bal,
                          accuracy,
                          area_roc)

comparacion

#                            algoritmos sensibilidad especificidad accuracy_bal accuracy area_roc
#1                          Reg. Lineal      0.54673        0.8308       0.6888   0.7680   0.7508
#2         Reg. Lineal con corte optimo      0.57337        0.8098       0.6916   0.7575   0.7508
#3                                  KNN      0.51407        0.7345       0.6243   0.6857   0.6261
#4                 KNN con corte optimo      0.52010        0.7326       0.6264   0.6856   0.6261
#5                          Naive Bayes      0.02714        0.9963       0.5117   0.7820   0.7241
#6         Naive Bayes con corte optimo      0.65477        0.6933       0.6740   0.6847   0.7241
#7                                  SVM      0.41759        0.9255       0.6716   0.8132   0.7245
#8                 SVM con corte optimo      0.55628        0.8305       0.6934   0.7699   0.7245
#9                           Arbol C5.0      0.43568        0.9155       0.6756   0.8094   0.7670
#10         Arbol C5.0 con corte optimo      0.43568        0.8137       0.7053   0.7658   0.7670
#11                   Stacking con Cart      0.48342        0.8712       0.6773   0.7854   0.6918
#12 Stacking con Cart con umbral optimo      0.50402        0.8667       0.6854   0.7865   0.6918