# 📊 Modelo Predictivo de Cumplimiento de Pago de Tarjetas de Crédito

## 📌 Contexto
Este proyecto desarrolla un modelo predictivo para determinar la probabilidad de incumplimiento de pago de tarjetas de crédito de clientes de un banco en Taiwán, utilizando técnicas de Minería de Datos y Machine Learning. El objetivo es apoyar la gestión del riesgo crediticio y la toma de decisiones del negocio.

---

## 🧠 Enfoque Analítico
El proyecto se desarrolló siguiendo la metodología CRISP-DM, cubriendo todo el ciclo analítico:
- Entendimiento del negocio y de los datos  
- Preparación y preprocesamiento del dataset  
- Modelado, evaluación y selección del mejor modelo  
- Propuesta de uso en un entorno real  

Se trabajó con un dataset desbalanceado, aplicando técnicas específicas para mejorar la calidad predictiva.

---

## 🛠️ Herramientas y Técnicas
- Lenguaje: R  
- Preprocesamiento:
  - Imputación de valores faltantes (KNN)  
  - Selección de variables (Boruta)  
  - Estandarización y creación de variables dummy  
  - Balanceo de clases con SMOTE  
- Modelos evaluados:
  - Regresión Logística  
  - K-Nearest Neighbors (KNN)  
  - Naive Bayes  
  - Support Vector Machine (SVM)  
  - Árboles de decisión C5.0  
  - Stacking (C5.0 + SVM + KNN con CART)  
- Evaluación:
  - Sensibilidad  
  - Accuracy Balanceado  

---

## 📊 Resultados
- Se compararon múltiples modelos con y sin umbral óptimo.  
- El mejor desempeño se obtuvo con:
  - Regresión Logística con umbral óptimo  
  - Accuracy balanceado ≈ 69.2%  
- El modelo logra un equilibrio adecuado entre sensibilidad y precisión, evitando underfitting y overfitting.

---

## 🚀 Aplicación al Negocio
El modelo permite:
- Identificar clientes con mayor riesgo de incumplimiento.  
- Apoyar estrategias de prevención temprana, ajustes de condiciones de pago o personalización de productos.  
- Mejorar la gestión del riesgo y la experiencia del cliente.

---

## 📚 Aprendizajes Clave
- Aplicación práctica de Machine Learning en riesgo crediticio.  
- Importancia del balanceo de datos y la selección de métricas adecuadas.  
- Comparación crítica de modelos y trade-offs entre métricas.  
- Uso de metodologías estructuradas para proyectos de Data Science.
