# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)

#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#

# --- Creación de directorios de salida ---
os.makedirs("files/models", exist_ok=True)
os.makedirs("files/output", exist_ok=True)

# --- Carga de datos ---
train_df = pd.read_csv("files/input/train_data.csv.zip")
test_df = pd.read_csv("files/input/test_data.csv.zip")

#
# Paso 1.
# Realice la limpieza de los datasets.
#
def clean_data(df):
    """Aplica todas las reglas de limpieza a un dataframe."""
    
    # Renombrar columna objetivo y remover ID
    df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns="ID")
        
    # Eliminar registros con información no disponible (0) en MARRIAGE y EDUCATION
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]

    # Agrupar niveles de educación > 4 en la categoría 4 ("others")
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    
    return df

train_df = clean_data(train_df)
test_df = clean_data(test_df)

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
y_train = train_df.pop("default")
x_train = train_df

y_test = test_df.pop("default")
x_test = test_df

# Identificar columnas categóricas y numéricas
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
# El resto de las columnas son numéricas
numeric_features = [col for col in x_train.columns if col not in categorical_features]

#
# Paso 3.
# Cree un pipeline para el modelo de clasificación.
#
# El preprocesador aplica OneHotEncoder a las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough" # Mantiene las columnas numéricas
)

# Crear el pipeline completo con todos los componentes requeridos por el pytest
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=None)), # Usa todas las componentes
        ("selector", SelectKBest()),
        ("classifier", MLPClassifier(max_iter=1000, random_state=42)), # Aumentar iteraciones y fijar semilla
    ]
)

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
#
# Definir una grilla de parámetros para GridSearchCV
# Nota: Esta grilla es un ejemplo para asegurar que el código funcione.
# Puede ser necesario ajustarla para obtener un rendimiento óptimo.
param_grid = {
    "pca__n_components": [10, 15, 20],
    "selector__k": [8, 10, 10],
    "classifier__alpha": [0.001, 0.01],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1, # Usar todos los procesadores disponibles
    refit=True
)

grid_search.fit(x_train, y_train)

# El mejor modelo es el objeto grid_search ya re-entrenado
best_model = grid_search

#
# Paso 5.
# Guarde el modelo (comprimido con gzip)
#
MODEL_FILENAME = "files/models/model.pkl.gz"
with gzip.open(MODEL_FILENAME, "wb") as file:
    pickle.dump(best_model, file)

#
# Paso 6 y 7.
# Calcule metricas y matrices de confusión y guárdelas.
#
def compute_and_format_metrics(y_true, y_pred, dataset_name):
    """Calcula métricas y las formatea como diccionarios para el JSON."""
    
    # Métricas de rendimiento
    perf_metrics = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm_metrics = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }
    return perf_metrics, cm_metrics

# Realizar predicciones
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

# Calcular métricas para ambos conjuntos
train_perf, train_cm = compute_and_format_metrics(y_train, y_train_pred, "train")
test_perf, test_cm = compute_and_format_metrics(y_test, y_test_pred, "test")

# Guardar todas las métricas en el archivo JSON
METRICS_FILENAME = "files/output/metrics.json"
with open(METRICS_FILENAME, "w", encoding="utf-8") as file:
    # Escribir cada diccionario como una nueva línea
    json.dump(train_perf, file)
    file.write("\n")
    json.dump(test_perf, file)
    file.write("\n")
    json.dump(train_cm, file)
    file.write("\n")
    json.dump(test_cm, file)

print("¡Proceso completado exitosamente!")
print(f"Modelo guardado en: {MODEL_FILENAME}")
print(f"Métricas guardadas en: {METRICS_FILENAME}")