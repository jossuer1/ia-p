import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from model_utils import preprocesar, COLUMNAS_NUMERICAS

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))

def entrenar(test_size=0.2, normalizar=True, random_state=42):
    csv_path = os.path.join(DIR_ACTUAL, "data", "dataset_casas_quito.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontro el dataset en: {csv_path}")

    df = pd.read_csv(csv_path)
    X, y = preprocesar(df, entrenando=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if normalizar:
        scaler = StandardScaler()
        X_train[COLUMNAS_NUMERICAS] = scaler.fit_transform(X_train[COLUMNAS_NUMERICAS])
        X_test[COLUMNAS_NUMERICAS]  = scaler.transform(X_test[COLUMNAS_NUMERICAS])

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    models_dir = os.path.join(DIR_ACTUAL, "models")
    os.makedirs(models_dir, exist_ok=True)

    paquete = {"modelo": modelo, "scaler": scaler, "normalizar": normalizar}
    with open(os.path.join(models_dir, "modelo_regresion.pkl"), "wb") as f:
        pickle.dump(paquete, f)

    metricas = {
        "R2": float(r2), "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse),
        "parametros": {"test_size": test_size, "normalizar": normalizar, "random_state": random_state}
    }
    with open(os.path.join(models_dir, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=4)

    return metricas

if __name__ == "__main__":
    print(entrenar())
