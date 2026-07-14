import os
import json
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_utils import construir_fila_nueva, preprocesar, COLUMNAS_NUMERICAS
from train import entrenar as ejecutar_entrenamiento

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="API Prediccion de Precios de Casas en Quito")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELO_PATH   = os.path.join(DIR_ACTUAL, "models", "modelo_regresion.pkl")
METRICAS_PATH = os.path.join(DIR_ACTUAL, "models", "metricas.json")

class CasaNueva(BaseModel):
    area_m2: float
    pisos: int
    habitaciones: int
    banos: int
    antiguedad: int
    garaje: int
    sector: str

class ParametrosEntrenamiento(BaseModel):
    test_size: float = 0.2
    normalizar: bool = True
    random_state: int = 42

@app.get("/")
def raiz():
    return {"mensaje": "API de Prediccion de Precios activa"}

@app.get("/metricas")
def obtener_metricas():
    if not os.path.exists(METRICAS_PATH):
        return {"error": "Modelo no entrenado. Llama primero a /entrenar"}
    with open(METRICAS_PATH, "r") as f:
        return json.load(f)

@app.post("/entrenar")
def entrenar_modelo(params: ParametrosEntrenamiento):
    try:
        metricas = ejecutar_entrenamiento(
            test_size=params.test_size,
            normalizar=params.normalizar,
            random_state=params.random_state
        )
        return {"mensaje": "Entrenamiento exitoso", "metricas": metricas}
    except Exception as e:
        return {"error": f"Fallo al entrenar: {str(e)}"}

@app.post("/predecir")
def predecir(casa: CasaNueva):
    if not os.path.exists(MODELO_PATH):
        return {"error": "Modelo no encontrado. Ejecuta /entrenar primero"}

    with open(MODELO_PATH, "rb") as f:
        paquete = pickle.load(f)

    modelo     = paquete["modelo"]
    scaler     = paquete["scaler"]
    normalizar = paquete["normalizar"]

    fila    = construir_fila_nueva(casa.area_m2, casa.pisos, casa.habitaciones,
                                    casa.banos, casa.antiguedad, casa.garaje, casa.sector)
    X_nuevo = preprocesar(fila, entrenando=False)

    if normalizar and scaler:
        X_nuevo[COLUMNAS_NUMERICAS] = scaler.transform(X_nuevo[COLUMNAS_NUMERICAS])

    precio = float(modelo.predict(X_nuevo)[0])
    return {
        "precio_estimado_usd": round(precio, 2),
        "precio_formato": f"${precio:,.2f}",
        "datos_ingresados": casa.dict()
    }
