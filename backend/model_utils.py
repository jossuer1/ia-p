import pandas as pd

COLUMNAS_NUMERICAS = ["area_m2", "pisos", "habitaciones", "banos", "antiguedad", "garaje"]
COLUMNA_CATEGORICA = "sector"
COLUMNA_OBJETIVO   = "precio"

SECTORES_VALIDOS = [
    "Cumbaya", "La Carolina", "Gonzalez Suarez", "Tumbaco", "Centro Norte",
    "Quitumbe", "Calderon", "Carapungo", "Conocoto", "Centro Historico",
]

def preprocesar(df, entrenando=True):
    df = df.copy()
    df[COLUMNA_CATEGORICA] = df[COLUMNA_CATEGORICA].apply(
        lambda s: s if s in SECTORES_VALIDOS else "Otro"
    )
    dummies = pd.DataFrame(0, index=df.index,
                           columns=[f"sector_{s}" for s in SECTORES_VALIDOS])
    for s in SECTORES_VALIDOS:
        dummies[f"sector_{s}"] = (df[COLUMNA_CATEGORICA] == s).astype(int)
    X = pd.concat([df[COLUMNAS_NUMERICAS].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    if entrenando:
        return X, df[COLUMNA_OBJETIVO].reset_index(drop=True)
    return X

def construir_fila_nueva(area_m2, pisos, habitaciones, banos, antiguedad, garaje, sector):
    return pd.DataFrame([{
        "area_m2": area_m2, "pisos": pisos, "habitaciones": habitaciones,
        "banos": banos, "antiguedad": antiguedad, "garaje": garaje, "sector": sector,
    }])
