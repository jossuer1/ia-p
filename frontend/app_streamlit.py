import streamlit as st
import requests
import json
import os

# En producción, busca la variable de entorno de Streamlit Cloud. Si no existe, usa localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Prediccion Casas Quito", page_icon="🏠", layout="wide")
st.title("🏠 Prediccion de Precios de Casas en Quito")
st.markdown("**Modelo:** Regresion Lineal Multiple | **Datos:** Quito, Ecuador")
st.divider()

seccion = st.sidebar.radio(
    "Navegacion",
    ["Predecir precio", "Metricas del modelo"]
)
st.sidebar.info(f"API conectada en:\n`{API_URL}`")

SECTORES = ["Cumbaya","La Carolina","Gonzalez Suarez","Tumbaco","Centro Norte",
            "Quitumbe","Calderon","Carapungo","Conocoto","Centro Historico"]

if seccion == "Predecir precio":
    st.header("Predecir el precio de una casa")
    col1, col2 = st.columns(2)
    with col1:
        area_m2      = st.number_input("Area (m2)", 30, 500, 120)
        pisos        = st.selectbox("Numero de pisos", [1, 2, 3])
        habitaciones = st.selectbox("Habitaciones", [1, 2, 3, 4, 5])
        banos        = st.selectbox("Baños", [1, 2, 3, 4])
    with col2:
        antiguedad = st.slider("Antiguedad (años)", 0, 45, 5)
        garaje_txt = st.selectbox("Tiene garaje?", ["Si", "No"])
        sector     = st.selectbox("Sector", SECTORES)
    st.divider()
    
    if st.button("Calcular precio estimado", type="primary", use_container_width=True):
        payload = {"area_m2": area_m2, "pisos": pisos,
                   "habitaciones": habitaciones, "banos": banos,
                   "antiguedad": antiguedad,
                   "garaje": 1 if garaje_txt == "Si" else 0,
                   "sector": sector}
        try:
            resp = requests.post(f"{API_URL}/predecir", json=payload, timeout=10)
            r = resp.json()
            if "precio_estimado_usd" in r:
                st.success(f"## Precio estimado: **{r['precio_formato']}**")
                st.info(f"Sector: {sector} | Area: {area_m2} m2 | "
                        f"Pisos: {pisos} | Hab: {habitaciones} | "
                        f"Banos: {banos} | Antiguedad: {antiguedad} años | "
                        f"Garaje: {garaje_txt}")
            else:
                st.error(f"Error: {r}")
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")

elif seccion == "Metricas del modelo":
    st.header("Metricas del modelo entrenado")
    try:
        m = requests.get(f"{API_URL}/metricas", timeout=10).json()
        if "error" in m:
            st.warning(m["error"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R2",   f"{m['R2']:.4f}")
            c2.metric("MAE",  f"${m['MAE']:,.0f}")
            c3.metric("RMSE", f"${m['RMSE']:,.0f}")
            c4.metric("MSE",  f"{m['MSE']:,.0f}")
            st.divider()
            st.subheader("Interpretacion")
            st.markdown(
                f"- **R2 = {m['R2']:.2%}** : el modelo explica el {m['R2']:.0%} de la variabilidad del precio.\n"
                f"- **MAE = ${m['MAE']:,.0f}** : en promedio el precio predicho se aleja +/-${m['MAE']:,.0f} del real.\n"
                f"- **RMSE = ${m['RMSE']:,.0f}** : similar al MAE pero penaliza mas los errores grandes."
            )
            st.subheader("Parametros de entrenamiento")
            st.json(m.get("parametros", {}))
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")