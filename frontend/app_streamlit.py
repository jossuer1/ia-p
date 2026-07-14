import streamlit as st
import requests
import os

# En produccion, busca la variable de entorno de Streamlit Cloud. Si no existe, usa localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Prediccion Casas Quito", page_icon="🏠", layout="wide")
st.title("🏠 Prediccion de Precios de Casas en Quito")
st.markdown("**Modelo:** Regresion Lineal Multiple | **Datos:** Quito, Ecuador")
st.divider()


def verificar_estado_modelo():
    """Consulta /metricas para saber si el modelo ya esta entrenado."""
    try:
        r = requests.get(f"{API_URL}/metricas", timeout=10)
        data = r.json()
        if "error" in data:
            return False, None
        return True, data
    except Exception:
        return None, None  # None = no se pudo ni conectar con la API


modelo_ok, metricas_actuales = verificar_estado_modelo()

seccion = st.sidebar.radio(
    "Navegacion",
    ["Entrenar modelo", "Predecir precio", "Metricas del modelo"]
)

st.sidebar.info(f"API conectada en:\n`{API_URL}`")

if modelo_ok is True:
    st.sidebar.success("Modelo entrenado y listo ✅")
elif modelo_ok is False:
    st.sidebar.warning("Modelo aun no entrenado ⚠️\nVe a 'Entrenar modelo'")
else:
    st.sidebar.error("No se pudo conectar con la API ❌")

SECTORES = ["Cumbaya", "La Carolina", "Gonzalez Suarez", "Tumbaco", "Centro Norte",
            "Quitumbe", "Calderon", "Carapungo", "Conocoto", "Centro Historico"]

# ---------------------------------------------------------------------------
# SECCION 1: ENTRENAR MODELO
# ---------------------------------------------------------------------------
if seccion == "Entrenar modelo":
    st.header("Entrenar el modelo")
    st.markdown(
        "Antes de predecir precios, el modelo debe entrenarse en el servidor. "
        "En el plan gratuito de Render el disco no es permanente, asi que si el "
        "servicio se reinicia por inactividad, tendras que entrenar de nuevo aqui."
    )

    if modelo_ok is True:
        st.success("Ya existe un modelo entrenado en el servidor. Puedes re-entrenar si quieres ajustar los parametros.")
        with st.expander("Ver metricas actuales"):
            st.json(metricas_actuales)
    elif modelo_ok is False:
        st.warning("Todavia no hay un modelo entrenado. Usa el boton de abajo para generarlo.")
    else:
        st.error("No se pudo conectar con la API. Verifica que el backend en Render este activo.")

    st.divider()
    st.subheader("Parametros de entrenamiento")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Proporcion de datos de prueba", 0.1, 0.4, 0.2, 0.05)
    with col2:
        normalizar = st.selectbox("Normalizar datos numericos", [True, False])
    with col3:
        random_state = st.number_input("random_state (semilla)", 0, 999, 42)

    if st.button("Entrenar modelo ahora", type="primary", use_container_width=True):
        payload = {"test_size": test_size, "normalizar": normalizar, "random_state": int(random_state)}
        with st.spinner("Entrenando modelo en el servidor... puede tardar unos segundos"):
            try:
                resp = requests.post(f"{API_URL}/entrenar", json=payload, timeout=60)
                r = resp.json()
                if "metricas" in r:
                    st.success(r.get("mensaje", "Entrenamiento exitoso"))
                    m = r["metricas"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R2", f"{m['R2']:.4f}")
                    c2.metric("MAE", f"${m['MAE']:,.0f}")
                    c3.metric("RMSE", f"${m['RMSE']:,.0f}")
                    c4.metric("MSE", f"{m['MSE']:,.0f}")
                    st.info("Ya puedes ir a la seccion 'Predecir precio' para hacer simulaciones.")
                else:
                    st.error(f"Error: {r}")
            except Exception as e:
                st.error(f"No se pudo conectar con la API: {e}")

# ---------------------------------------------------------------------------
# SECCION 2: PREDECIR PRECIO
# ---------------------------------------------------------------------------
elif seccion == "Predecir precio":
    st.header("Predecir el precio de una casa")

    if modelo_ok is False:
        st.warning("El modelo aun no esta entrenado. Ve a la seccion **'Entrenar modelo'** primero.")
    elif modelo_ok is None:
        st.error("No se pudo conectar con la API. Verifica que el backend en Render este activo.")

    col1, col2 = st.columns(2)
    with col1:
        area_m2 = st.number_input("Area (m2)", 30, 500, 120)
        pisos = st.selectbox("Numero de pisos", [1, 2, 3])
        habitaciones = st.selectbox("Habitaciones", [1, 2, 3, 4, 5])
        banos = st.selectbox("Baños", [1, 2, 3, 4])
    with col2:
        antiguedad = st.slider("Antiguedad (años)", 0, 45, 5)
        garaje_txt = st.selectbox("Tiene garaje?", ["Si", "No"])
        sector = st.selectbox("Sector", SECTORES)
    st.divider()

    if st.button("Calcular precio estimado", type="primary", use_container_width=True,
                  disabled=(modelo_ok is not True)):
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

# ---------------------------------------------------------------------------
# SECCION 3: METRICAS DEL MODELO
# ---------------------------------------------------------------------------
elif seccion == "Metricas del modelo":
    st.header("Metricas del modelo entrenado")

    if modelo_ok is False:
        st.warning("El modelo aun no esta entrenado. Ve a la seccion **'Entrenar modelo'** primero.")
    elif modelo_ok is None:
        st.error("No se pudo conectar con la API. Verifica que el backend en Render este activo.")
    else:
        m = metricas_actuales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R2", f"{m['R2']:.4f}")
        c2.metric("MAE", f"${m['MAE']:,.0f}")
        c3.metric("RMSE", f"${m['RMSE']:,.0f}")
        c4.metric("MSE", f"{m['MSE']:,.0f}")
        st.divider()
        st.subheader("Interpretacion")
        st.markdown(
          f"- **R2 = {m['R2']:.2%}** : el modelo explica el {m['R2']:.0%} de la variabilidad del precio.\n"
          f"- **MAE = \\${m['MAE']:,.0f}** : en promedio el precio predicho se aleja +/-\\${m['MAE']:,.0f} del real.\n"
          f"- **RMSE = \\${m['RMSE']:,.0f}** : similar al MAE pero penaliza mas los errores grandes."
        )
        st.subheader("Parametros de entrenamiento")
        st.json(m.get("parametros", {}))
                                                                                                                                                                              