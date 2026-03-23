import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------------
# CONFIGURACION
# -----------------------------------
st.set_page_config(
    page_title="AI Predictor de Acciones",
    page_icon="📈",
    layout="centered"
)

st.title("📈 AI Predictor de Acciones")
st.markdown("### 🔮 Simulación del mercado + predicción inteligente")

st.divider()

# -----------------------------------
# EXPLICACION GENERAL
# -----------------------------------
st.info("""
Esta aplicación simula el comportamiento de una acción durante el último año 
y predice su precio basado en variables clave del mercado.

📌 Ajusta las variables para ver cómo cambian las predicciones.
""")

# -----------------------------------
# GENERAR DATOS
# -----------------------------------
@st.cache_data
def generar_datos():
    np.random.seed(42)
    
    dias = 365
    fechas = pd.date_range(end=pd.Timestamp.today(), periods=dias)

    precio = np.cumsum(np.random.normal(0.5, 2, dias)) + 100

    volumen = np.random.randint(1000, 10000, dias)
    sentimiento = np.random.uniform(-1, 1, dias)
    interes = np.random.randint(50, 100, dias)

    df = pd.DataFrame({
        "fecha": fechas,
        "precio": precio,
        "volumen": volumen,
        "sentimiento": sentimiento,
        "interes_busqueda": interes
    })

    return df

df = generar_datos()

# -----------------------------------
# GRÁFICA
# -----------------------------------
st.subheader("📅 Comportamiento de la acción (último año)")
st.line_chart(df.set_index("fecha")["precio"])

# -----------------------------------
# MODELO
# -----------------------------------
def entrenar_modelo(df):
    X = df[["volumen", "sentimiento", "interes_busqueda"]]
    y = df["precio"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    return modelo

modelo = entrenar_modelo(df)

# -----------------------------------
# INPUTS + CONTEXTO
# -----------------------------------
st.subheader("🔧 Variables del mercado")

# Volumen
volumen = st.slider(
    "📊 Volumen de transacciones",
    1000, 10000, 5000,
    help="Cantidad de acciones negociadas. Un volumen alto indica mayor actividad del mercado."
)

# Sentimiento
sentimiento = st.slider(
    "💬 Sentimiento del mercado",
    -1.0, 1.0, 0.0,
    help="""
-1 → Muy negativo (miedo, malas noticias)
0 → Neutral
+1 → Muy positivo (optimismo, hype)
"""
)

# Interés
interes = st.slider(
    "🔎 Interés de búsqueda",
    50, 100, 70,
    help="Nivel de interés del público (simula Google Trends). Más interés = más atención del mercado."
)

# -----------------------------------
# EXPLICACIÓN DINÁMICA
# -----------------------------------
st.markdown("### 🧠 Interpretación actual")

if sentimiento > 0.5:
    st.success("El mercado muestra un fuerte optimismo 📈")
elif sentimiento < -0.5:
    st.error("El mercado muestra señales negativas 📉")
else:
    st.warning("El mercado está en equilibrio ⚖️")

if volumen > 8000:
    st.write("🔥 Alto volumen → Posible movimiento fuerte")
elif volumen < 3000:
    st.write("🐢 Bajo volumen → Mercado lento")

# -----------------------------------
# PREDICCIÓN
# -----------------------------------
if st.button("Predecir precio 🚀"):

    entrada = np.array([[volumen, sentimiento, interes]])
    prediccion = modelo.predict(entrada)[0]

    st.subheader("📊 Resultado de la predicción")

    st.metric("💰 Precio estimado", f"${prediccion:.2f}")

    # Indicador visual
    if sentimiento > 0.5:
        st.success("📈 Tendencia alcista esperada")
    elif sentimiento < -0.5:
        st.error("📉 Posible caída del precio")
    else:
        st.warning("⚖️ Comportamiento estable")

st.divider()

st.caption("⚠️ Simulación educativa - No usar para decisiones financieras reales")