import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("🦅 Simons GG - Test de Arranque")

# Prueba simple de datos
st.write("Probando conexión con Yahoo Finance...")
try:
    data = yf.download("AAPL", period="1d")
    st.success("¡Yahoo Finance funcionando!")
    st.write(data.tail())
except Exception as e:
    st.error(f"Error en Yahoo: {e}")

st.divider()
st.write("Estado de la App: **En línea**")
st.write("Saldo inicial cargado: **AR$ 33,362,112.69**")
