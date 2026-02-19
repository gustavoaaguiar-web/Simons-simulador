import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
import smtplib
import json
import os
from email.message import EmailMessage

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v11.3", page_icon="🦅", layout="wide")

# Tickers completos (14 activos)
activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

# --- PERSISTENCIA ---
ARCHIVO_ESTADO = "simons_state.json"
def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        with open(ARCHIVO_ESTADO, "r") as f: return json.load(f)
    return {"saldo": 33362112.69, "pos": {}, "historial": []}

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

# --- LÓGICA DE MERCADO (MEJORADA) ---
@st.cache_data(ttl=60)
def fetch_full_market():
    datos, ccls = [], []
    for t, r in activos_dict.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            if h_usd.empty or h_ars.empty: raise ValueError()
            
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a})
        except:
            # Si falla, agregamos fila vacía para que no "desaparezca" de la lista
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0})
    
    df = pd.DataFrame(datos)
    ccl_m = np.nanmedian(ccls) if ccls else None
    return df, ccl_m

df_m, ccl_m = fetch_full_market()

# --- CÁLCULOS DE CARTERA ---
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    precio_actual = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0]
    if precio_actual == 0: precio_actual = info['p'] # Respaldo
    valor_cedears += (info['m'] / info['p']) * precio_actual

patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento = ((patrimonio_total / 30000000.0) - 1) * 100

# --- INTERFAZ ---
st.title("🦅 Simons GG v11.3")
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento:+.2f}%")
c2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Invertido", f"AR$ {valor_cedears:,.2f}")

st.divider()

if ccl_m:
    st.header(f"CCL ${ccl_m:,.2f}")
    
    # Generar Señales
    def get_signal(r):
        if np.isnan(r['CCL']): return "⌛ S/D"
        desvio = (r['CCL'] / ccl_m) - 1
        if desvio < -0.005 and r['Clima'] == "🟢": return "🟢 COMPRA"
        if desvio > 0.005: return "🔴 VENTA"
        return "⚖️ MANTENER"

    df_m['Señal'] = df_m.apply(get_signal, axis=1)
    
    # Mostrar TODA la tabla (height=500 asegura que no se corten filas)
    st.dataframe(df_m, use_container_width=True, hide_index=True, height=530)
else:
    st.warning("No se pudieron obtener datos del mercado. Reintentando...")

# SIDEBAR: DETALLE POR ACTIVO
st.sidebar.header("📂 Cartera Detallada")
for t, info in st.session_state.pos.items():
    p_actual = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0]
    gan_p = ((p_actual / info['p']) - 1) * 100 if p_actual > 0 else 0
    st.sidebar.write(f"**{t}**: {gan_p:+.2f}%")
