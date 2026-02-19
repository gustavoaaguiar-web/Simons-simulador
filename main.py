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
st.set_page_config(page_title="Simons GG v11.4", page_icon="🦅", layout="wide")

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

# --- LÓGICA DE MERCADO ---
@st.cache_data(ttl=60)
def fetch_full_market():
    datos, ccls = [], []
    for t, r in activos_dict.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a, "USD_Ticker": p_u})
        except:
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD_Ticker": 0})
    
    df = pd.DataFrame(datos)
    ccl_m = np.nanmedian(ccls) if ccls else None
    return df, ccl_m

df_m, ccl_m = fetch_full_market()

# --- PROCESAMIENTO DE DATOS ---
if ccl_m:
    # 1. Calcular Diferencia % (Desvío respecto a la mediana)
    df_m['%'] = df_m['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if not np.isnan(x) else "S/D")
    
    # 2. Definir Señal
    def get_signal(r):
        if np.isnan(r['CCL']): return "⌛ S/D"
        desvio = (r['CCL'] / ccl_m) - 1
        if desvio < -0.005 and r['Clima'] == "🟢": return "🟢 COMPRA"
        if desvio > 0.005: return "🔴 VENTA"
        return "⚖️ MANTENER"
    df_m['Señal'] = df_m.apply(get_signal, axis=1)

    # 3. Renombrar y Ordenar Columnas según pedido
    # Activo | % | Clima | Señal | ARS | USD
    df_m = df_m.rename(columns={"USD_Ticker": "USD", "CCL": "Valor USD"})
    df_final = df_m[['Activo', '%', 'Clima', 'Señal', 'ARS', 'USD']]

# --- CÁLCULO PATRIMONIO ---
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    precio_hoy = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0]
    if precio_hoy == 0: precio_hoy = info['p']
    valor_cedears += (info['m'] / info['p']) * precio_hoy

patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento_total = ((patrimonio_total / 30000000.0) - 1) * 100

# --- INTERFAZ PRINCIPAL ---
st.title("🦅 Simons GG v11.4 🤑")
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("En Cedears", f"AR$ {valor_cedears:,.2f}")

st.divider()
if ccl_m:
    st.header(f"CCL ${ccl_m:,.2f}")
    st.dataframe(df_final, use_container_width=True, hide_index=True, height=530)

# --- PANEL LATERAL RECARGADO ---
st.sidebar.header("📂 Cartera Detallada")
if st.session_state.pos:
    for t, info in st.session_state.pos.items():
        # Obtener precio actual para calcular ganancia
        p_actual = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0]
        if p_actual == 0: p_actual = info['p']
        
        cant_nom = info['m'] / info['p']
        valor_hoy = cant_nom * p_actual
        ganancia_ars = valor_hoy - info['m']
        ganancia_pct = ((p_actual / info['p']) - 1) * 100
        
        color = "green" if ganancia_ars >= 0 else "red"
        
        with st.sidebar.expander(f"🦅 {t}", expanded=True):
            st.write(f"**Ganancia:** :{color}[AR$ {ganancia_ars:,.2f} ({ganancia_pct:+.2f}%)]")
            st.write(f"**Precio Compra:** ${info['p']:,.2f}")
            st.write(f"**Precio Actual:** ${p_actual:,.2f}")
            st.caption(f"Inversión inicial: AR$ {info['m']:,.2f}")
else:
    st.sidebar.info("No hay activos en cartera.")
