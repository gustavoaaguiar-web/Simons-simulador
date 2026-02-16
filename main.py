import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime, time, timedelta

# --- CONFIGURACIÓN DE ZONA HORARIA ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()
ahora = ahora_dt.time()

# --- CONFIGURACIÓN GENERAL ---
# Cargamos tus datos actuales manualmente para no depender del Excel
CAPITAL_INICIAL = 30000000.0
SALDO_ACTUAL = 33362112.69 

st.set_page_config(page_title="Simons GG v10.4", page_icon="🦅", layout="wide")

# --- LÓGICA DE TIEMPO ---
mercado_abierto = time(11, 0) <= ahora <= time(17, 0)

# --- ESTADO DE LA APP ---
if 'saldo' not in st.session_state:
    st.session_state.update({'saldo': SALDO_ACTUAL, 'pos': {}, 'hist': []})

# --- INTERFAZ ---
st.title("🦅 Simons GG v10.4 🤑")

if mercado_abierto:
    st.success(f"🟢 MERCADO ABIERTO - Hora Arg: {ahora.strftime('%H:%M')}")
else:
    st.info(f"⚪ MERCADO CERRADO - Hora Arg: {ahora.strftime('%H:%M')}")

patrimonio_total = st.session_state.saldo
rendimiento_h = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_h:+.2f}% vs Inicial")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {(patrimonio_total * 0.08):,.2f}")

# --- MONITOR DE MERCADO ---
st.subheader("📊 Monitor de Arbitraje (Yahoo Finance)")

activos = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=300)
def fetch_market():
    datos = []
    ccls = []
    for t, r in activos.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima_idx = model.predict(ret)[-1]
            clima = "🟢" if clima_idx == 0 else "🔴"
            
            datos.append({
                "Activo": t, "CCL": ccl, "Clima": clima,
                "USD": p_u, "ARS": p_a
            })
        except: continue
    
    df = pd.DataFrame(datos)
    ccl_m = np.median(ccls) if ccls else 0
    return df, ccl_m

df_res, ccl_m = fetch_market()

if not df_res.empty:
    st.caption(f"CCL Mediano: ${ccl_m:.2f}")
    
    def procesar_df(row):
        desvio = (row['CCL'] / ccl_m) - 1
        row['Desvío %'] = f"{desvio*100:+.2f}%"
        if desvio < -0.0065 and row['Clima'] == "🟢": row['Señal'] = "🟢 COMPRA"
        elif desvio > 0.0065: row['Señal'] = "🔴 VENTA"
        else: row['Señal'] = "⚖️ MANTENER"
        return row

    df_final = df_res.apply(procesar_df, axis=1)
    
    def color_señal(val):
        if 'COMPRA' in str(val): return 'background-color: #004d00; color: white'
        if 'VENTA' in str(val): return 'background-color: #4d0000; color: white'
        return ''

    st.dataframe(df_final[['Activo', 'CCL', 'Clima', 'Señal', 'Desvío %', 'ARS', 'USD']].style.applymap(color_señal, subset=['Señal']), use_container_width=True, hide_index=True)
else:
    st.warning("Esperando datos de Yahoo Finance... (Mercado cerrado o cargando)")
