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
CAPITAL_INICIAL = 30000000.0
SALDO_ACTUAL = 33362112.69 # Tu saldo actual manual

st.set_page_config(page_title="Simons GG v10.4", page_icon="🦅", layout="wide")

# --- LÓGICA DE TIEMPO ---
mercado_abierto = time(11, 0) <= ahora <= time(17, 0)

# --- INTERFAZ ---
st.title("🦅 Simons GG v10.4 🤑")

if mercado_abierto:
    st.success(f"🟢 MERCADO ABIERTO - Hora Arg: {ahora.strftime('%H:%M')}")
else:
    st.info(f"⚪ MERCADO CERRADO - Hora Arg: {ahora.strftime('%H:%M')}")

# Métricas
rendimiento_h = ((SALDO_ACTUAL / CAPITAL_INICIAL) - 1) * 100
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {SALDO_ACTUAL:,.2f}", f"{rendimiento_h:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {SALDO_ACTUAL:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {(SALDO_ACTUAL * 0.08):,.2f}")

# --- MONITOR DE MERCADO ---
st.subheader("📊 Monitor de Arbitraje")

# Los 14 activos con sus ratios
activos = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=120)
def fetch_market():
    datos, ccls = [], []
    for t, r in activos.items():
        try:
            # AJUSTE DE TICKERS LOCALES (YPF y PAM)
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            # Algoritmo Simons (HMM)
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            # 0 suele ser el régimen de baja volatilidad (verde)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            
            datos.append({
                "Activo": t, "CCL": ccl, "Clima": clima,
                "USD": p_u, "ARS": p_a
            })
        except: continue
    
    df = pd.DataFrame(datos)
    if not df.empty:
        ccl_m = np.median(ccls)
        
        def asignar_señal(row):
            desvio = (row['CCL'] / ccl_m) - 1
            row['Desvío %'] = f"{desvio*100:+.2f}%"
            if desvio < -0.0065 and row['Clima'] == "🟢": return "🟢 COMPRA"
            if desvio > 0.0065: return "🔴 VENTA"
            return "⚖️ MANTENER"
        
        df['Señal'] = df.apply(asignar_señal, axis=1)
        return df[['Activo', 'CCL', 'Clima', 'Señal', 'Desvío %', 'ARS', 'USD']], ccl_m
    return pd.DataFrame(), 0

df_m, ccl_m = fetch_market()

if not df_m.empty:
    st.caption(f"CCL Mediano: ${ccl_m:.2f}")

    def color_señal(val):
        if 'COMPRA' in str(val): color = '#004d00'
        elif 'VENTA' in str(val): color = '#4d0000'
        else: return ''
        return f'background-color: {color}; color: white; font-weight: bold'

    st.dataframe(df_m.style.applymap(color_señal, subset=['Señal']), use_container_width=True, hide_index=True)
else:
    st.warning("Aguardando datos del mercado...")

# --- NOTIFICACIONES (Simulado) ---
if any("COMPRA" in str(s) for s in df_m['Señal']):
    st.sidebar.info("📩 Señales detectadas. (Listo para enviar al mail cuando configures SMTP)")
