import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Simons GG v10.4", page_icon="🦅", layout="wide")

# Datos manuales para no depender del Excel
CAPITAL_INICIAL = 30000000.0
SALDO_ACTUAL = 33362112.69 

# --- INTERFAZ PRINCIPAL ---
st.title("🦅 Simons GG v10.4 🤑")
st.subheader("Simulador de Inversiones - Algoritmo Simons")

# Métricas principales
rendimiento = ((SALDO_ACTUAL / CAPITAL_INICIAL) - 1) * 100
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {SALDO_ACTUAL:,.2f}", f"{rendimiento:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {SALDO_ACTUAL:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {(SALDO_ACTUAL * 0.08):,.2f}")

# --- MONITOR DE MERCADO ---
st.divider()
st.write("### 📊 Monitor de Arbitraje (Precios en tiempo real)")

activos = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=300)
def fetch_market():
    datos, ccls = [], []
    for t, r in activos.items():
        try:
            # Descarga datos
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u = float(h_usd.Close.iloc[-1])
            p_a = float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            # Modelo HMM (Clima de mercado)
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima_idx = model.predict(ret)[-1]
            clima = "🟢" if clima_idx == 0 else "🔴"
            
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "USD": p_u, "ARS": p_a})
        except: continue
    
    df = pd.DataFrame(datos)
    ccl_m = np.median(ccls) if ccls else 0
    return df, ccl_m

df_res, ccl_m = fetch_market()

if not df_res.empty:
    st.caption(f"CCL Mediano calculado: ${ccl_m:.2f}")
    
    def procesar(row):
        desvio = (row['CCL'] / ccl_m) - 1
        row['Desvío %'] = f"{desvio*100:+.2f}%"
        if desvio < -0.0065 and row['Clima'] == "🟢": row['Señal'] = "🟢 COMPRA"
        elif desvio > 0.0065: row['Señal'] = "🔴 VENTA"
        else: row['Señal'] = "⚖️ MANTENER"
        return row

    df_final = df_res.apply(procesar, axis=1)
    
    # Estilo de la tabla
    def color_señal(val):
        if 'COMPRA' in str(val): return 'background-color: #004d00; color: white'
        if 'VENTA' in str(val): return 'background-color: #4d0000; color: white'
        return ''

    st.dataframe(
        df_final[['Activo', 'CCL', 'Clima', 'Señal', 'Desvío %', 'ARS', 'USD']]
        .style.applymap(color_señal, subset=['Señal']), 
        use_container_width=True, hide_index=True
    )
else:
    st.warning("Cargando datos financieros...")
