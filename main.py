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
st.set_page_config(page_title="Simons GG v11.5", page_icon="🦅", layout="wide")

activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

# --- PERSISTENCIA (Fundamental para que el despertador no repita mails) ---
ARCHIVO_ESTADO = "simons_state.json"
def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        with open(ARCHIVO_ESTADO, "r") as f: return json.load(f)
    return {"saldo": 33362112.69, "pos": {}, "historial": []}

def guardar_estado():
    estado = {"saldo": st.session_state.saldo, "pos": st.session_state.pos, "historial": st.session_state.historial}
    with open(ARCHIVO_ESTADO, "w") as f: json.dump(estado, f)

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

# --- LÓGICA DE MERCADO ---
@st.cache_data(ttl=120) # Cache de 2 min para el despertador
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
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a, "USD": p_u})
        except:
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD": 0})
    df = pd.DataFrame(datos)
    ccl_m = np.nanmedian(ccls) if ccls else None
    return df, ccl_m

df_m, ccl_m = fetch_full_market()

# --- LÓGICA DE ALERTAS AUTOMÁTICAS ---
if ccl_m and not df_m.empty:
    for _, row in df_m.iterrows():
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        # Lógica de Compra
        if desvio < -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = (st.session_state.saldo + sum(v['m'] for v in st.session_state.pos.values())) * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS'], 'ccl': row['CCL']}
                # Enviar Mail... (aquí va tu función de mail)
                guardar_estado()
                st.rerun()

# --- INTERFAZ (Tabla Ordenada) ---
if ccl_m:
    df_m['%'] = df_m['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if not np.isnan(x) else "S/D")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if ((r['CCL']/ccl_m)-1) < -0.005 and r['Clima'] == "🟢" else "⚖️ MANTENER", axis=1)
    df_final = df_m[['Activo', '%', 'Clima', 'Señal', 'ARS', 'USD']]
    
    st.title("🦅 Simons GG v11.5")
    st.metric("CCL Mercado", f"${ccl_m:,.2f}")
    st.dataframe(df_final, use_container_width=True, hide_index=True, height=530)

# PANEL LATERAL CON GANANCIAS (IGUAL A V11.4)
# ... (Se mantiene igual que la versión anterior para ver ganancias en pesos y %)
