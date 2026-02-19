import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import smtplib
import json
import os
from email.message import EmailMessage

# --- CONFIGURACIÓN BÁSICA ---
st.set_page_config(page_title="Simons GG v11.9", page_icon="🦅", layout="wide")

# Refresco automático cada 5 minutos
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

# --- ESTADO INICIAL & PERSISTENCIA ---
ARCHIVO_ESTADO = "simons_state.json"

def inicializar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, "r") as f:
                data = json.load(f)
                # Validamos que existan las claves necesarias
                if "saldo" not in data: data["saldo"] = 33362112.69
                if "pos" not in data: data["pos"] = {}
                return data
        except:
            pass
    return {"saldo": 33362112.69, "pos": {}}

if 'saldo' not in st.session_state:
    state = inicializar_estado()
    st.session_state.saldo = state["saldo"]
    st.session_state.pos = state["pos"]

def guardar_cambios():
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump({"saldo": st.session_state.saldo, "pos": st.session_state.pos}, f)

# --- LÓGICA DE MERCADO ---
activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=290)
def fetch_market():
    datos, ccls = [], []
    for t, r in activos_dict.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u = float(h_usd.Close.iloc[-1])
            p_a = float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            # Markov para Clima
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a, "USD": p_u})
        except:
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD": 0})
    
    df = pd.DataFrame(datos)
    mediana = np.nanmedian(ccls) if ccls else None
    return df, mediana

df_m, ccl_m = fetch_market()

# --- CÁLCULO DE VALORES ---
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    p_actual_fila = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    precio = p_actual_fila[0] if len(p_actual_fila) > 0 and p_actual_fila[0] > 0 else info['p']
    valor_cedears += (info['m'] / info['p']) * precio

patrimonio_total = st.session_state.saldo + valor_cedears

# --- LÓGICA DE TRADING (0.5%) ---
if ccl_m:
    for _, row in df_m.iterrows():
        if np.isnan(row['CCL']): continue
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        # COMPRA: Desvío <= -0.5% y Clima Verde
        if desvio <= -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS']}
                guardar_cambios()
                st.rerun()
        
        # VENTA: Desvío >= 0.5%
        elif desvio >= 0.005 and activo in st.session_state.pos:
            info_c = st.session_state.pos[activo]
            st.session_state.saldo += (info_c['m'] / info_c['p']) * row['ARS']
            del st.session_state.pos[activo]
            guardar_cambios()
            st.rerun()

# --- INTERFAZ ---
st.title("🦅 Simons GG v11.9")

# Métricas Principales
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}")
c2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Cedears", f"AR$ {valor_cedears:,.2f}")

st.divider()

# Tabla Principal con el Orden Solicitado
if ccl_m:
    st.subheader(f"Mediana CCL: ${ccl_m:,.2f}")
    df_m['%'] = df_m['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if not np.isnan(x) else "S/D")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if not np.isnan(r['CCL']) and ((r['CCL']/ccl_m)-1) <= -0.005 and r['Clima'] == "🟢" else ("🔴 VENTA" if not np.isnan(r['CCL']) and ((r['CCL']/ccl_m)-1) >= 0.005 else "⚖️ MANTENER"), axis=1)
    
    # Orden: Activo | % | Clima | Señal | ARS | USD
    st.dataframe(df_m[['Activo', '%', 'Clima', 'Señal', 'ARS', 'USD']], use_container_width=True, hide_index=True, height=530)

# Sidebar: Cartera Detallada
st.sidebar.header("📂 Cartera")
for t, info in st.session_state.pos.items():
    p_actual_arr = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    p_act = p_actual_arr[0] if len(p_actual_arr) > 0 and p_actual_arr[0] > 0 else info['p']
    gan_p = ((p_act / info['p']) - 1) * 100
    st.sidebar.write(f"**{t}**: {gan_p:+.2f}%")
    st.sidebar.caption(f"Compra: ${info['p']} | Actual: ${p_act}")
