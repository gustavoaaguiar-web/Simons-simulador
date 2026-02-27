import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, time
import smtplib
import json
import os
from email.message import EmailMessage
import pytz
import time as time_lib

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v14.2", page_icon="🦅", layout="wide")
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

# 1. ZONA HORARIA
try:
    arg_tz = pytz.timezone('America/Argentina/Buenos_Aires')
    ahora_arg_dt = datetime.now(arg_tz)
except:
    ahora_arg_dt = datetime.now()

ahora_arg_time = ahora_arg_dt.time()
dia_semana = ahora_arg_dt.weekday()

# 2. PERSISTENCIA
ARCHIVO_ESTADO = "simons_state.json"
SALDO_OBJETIVO = 34130883.81

def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, "r") as f:
                data = json.load(f)
                if not data.get("pos"): data["saldo"] = SALDO_OBJETIVO
                return data
        except: pass
    return {"saldo": SALDO_OBJETIVO, "pos": {}, "notificados": []}

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

def guardar_estado():
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump({k: st.session_state[k] for k in ["saldo", "pos", "notificados"]}, f)

# 3. CAPTURA DE MERCADO (Con Triple Reintento para TSLA y otros)
activos_dict = {'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 'AMZN':144, 'META':24, 'VIST':3, 'PAM':25}

@st.cache_data(ttl=290)
def fetch_market_ultra_safe():
    datos, ccls = [], []
    for t, r in activos_dict.items():
        exito = False
        for intento in range(3): # Reintenta hasta 3 veces por activo
            try:
                tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
                h_usd = yf.download(t, period="3mo", interval="1d", progress=False, timeout=15)
                h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False, timeout=15)
                
                if not h_usd.empty and not h_ars.empty:
                    p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
                    ccl_i = (p_a * r) / p_u
                    ccls.append(ccl_i)
                    ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
                    clima = "🟢" if GaussianHMM(n_components=3, random_state=42).fit(ret).predict(ret)[-1] == 0 else "🔴"
                    datos.append({"Activo": t, "CCL": ccl_i, "Clima": clima, "ARS": p_a, "USD": p_u})
                    exito = True
                    break # Sale del bucle de reintentos si funcionó
            except:
                time_lib.sleep(1) # Espera un segundo antes de reintentar
        
        if not exito:
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD": 0})
            
    df = pd.DataFrame(datos)
    mediana = np.nanmedian(ccls) if ccls else None
    return df, mediana

df_m, ccl_m = fetch_market_ultra_safe()

# 4. TRADING Y CÁLCULOS
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    p_act = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    precio = p_act[0] if len(p_act) > 0 and p_act[0] > 0 else info['p']
    valor_cedears += (info['m'] / info['p']) * precio

patrimonio = st.session_state.saldo + valor_cedears
rendimiento = ((patrimonio / 30000000.0) - 1) * 100

# 5. UI PRINCIPAL
st.title("🦅 Simons GG v14.2 🤑")
status = "🟢 OPERANDO" if (0 <= dia_semana <= 4 and time(11,0) <= ahora_arg_time < time(16,50)) else "🔴 CERRADO"
st.caption(f"Actualizado: {ahora_arg_dt.strftime('%H:%M:%S')} | {status}")

m1, m2, m3 = st.columns(3)
m1.metric("Patrimonio Total", f"AR$ {patrimonio:,.2f}", f"{rendimiento:+.2f}%")
m2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
m3.metric("En CEDEARs", f"AR$ {valor_cedears:,.2f}")

st.divider()

if not df_m.empty and ccl_m:
    df_vis = df_m.copy()
    df_vis['% Desv'] = df_vis['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if pd.notnull(x) else "ERR")
    df_vis['Señal'] = df_vis.apply(lambda r: "🟢 COMPRA" if pd.notnull(r['CCL']) and ((r['CCL']/ccl_m)-1) <= -0.005 and r['Clima'] == "🟢" else ("🔴 VENTA" if pd.notnull(r['CCL']) and ((r['CCL']/ccl_m)-1) >= 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_vis[['Activo', '% Desv', 'Clima', 'Señal', 'ARS']], use_container_width=True, hide_index=True)

# 6. SIDEBAR
with st.sidebar:
    st.header("📂 Cartera")
    if st.session_state.pos:
        for t, info in list(st.session_state.pos.items()):
            with st.expander(f"📦 {t}", expanded=True):
                if st.button(f"Vender {t}", key=f"v_{t}"):
                    p_v = df_m.loc[df_m['Activo']==t, 'ARS'].values[0] if t in df_m['Activo'].values else info['p']
                    st.session_state.saldo += (info['m']/info['p']) * p_v
                    del st.session_state.pos[t]
                    guardar_estado()
                    st.rerun()
    else:
        st.info("Cartera vacía.")
