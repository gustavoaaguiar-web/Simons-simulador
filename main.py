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

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v11.8", page_icon="🦅", layout="wide")

# Auto-Refresh cada 5 minutos
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

# --- CREDENCIALES ---
try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

# --- PERSISTENCIA CORREGIDA (Eliminado error de 'historial') ---
ARCHIVO_ESTADO = "simons_state.json"

def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        with open(ARCHIVO_ESTADO, "r") as f: 
            data = json.load(f)
            # Aseguramos que existan las llaves básicas para evitar el AttributeError
            if "saldo" not in data: data["saldo"] = 33362112.69
            if "pos" not in data: data["pos"] = {}
            return data
    return {"saldo": 33362112.69, "pos": {}}

def guardar_estado():
    # Solo guardamos lo que realmente existe en session_state
    estado = {
        "saldo": st.session_state.saldo,
        "pos": st.session_state.pos
    }
    with open(ARCHIVO_ESTADO, "w") as f: 
        json.dump(estado, f)

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

def enviar_alerta_mail(asunto, cuerpo):
    msg = EmailMessage()
    msg.set_content(cuerpo)
    msg['Subject'] = asunto
    msg['From'] = MI_MAIL
    msg['To'] = MI_MAIL
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(MI_MAIL, CLAVE_APP)
        server.send_message(msg)
        server.quit()
    except: pass

# --- CAPTURA DE MERCADO ---
@st.cache_data(ttl=290)
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
            
            # Método Markov (HMM)
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

# --- LÓGICA DE EJECUCIÓN (Lógica de desvío 0.50%) ---
if ccl_m is not None:
    for _, row in df_m.iterrows():
        if np.isnan(row['CCL']): continue
        
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']

        # COMPRA: Desvío < -0.50% (-0.005) y Clima Verde
        if desvio <= -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            patrimonio_ref = st.session_state.saldo + sum(v['m'] for v in st.session_state.pos.values())
            monto_t = patrimonio_ref * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS'], 'ccl': row['CCL']}
                enviar_alerta_mail(f"🦅 COMPRA: {activo}", f"Comprado {activo} con desvío de {desvio*100:.2f}%")
                guardar_estado()
                st.rerun()

        # VENTA: Desvío > +0.50% (+0.005)
        elif desvio >= 0.005 and activo in st.session_state.pos:
            info_c = st.session_state.pos[activo]
            monto_final = (info_c['m'] / info_c['p']) * row['ARS']
            st.session_state.saldo += monto_final
            del st.session_state.pos[activo]
            enviar_alerta_mail(f"🦅 VENTA: {activo}", f"Vendido {activo} con desvío de {desvio*100:.2f}%")
            guardar_estado()
            st.rerun()

# --- INTERFAZ ---
# ... (Calculos de patrimonio total y tabla principal idénticos a v11.7)
# ... (Sidebar con Precio Compra, Precio Actual, Ganancia ARS y % corregido)
