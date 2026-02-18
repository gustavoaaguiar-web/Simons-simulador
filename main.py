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

# --- CONFIGURACIÓN DE TIEMPO ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()

# --- CONFIGURACIÓN APP & SEGURIDAD ---
st.set_page_config(page_title="Simons GG v11.0", page_icon="🦅", layout="wide")

try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

CAPITAL_INICIAL = 30000000.0
ARCHIVO_ESTADO = "simons_state.json" # Persistencia local/nube

# --- FUNCIONES DE PERSISTENCIA (Para evitar avisos repetidos) ---
def cargar_estado_persistente():
    if os.path.exists(ARCHIVO_ESTADO):
        with open(ARCHIVO_ESTADO, "r") as f:
            return json.load(f)
    return {
        "saldo": 33362112.69,
        "pos": {},
        "historial_patrimonio": []
    }

def guardar_estado_persistente():
    estado = {
        "saldo": st.session_state.saldo,
        "pos": st.session_state.pos,
        "historial_patrimonio": st.session_state.historial_patrimonio
    }
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump(estado, f)

# Inicialización única
if 'saldo' not in st.session_state:
    data = cargar_estado_persistente()
    st.session_state.update(data)

# --- FUNCIONES CORE ---
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
        return True
    except: return False

# --- CÁLCULO DE MÉTRICAS ---
valor_cedears = sum(float(v['m']) for v in st.session_state.pos.values())
patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento_total = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

# --- INTERFAZ ---
st.title("🦅 Simons GG v11.0 🤑")

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Invertido en Cedears", f"AR$ {valor_cedears:,.2f}")

# --- MONITOR DE MERCADO ---
activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=120)
def fetch_mercado():
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
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a})
        except: continue
    df = pd.DataFrame(datos)
    ccl_m = np.median(ccls) if len(ccls) > 0 else None
    return df, ccl_m

df_m, ccl_m = fetch_mercado()

# --- LÓGICA DE TRADING (Aviso Único) ---
if ccl_m is not None and not df_m.empty:
    for _, row in df_m.iterrows():
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        # COMPRA: Solo si NO existe ya en la cartera guardada
        if desvio < -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS'], 'ccl': row['CCL']}
                enviar_alerta_mail(f"🦅 COMPRA: {activo}", f"Bot compró {activo}\nCCL: {row['CCL']:.2f}\nDesvío: {desvio*100:.2f}%")
                guardar_estado_persistente()
                st.rerun()

        # VENTA: Solo si existe en la cartera
        elif desvio > 0.005 and activo in st.session_state.pos:
            monto_v = st.session_state.pos[activo]['m']
            st.session_state.saldo += monto_v
            del st.session_state.pos[activo]
            enviar_alerta_mail(f"🦅 VENTA: {activo}", f"Bot vendió {activo}\nDesvío: {desvio*100:.2f}%")
            guardar_estado_persistente()
            st.rerun()

# --- VISUALIZACIÓN ---
st.divider()
if ccl_m:
    st.header(f"CCL ${ccl_m:,.2f}")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if ((r['CCL']/ccl_m)-1) < -0.005 and r['Clima']=="🟢" else ("🔴 VENTA" if ((r['CCL']/ccl_m)-1) > 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_m[['Activo', 'Señal', 'Clima', 'CCL', 'ARS']], use_container_width=True, hide_index=True)

# CARTERA EN SIDEBAR
st.sidebar.header("📂 Cartera Actual")
if st.session_state.pos:
    for t, info in st.session_state.pos.items():
        st.sidebar.markdown(f"**{t}**")
        st.sidebar.write(f"Invertido: AR$ {info['m']:,.2f}")
        st.sidebar.divider()
else:
    st.sidebar.info("Sin posiciones abiertas.")
