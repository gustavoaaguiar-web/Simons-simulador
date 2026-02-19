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
st.set_page_config(page_title="Simons GG v11.2", page_icon="🦅", layout="wide")

try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

CAPITAL_INICIAL = 30000000.0
ARCHIVO_ESTADO = "simons_state.json"

# --- PERSISTENCIA MEJORADA ---
def cargar_estado_persistente():
    if os.path.exists(ARCHIVO_ESTADO):
        with open(ARCHIVO_ESTADO, "r") as f:
            return json.load(f)
    return {"saldo": 33362112.69, "pos": {}, "historial_patrimonio": []}

def guardar_estado_persistente():
    estado = {
        "saldo": st.session_state.saldo,
        "pos": st.session_state.pos,
        "historial_patrimonio": st.session_state.historial_patrimonio
    }
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump(estado, f)

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado_persistente())

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

# --- MONITOR DE MERCADO ---
activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=60)
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

# --- CORRECCIÓN: CÁLCULO DINÁMICO DE VALORES ---
valor_actual_cedears = 0.0
detalle_posiciones = []

if not df_m.empty:
    for t, info in st.session_state.pos.items():
        try:
            # Obtener precio actual de la tabla de mercado
            precio_hoy = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0]
            # Valorización
            cant_nominal = info['m'] / info['p']
            valor_mercado_activo = cant_nominal * precio_hoy
            valor_actual_cedears += valor_mercado_activo
            
            # Datos para el sidebar
            ganancia_pct = ((precio_hoy / info['p']) - 1) * 100
            ganancia_ars = valor_mercado_activo - info['m']
            detalle_posiciones.append({
                "t": t, "inv": info['m'], "gan_a": ganancia_ars, 
                "gan_p": ganancia_pct, "p_e": info['p'], "p_h": precio_hoy
            })
        except:
            # Si no hay datos hoy, mantenemos el valor de compra para no mostrar 0
            valor_actual_cedears += info['m']
            detalle_posiciones.append({
                "t": t, "inv": info['m'], "gan_a": 0, "gan_p": 0, "p_e": info['p'], "p_h": info['p']
            })

patrimonio_total = st.session_state.saldo + valor_actual_cedears
rendimiento_total = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

# --- LÓGICA DE TRADING (Aviso Único) ---
if ccl_m is not None and not df_m.empty:
    for _, row in df_m.iterrows():
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        if desvio < -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS'], 'ccl': row['CCL']}
                enviar_alerta_mail(f"🦅 COMPRA: {activo}", f"Bot compró {activo}\nARS: {row['ARS']}\nDesvío: {desvio*100:.2f}%")
                guardar_estado_persistente()
                st.rerun()

        elif desvio > 0.005 and activo in st.session_state.pos:
            precio_v = row['ARS']
            info_c = st.session_state.pos[activo]
            monto_recuperado = (info_c['m'] / info_c['p']) * precio_v
            st.session_state.saldo += monto_recuperado
            ganancia_neta = monto_recuperado - info_c['m']
            del st.session_state.pos[activo]
            enviar_alerta_mail(f"🦅 VENTA: {activo}", f"Vendido {activo}\nGanancia: AR$ {ganancia_neta:,.2f}")
            guardar_estado_persistente()
            st.rerun()

# --- INTERFAZ ---
st.title("🦅 Simons GG v11.2 🤑")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
col_b.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
col_c.metric("Valor Mercado Cedears", f"AR$ {valor_actual_cedears:,.2f}")

st.divider()
if ccl_m:
    st.header(f"CCL ${ccl_m:,.2f}")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if ((r['CCL']/ccl_m)-1) < -0.005 and r['Clima']=="🟢" else ("🔴 VENTA" if ((r['CCL']/ccl_m)-1) > 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_m[['Activo', 'Señal', 'Clima', 'CCL', 'ARS']], use_container_width=True, hide_index=True)

# SIDEBAR ACTUALIZADO
st.sidebar.header("📂 Cartera y Ganancias")
for d in detalle_posiciones:
    color = "green" if d['gan_a'] >= 0 else "red"
    st.sidebar.markdown(f"### {d['t']}")
    st.sidebar.write(f"Inversión: AR$ {d['inv']:,.2f}")
    st.sidebar.markdown(f"**Ganancia: :{color}[AR$ {d['gan_a']:,.2f} ({d['gan_p']:+.2f}%)]**")
    st.sidebar.caption(f"Entrada: ${d['p_e']:.2f} | Actual: ${d['p_h']:.2f}")
    st.sidebar.divider()
