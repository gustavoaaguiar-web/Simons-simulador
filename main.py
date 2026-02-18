import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# --- CONFIGURACIÓN DE TIEMPO ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()

# --- CONFIGURACIÓN APP & SEGURIDAD ---
st.set_page_config(page_title="Simons GG v10.9.2", page_icon="🦅", layout="wide")

# Credenciales (Secrets o Respaldo)
try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

CAPITAL_INICIAL = 30000000.0

# --- INICIALIZACIÓN INTERNA (Sin Excel) ---
if 'saldo' not in st.session_state:
    st.session_state.saldo = 33362112.69
if 'pos' not in st.session_state:
    st.session_state.pos = {}  # Formato: {'VIST': {'m': monto, 'p': precio, 'ccl': ccl}}
if 'historial_patrimonio' not in st.session_state:
    st.session_state.historial_patrimonio = []

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

# Guardar evolución en el historial interno
st.session_state.historial_patrimonio.append({"fecha": ahora_dt.strftime("%H:%M:%S"), "valor": patrimonio_total})

# --- INTERFAZ SUPERIOR ---
st.title("🦅 Simons GG v10.9.2 🤑")
st.subheader("Modo: Almacenamiento Local (Alta Velocidad)")

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

# --- LÓGICA DE TRADING AUTOMÁTICO ---
if ccl_m is not None and not df_m.empty:
    for _, row in df_m.iterrows():
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        # COMPRA AUTO
        if desvio < -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS'], 'ccl': row['CCL']}
                enviar_alerta_mail(f"🦅 COMPRA: {activo}", f"Bot compró {activo}\nCCL: {row['CCL']:.2f}\nDesvío: {desvio*100:.2f}%")
                st.rerun()

        # VENTA AUTO
        elif desvio > 0.005 and activo in st.session_state.pos:
            monto_v = st.session_state.pos[activo]['m']
            st.session_state.saldo += monto_v
            del st.session_state.pos[activo]
            enviar_alerta_mail(f"🦅 VENTA: {activo}", f"Bot vendió {activo}\nDesvío: {desvio*100:.2f}%")
            st.rerun()

# --- VISUALIZACIÓN ---
st.divider()
if ccl_m:
    st.write(f"### 🎯 CCL Mediano Mercado: ${ccl_m:.2f}")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if ((r['CCL']/ccl_m)-1) < -0.005 and r['Clima']=="🟢" else ("🔴 VENTA" if ((r['CCL']/ccl_m)-1) > 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_m[['Activo', 'Señal', 'Clima', 'CCL', 'ARS']], use_container_width=True, hide_index=True)

# CARTERA Y EVOLUCIÓN
st.sidebar.header("📂 Cartera en Tiempo Real")
if st.session_state.pos:
    for t, info in st.session_state.pos.items():
        st.sidebar.markdown(f"**{t}**")
        st.sidebar.write(f"Invertido: AR$ {info['m']:,.2f}")
        st.sidebar.caption(f"Compra: ${info['p']:.2f} | CCL: ${info['ccl']:.2f}")
        st.sidebar.divider()
else:
    st.sidebar.info("Sin posiciones abiertas.")

# Gráfico de evolución rápida
if len(st.session_state.historial_patrimonio) > 1:
    st.sidebar.write("### Evolución hoy")
    df_hist = pd.DataFrame(st.session_state.historial_patrimonio)
    st.sidebar.line_chart(df_hist.set_index('fecha'))
