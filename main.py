import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# --- CONFIGURACIÓN DE TIEMPO ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()

# --- CONFIGURACIÓN APP & SEGURIDAD ---
st.set_page_config(page_title="Simons GG v10.9.1 AUTO", page_icon="🦅", layout="wide")

try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

URL_DB = "https://docs.google.com/spreadsheets/d/19BvTkyD2ddrMsX1ghYGgnnq-BAfYJ_7qkNGqAsJel-M/edit?usp=drivesdk"
CAPITAL_INICIAL = 30000000.0

conn = st.connection("gsheets", type=GSheetsConnection)

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

def cargar_datos():
    try:
        df = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        if not df.empty:
            u = df.iloc[-1]
            pos = json.loads(str(u['posiciones']).replace("'", '"'))
            hist = json.loads(str(u['historial']).replace("'", '"'))
            return float(u['saldo']), pos, hist
    except:
        return 33362112.69, {}, [{"fecha": ahora_dt.strftime("%Y-%m-%d"), "t": 33362112.69}]

def guardar_progreso_auto():
    try:
        df_actual = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        nueva_fila = pd.DataFrame([{
            "saldo": float(st.session_state.saldo),
            "posiciones": json.dumps(st.session_state.pos),
            "historial": json.dumps(st.session_state.hist),
            "update": obtener_hora_argentina().strftime("%Y-%m-%d %H:%M")
        }])
        df_final = pd.concat([df_actual, nueva_fila], ignore_index=True)
        conn.update(spreadsheet=URL_DB, worksheet="Hoja1", data=df_final)
    except: pass

# --- INICIALIZACIÓN DE SESIÓN ---
if 'saldo' not in st.session_state:
    s, p, h = cargar_datos()
    st.session_state.update({'saldo': s, 'pos': p, 'hist': h})

# --- CÁLCULO DE MÉTRICAS EN TIEMPO REAL ---
# Calculamos esto ANTES de mostrar nada para que refleje la última operación
valor_cedears = sum(float(v.get('m', 0)) for v in st.session_state.pos.values())
patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento_total = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

# --- INTERFAZ SUPERIOR ---
st.title("🦅 Simons GG v10.9.1 - BOT AUTO 🤑")

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Invertido en Cedears", f"AR$ {valor_cedears:,.2f}")

# --- MONITOR DE MERCADO Y BOT ---
activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=120)
def procesar_bot():
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
    ccl_m = np.median(ccls) if ccls else None
    return df, ccl_m

df_m, ccl_m = procesar_bot()

# --- LÓGICA DE EJECUCIÓN (VIST COMPRADO SEGÚN TU MAIL) ---
if ccl_m and not df_m.empty:
    for _, row in df_m.iterrows():
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        # COMPRA AUTO (Ticket 8%)
        if desvio < -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS'], 'ccl': row['CCL']}
                enviar_alerta_mail(f"🦅 COMPRA: {activo}", f"Bot ejecutó compra de {activo}\nCCL: {row['CCL']:.2f}\nDesvío: {desvio*100:.2f}%")
                guardar_progreso_auto()
                st.rerun() # Forzar recarga para actualizar total y efectivo

        # VENTA AUTO
        elif desvio > 0.005 and activo in st.session_state.pos:
            monto_v = st.session_state.pos[activo]['m']
            st.session_state.saldo += monto_v
            del st.session_state.pos[activo]
            enviar_alerta_mail(f"🦅 VENTA: {activo}", f"Bot ejecutó venta de {activo}\nDesvío: {desvio*100:.2f}%")
            guardar_progreso_auto()
            st.rerun()

# --- VISUALIZACIÓN DE TABLA Y CARTERA ---
if ccl_m:
    st.write(f"### CCL Mediano: ${ccl_m:.2f}")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if ((r['CCL']/ccl_m)-1) < -0.005 and r['Clima']=="🟢" else ("🔴 VENTA" if ((r['CCL']/ccl_m)-1) > 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_m[['Activo', 'Señal', 'Clima', 'CCL', 'ARS']], use_container_width=True, hide_index=True)

# Panel lateral con la operación que mencionaste
st.sidebar.header("📂 Cartera Actual")
if st.session_state.pos:
    for ticker, info in st.session_state.pos.items():
        st.sidebar.subheader(f"🦅 {ticker}")
        st.sidebar.write(f"Inversión: AR$ {info['m']:,.2f}")
        st.sidebar.caption(f"Precio compra: ${info['p']:.2f} | CCL: ${info['ccl']:.2f}")
else:
    st.sidebar.info("Buscando oportunidades...")

st.sidebar.divider()
if st.sidebar.button("💾 Sincronizar Excel"):
    guardar_progreso_auto()
    st.sidebar.success("OK")
