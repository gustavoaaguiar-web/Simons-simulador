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

# --- CONFIGURACIÓN DE ZONA HORARIA ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()
ahora = ahora_dt.time()

# --- CONFIGURACIÓN DE APP & SECRETS ---
st.set_page_config(page_title="Simons GG v10.6", page_icon="🦅", layout="wide")

# Intentar leer desde secrets, si no, usa el respaldo
try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

URL_DB = "https://docs.google.com/spreadsheets/d/19BvTkyD2ddrMsX1ghYGgnnq-BAfYJ_7qkNGqAsJel-M/edit?usp=drivesdk"
CAPITAL_INICIAL = 30000000.0

conn = st.connection("gsheets", type=GSheetsConnection)

# --- FUNCIONES DE COMUNICACIÓN Y PERSISTENCIA ---
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
    except:
        return False

def cargar_datos():
    try:
        df = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        if not df.empty:
            u = df.iloc[-1]
            return float(u['saldo']), json.loads(str(u['posiciones']).replace("'", '"')), json.loads(str(u['historial']).replace("'", '"'))
    except:
        return 33362112.69, {}, [{"fecha": ahora_dt.strftime("%Y-%m-%d"), "t": 33362112.69}]

def guardar_progreso_auto(saldo, pos, hist):
    try:
        df_actual = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        nueva_fila = pd.DataFrame([{
            "saldo": float(saldo),
            "posiciones": json.dumps(pos),
            "historial": json.dumps(hist),
            "update": obtener_hora_argentina().strftime("%Y-%m-%d %H:%M")
        }])
        df_final = pd.concat([df_actual, nueva_fila], ignore_index=True)
        conn.update(spreadsheet=URL_DB, worksheet="Hoja1", data=df_final)
    except:
        pass

# Inicializar sesión
if 'saldo' not in st.session_state:
    s, p, h = cargar_datos()
    st.session_state.update({'saldo': s, 'pos': p, 'hist': h})

# --- INTERFAZ PRINCIPAL ---
st.title("🦅 Simons GG v10.6 🤑")

patrimonio_total = st.session_state.saldo + sum(float(i.get('m', 0)) for i in st.session_state.pos.values())
rendimiento_h = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_h:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {(patrimonio_total * 0.08):,.2f}")

# --- MONITOR DE MERCADO ---
st.subheader("📊 Monitor de Arbitraje")

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
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            
            datos.append({
                "Activo": t, "CCL": ccl, "Clima": clima,
                "ARS": p_a, "USD": p_u, "raw_clima": clima
            })
        except: continue
    
    df = pd.DataFrame(datos)
    if not df.empty:
        ccl_m = np.median(df['CCL'])
        def asignar_señal(row):
            desvio = (row['CCL'] / ccl_m) - 1
            if desvio < -0.005 and row['raw_clima'] == "🟢": return "🟢 COMPRA"
            if desvio > 0.005: return "🔴 VENTA"
            return "⚖️ MANTENER"
        
        df['Desvío %'] = df['CCL'].apply(lambda x: f"{((x / ccl_m) - 1) * 100:+.2f}%")
        df['Señal'] = df.apply(asignar_señal, axis=1)
        df['CCL_f'] = df['CCL'].map("${:,.2f}".format)
        return df, ccl_m
    return pd.DataFrame(), 0

df_m, ccl_m = fetch_market()

if not df_m.empty:
    st.caption(f"CCL Mediano Sugerido: ${ccl_m:.2f}")
    
    st.dataframe(
        df_m[['Activo', 'Señal', 'Desvío %', 'Clima', 'CCL_f', 'ARS']]
        .style.applymap(lambda x: 'background-color: #004d00; color: white' if 'COMPRA' in str(x) else ('background-color: #4d0000; color: white' if 'VENTA' in str(x) else ''), subset=['Señal']), 
        use_container_width=True, hide_index=True
    )

    # --- SIDEBAR & ALERTAS ---
    st.sidebar.header("🛠 Simons Control")
    alertas = df_m[df_m['Señal'].str.contains("COMPRA|VENTA")]

    if st.sidebar.button("🧪 TEST MAIL"):
        if enviar_alerta_mail("🦅 Simons Test", "Conexión confirmada."):
            st.sidebar.success("Mail enviado.")
        else: st.sidebar.error("Error al enviar.")

    if not alertas.empty:
        if st.sidebar.button("📧 ENVIAR SEÑALES"):
            cuerpo = f"🦅 INFORME SIMONS GG v10.6\nPatrimonio: AR$ {patrimonio_total:,.2f}\n"
            cuerpo += "="*30 + "\n"
            for _, r in alertas.iterrows():
                cuerpo += f"{r['Activo']}: {r['Señal']} ({r['Desvío %']})\n"
                cuerpo += f"Clima HMM: {r['Clima']}\n"
                cuerpo += "-"*10 + "\n"
            if enviar_alerta_mail(f"🦅 Alerta: {len(alertas)} señales", cuerpo):
                st.sidebar.success(f"{len(alertas)} alertas enviadas.")

# Auto-guardado de progreso
guardar_progreso_auto(st.session_state.saldo, st.session_state.pos, st.session_state.hist)
st.caption(f"Sincronizado: {ahora_dt.strftime('%H:%M:%S')}")
