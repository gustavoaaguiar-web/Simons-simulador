import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime, time, timedelta
import smtplib
from email.message import EmailMessage

# --- CONFIGURACIÓN DE ZONA HORARIA ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()
ahora = ahora_dt.time()

# --- CONFIGURACIÓN APP & SEGURIDAD ---
st.set_page_config(page_title="Simons GG v11", page_icon="🦅", layout="wide")

try:
    MI_MAIL = st.secrets["MI_MAIL"]
    CLAVE_APP = st.secrets["CLAVE_APP"]
except:
    MI_MAIL = "gustavoaaguiar99@gmail.com"
    CLAVE_APP = "oshrmhfqzvabekzt"

URL_DB = "https://docs.google.com/spreadsheets/d/19BvTkyD2ddrMsX1ghYGgnnq-BAfYJ_7qkNGqAsJel-M/edit?usp=drivesdk"
CAPITAL_INICIAL = 30000000.0

# Conexión a Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# --- CARGA DE DATOS ---
def cargar_datos():
    try:
        df = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        if not df.empty:
            u = df.iloc[-1]
            return float(u['saldo']), json.loads(str(u['posiciones']).replace("'", '"')), json.loads(str(u['historial']).replace("'", '"'))
    except:
        return 33362112.69, {}, [{"fecha": ahora_dt.strftime("%Y-%m-%d"), "t": 33362112.69}]

if 'saldo' not in st.session_state:
    s, p, h = cargar_datos()
    st.session_state.update({'saldo': s, 'pos': p, 'hist': h})

# --- FUNCIONES AUXILIARES ---
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

# --- INTERFAZ ---
st.title("🦅 Simons GG v11 🤑")

patrimonio_total = st.session_state.saldo + sum(float(i.get('m', 0)) for i in st.session_state.pos.values())
rendimiento_h = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100
ticket_sugerido = patrimonio_total * 0.08

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_h:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {ticket_sugerido:,.2f}")

# --- MONITOR DE MERCADO ---
st.subheader("📊 Monitor de Arbitraje")

activos = {
    'AAPL': 20, 'TSLA': 15, 'NVDA': 24, 'MSFT': 30, 'MELI': 120, 
    'GGAL': 10, 'YPF': 1, 'VIST': 3, 'PAM': 25, 'BMA': 10,
    'CEPU': 10, 'GOOGL': 58, 'AMZN': 144, 'META': 24
}

@st.cache_data(ttl=300)
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
            
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "USD": p_u, "ARS": p_a, "raw_clima": clima})
        except: continue
    
    df = pd.DataFrame(datos)
    ccl_m = np.median([d['CCL'] for d in datos]) if datos else 0
    return df, ccl_m

df_res, ccl_m = fetch_market()

if not df_res.empty:
    st.caption(f"CCL Mediano: ${ccl_m:.2f}")
    
    def procesar(row):
        desvio = (row['CCL'] / ccl_m) - 1
        row['Desvío %'] = f"{desvio*100:+.2f}%"
        if desvio < -0.0065 and row['raw_clima'] == "🟢": row['Señal'] = "🟢 COMPRA"
        elif desvio > 0.0065: row['Señal'] = "🔴 VENTA"
        else: row['Señal'] = "⚖️ MANTENER"
        return row

    df_final = df_res.apply(procesar, axis=1)
    
    st.dataframe(
        df_final[['Activo', 'Señal', 'Desvío %', 'Clima', 'CCL', 'ARS', 'USD']]
        .style.applymap(lambda x: 'background-color: #004d00; color: white' if 'COMPRA' in str(x) else ('background-color: #4d0000; color: white' if 'VENTA' in str(x) else ''), subset=['Señal']), 
        use_container_width=True, hide_index=True
    )

    # --- PANEL MANUAL (INTEGRADO) ---
    st.divider()
    st.subheader("🕹️ Panel de Control Manual")
    col_sel, col_btns = st.columns([1, 2])
    
    with col_sel:
        activo_op = st.selectbox("Seleccionar activo", df_final['Activo'].unique())
        precio_ars = df_final[df_final['Activo'] == activo_op]['ARS'].values[0]
        st.write(f"Precio: AR$ {precio_ars:,.2f}")

    with col_btns:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button(f"🛒 Comprar {activo_op} (8%)", use_container_width=True):
                if st.session_state.saldo >= ticket_sugerido:
                    st.session_state.saldo -= ticket_sugerido
                    pos_actual = st.session_state.pos.get(activo_op, {'m': 0})
                    pos_actual['m'] = float(pos_actual['m']) + ticket_sugerido
                    pos_actual['fecha'] = ahora_dt.strftime("%Y-%m-%d %H:%M")
                    st.session_state.pos[activo_op] = pos_actual
                    st.success(f"Comprado {activo_op}")
                    st.rerun()
                else:
                    st.error("Saldo insuficiente")
        
        with btn_c2:
            if activo_op in st.session_state.pos:
                if st.button(f"💰 Vender TODO {activo_op}", use_container_width=True):
                    monto_recuperado = float(st.session_state.pos[activo_op]['m'])
                    st.session_state.saldo += monto_recuperado
                    del st.session_state.pos[activo_op]
                    st.warning(f"Posición cerrada en {activo_op}")
                    st.rerun()
            else:
                st.button("Sin posición", disabled=True, use_container_width=True)

    # --- SIDEBAR & ALERTAS ---
    st.sidebar.header("🛠 Simons Control")
    alertas = df_final[df_final['Señal'].str.contains("COMPRA|VENTA")]
    
    if st.sidebar.button("🧪 TEST DE CONEXIÓN"):
        if enviar_alerta_mail("🦅 Simons Test", "Conexión confirmada."):
            st.sidebar.success("Mail enviado.")

    if not alertas.empty:
        if st.sidebar.button("📧 ENVIAR SEÑALES"):
            cuerpo = f"🦅 INFORME SIMONS GG v11\nPatrimonio: AR$ {patrimonio_total:,.2f}\n"
            cuerpo += "="*30 + "\n"
            for _, r in alertas.iterrows():
                cuerpo += f"{r['Activo']}: {r['Señal']} ({r['Desvío %']})\n"
                cuerpo += f"Clima: {r['Clima']}\n"
                cuerpo += "-"*10 + "\n"
            if enviar_alerta_mail(f"🦅 Alerta: {len(alertas)} señales", cuerpo):
                st.sidebar.success("Señales enviadas.")

    # --- GUARDADO ---
    st.sidebar.divider()
    if st.sidebar.button("💾 GUARDAR EN SHEETS"):
        nueva_fila = pd.DataFrame([{
            "saldo": st.session_state.saldo,
            "posiciones": json.dumps(st.session_state.pos),
            "historial": json.dumps(st.session_state.hist),
            "update": ahora_dt.strftime("%Y-%m-%d %H:%M")
        }])
        try:
            conn.update(spreadsheet=URL_DB, data=nueva_fila)
            st.sidebar.success("Sincronizado correctamente.")
        except: st.sidebar.error("Error al guardar.")

else:
    st.warning("Escaneando mercado...")

# Mostrar Cartera abajo
st.subheader("💼 Mi Cartera")
if st.session_state.pos:
    st.table(pd.DataFrame.from_dict(st.session_state.pos, orient='index'))
else:
    st.write("Sin posiciones abiertas.")
