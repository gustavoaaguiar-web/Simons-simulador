import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE ZONA HORARIA ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v11", page_icon="🦅", layout="wide")

URL_DB = "https://docs.google.com/spreadsheets/d/19BvTkyD2ddrMsX1ghYGgnnq-BAfYJ_7qkNGqAsJel-M/edit?usp=drivesdk"
CAPITAL_INICIAL = 30000000.0

# Intentar conexión con Google Sheets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    db_conectada = True
except Exception as e:
    db_conectada = False
    st.error(f"Error de conexión a la DB: {e}. Las operaciones no se guardarán al recargar.")

# --- CARGA DE DATOS ---
def cargar_datos():
    if db_conectada:
        try:
            df = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
            if not df.empty:
                u = df.iloc[-1]
                return float(u['saldo']), json.loads(str(u['posiciones']).replace("'", '"'))
        except:
            pass
    return 33362112.69, {} # Valores por defecto si falla la DB

if 'saldo' not in st.session_state:
    s, p = cargar_datos()
    st.session_state.update({'saldo': s, 'pos': p})

# --- CÁLCULOS DE PATRIMONIO ---
patrimonio_total = st.session_state.saldo + sum(float(i.get('m', 0)) for i in st.session_state.pos.values())
rendimiento_h = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100
ticket_sugerido = patrimonio_total * 0.08

# --- INTERFAZ ---
st.title("🦅 Simons GG v11 🤑")
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_h:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {ticket_sugerido:,.2f}")

# --- MONITOR (FETCH MARKET) ---
activos = {
    'AAPL': 20, 'TSLA': 15, 'NVDA': 24, 'MSFT': 30, 'MELI': 120, 
    'GGAL': 10, 'YPF': 1, 'VIST': 3, 'PAM': 25, 'BMA': 10,
    'CEPU': 10, 'GOOGL': 58, 'AMZN': 144, 'META': 24
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
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a, "USD": p_u})
        except: continue
    df = pd.DataFrame(datos)
    return df, np.median(ccls) if ccls else 0

df_res, ccl_m = fetch_market()

if not df_res.empty:
    def procesar(row):
        desvio = (row['CCL'] / ccl_m) - 1
        row['Desvío %'] = f"{desvio*100:+.2f}%"
        if desvio < -0.0065 and row['Clima'] == "🟢": row['Señal'] = "🟢 COMPRA"
        elif desvio > 0.0065: row['Señal'] = "🔴 VENTA"
        else: row['Señal'] = "⚖️ MANTENER"
        return row
    df_final = df_res.apply(procesar, axis=1)
    st.dataframe(df_final[['Activo', 'Señal', 'Desvío %', 'Clima', 'CCL', 'ARS']], use_container_width=True, hide_index=True)

    # --- EJECUCIÓN MANUAL ---
    st.divider()
    col_sel, col_btns = st.columns([1, 2])
    with col_sel:
        activo_op = st.selectbox("Operar activo", df_final['Activo'].unique())
    with col_btns:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button(f"🛒 Comprar {activo_op}"):
                if st.session_state.saldo >= ticket_sugerido:
                    st.session_state.saldo -= ticket_sugerido
                    st.session_state.pos[activo_op] = {'m': st.session_state.pos.get(activo_op, {}).get('m', 0) + ticket_sugerido}
                    st.success(f"Comprado {activo_op}. ¡RECUERDA GUARDAR!")
                    st.rerun()
        with btn_c2:
            if activo_op in st.session_state.pos:
                if st.button(f"💰 Vender {activo_op}"):
                    st.session_state.saldo += float(st.session_state.pos[activo_op]['m'])
                    del st.session_state.pos[activo_op]
                    st.rerun()

# --- GUARDADO EXPLÍCITO ---
st.sidebar.subheader("💾 Persistencia")
if st.sidebar.button("💾 GUARDAR CAMBIOS"):
    if db_conectada:
        nueva_fila = pd.DataFrame([{
            "saldo": st.session_state.saldo,
            "posiciones": json.dumps(st.session_state.pos),
            "historial": json.dumps([{"fecha": ahora_dt.strftime("%Y-%m-%d"), "t": patrimonio_total}]),
            "update": ahora_dt.strftime("%Y-%m-%d %H:%M")
        }])
        conn.update(spreadsheet=URL_DB, data=nueva_fila)
        st.sidebar.success("Sincronizado con Google Sheets")
    else:
        st.sidebar.error("No hay conexión con la base de datos.")

st.subheader("💼 Cartera")
st.write(st.session_state.pos)
