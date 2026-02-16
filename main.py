import streamlit as st
from st_gsheets_connection import GSheetsConnection
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime, time, timedelta

# --- CONFIGURACIÓN DE ZONA HORARIA ---
# Ajuste manual: Restamos 3 horas al UTC del servidor para tener la hora de Argentina
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()
ahora = ahora_dt.time()

# --- CONFIGURACIÓN GENERAL ---
URL_DB = "https://docs.google.com/spreadsheets/d/19BvTkyD2ddrMsX1ghYGgnnq-BAfYJ_7qkNGqAsJel-M/edit?usp=drivesdk"
CAPITAL_INICIAL = 30000000.0

st.set_page_config(page_title="Simons GG v10.4", page_icon="🦅", layout="wide")
conn = st.connection("gsheets", type=GSheetsConnection)

# --- LÓGICA DE TIEMPO ---
limite_compra = time(16, 30)
cierre_obligatorio = time(16, 50)
es_ventana_liq = limite_compra <= ahora <= cierre_obligatorio
mercado_abierto = time(11, 0) <= ahora <= time(17, 0)

# --- CARGA DE DATOS ---
def cargar_datos():
    try:
        df = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        if not df.empty:
            u = df.iloc[-1]
            return float(u['saldo']), json.loads(str(u['posiciones']).replace("'", '"')), json.loads(str(u['historial']).replace("'", '"'))
    except:
        return 33362112.69, {}, [{"fecha": "2026-02-14", "t": 33362112.69}]

if 'saldo' not in st.session_state:
    s, p, h = cargar_datos()
    st.session_state.update({'saldo': s, 'pos': p, 'hist': h})

# --- INTERFAZ ---
st.title("🦅 Simons GG v10.4 🤑")

# Cartel de estado con la hora corregida
if mercado_abierto:
    if es_ventana_liq:
        st.warning(f"⚠️ MODO LIQUIDACIÓN: Cerrar posiciones antes de las 16:50. Hora Arg: {ahora.strftime('%H:%M')}")
    else:
        st.success(f"🟢 MERCADO ABIERTO - Hora Arg: {ahora.strftime('%H:%M')}")
else:
    st.info(f"⚪ MERCADO CERRADO - Hora Arg: {ahora.strftime('%H:%M')}")

patrimonio_total = st.session_state.saldo + sum(float(i.get('m', 0)) for i in st.session_state.pos.values())
rendimiento_h = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_h:+.2f}% vs Inicial")
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
                "Activo": t, "CCL": f"{ccl:.2f}", "Clima": clima,
                "USD": f"{p_u:.2f}", "ARS": f"{p_a:.2f}",
                "raw_ccl": ccl, "raw_clima": clima
            })
        except: continue
    
    df = pd.DataFrame(datos)
    if not df.empty:
        ccl_m = np.median([float(x) for x in df['CCL']])
        df['Desvío %'] = df.apply(lambda x: f"{((float(x['CCL']) / ccl_m) - 1) * 100:+.2f}%", axis=1)
        
        def asignar_señal(row):
            desvio_num = (float(row['CCL']) / ccl_m) - 1
            if desvio_num < -0.0065 and row['raw_clima'] == "🟢": return "🟢 COMPRA"
            if desvio_num > 0.0065: return "🔴 VENTA"
            return "⚖️ MANTENER"
        
        df['Señal'] = df.apply(asignar_señal, axis=1)
        
    return df[['Activo', 'CCL', 'Clima', 'Señal', 'Desvío %', 'ARS', 'USD']], ccl_m

df_m, ccl_m = fetch_market()
st.caption(f"CCL Mediano: ${ccl_m:.2f}")

def color_señal(val):
    if 'COMPRA' in str(val): color = '#004d00'
    elif 'VENTA' in str(val): color = '#4d0000'
    else: return ''
    return f'background-color: {color}; color: white; font-weight: bold'

st.dataframe(df_m.style.applymap(color_señal, subset=['Señal']), use_container_width=True, hide_index=True)

# --- GUARDADO ---
st.divider()
if st.button("💾 GUARDAR EN EXCEL"):
    nueva_fila = pd.DataFrame([{
        "saldo": st.session_state.saldo,
        "posiciones": json.dumps(st.session_state.pos),
        "historial": json.dumps(st.session_state.hist),
        "update": ahora_dt.strftime("%Y-%m-%d %H:%M")
    }])
    try:
        conn.update(spreadsheet=URL_DB, data=nueva_fila)
        st.success("¡Datos sincronizados!")
    except: st.error("Error al guardar")
