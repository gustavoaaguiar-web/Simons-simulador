import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE ZONA HORARIA ARGENTINA ---
def obtener_hora_argentina():
    return datetime.now() - timedelta(hours=3)

ahora_dt = obtener_hora_argentina()
ahora = ahora_dt.time()

# --- CONFIGURACIÓN DE APP ---
st.set_page_config(page_title="Simons GG v10.5", page_icon="🦅", layout="wide")
URL_DB = "https://docs.google.com/spreadsheets/d/19BvTkyD2ddrMsX1ghYGgnnq-BAfYJ_7qkNGqAsJel-M/edit?usp=drivesdk"
CAPITAL_INICIAL = 30000000.0

conn = st.connection("gsheets", type=GSheetsConnection)

# --- LÓGICA DE PERSISTENCIA (AUTO-GUARDADO) ---
def cargar_datos():
    try:
        df = conn.read(spreadsheet=URL_DB, worksheet="Hoja1", ttl=0)
        if not df.empty:
            u = df.iloc[-1]
            return float(u['saldo']), json.loads(str(u['posiciones']).replace("'", '"')), json.loads(str(u['historial']).replace("'", '"'))
    except:
        # Valores por defecto si falla la conexión
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
        pass # Silencioso para no interrumpir la experiencia de usuario

# Inicializar sesión
if 'saldo' not in st.session_state:
    s, p, h = cargar_datos()
    st.session_state.update({'saldo': s, 'pos': p, 'hist': h})

# --- INTERFAZ PRINCIPAL ---
st.title("🦅 Simons GG v10.5 🤑")

# Estado del Mercado
mercado_abierto = datetime.strptime("11:00", "%H:%M").time() <= ahora <= datetime.strptime("17:00", "%H:%M").time()
if mercado_abierto:
    st.success(f"🟢 MERCADO ABIERTO - Hora Arg: {ahora.strftime('%H:%M')}")
else:
    st.info(f"⚪ MERCADO CERRADO - Hora Arg: {ahora.strftime('%H:%M')}")

# Cálculo de Patrimonio
patrimonio_total = st.session_state.saldo + sum(float(i.get('m', 0)) for i in st.session_state.pos.values())
rendimiento_h = ((patrimonio_total / CAPITAL_INICIAL) - 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_h:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {(patrimonio_total * 0.08):,.2f}")

# --- MONITOR DE ARBITRAJE (UMBRAL 0.5%) ---
st.subheader("📊 Monitor de Arbitraje")

activos = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

@st.cache_data(ttl=60)
def fetch_market():
    datos, ccls = [], []
    for t, r in activos.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            p_u = float(h_usd.Close.iloc[-1])
            p_a = float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            # Modelo HMM para Clima
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
        # CAMBIO SOLICITADO: Umbral del 0.5% (0.005)
        def asignar_señal(row):
            desvio = (row['CCL'] / ccl_m) - 1
            if desvio < -0.005 and row['raw_clima'] == "🟢": return "🟢 COMPRA"
            if desvio > 0.005: return "🔴 VENTA"
            return "⚖️ MANTENER"
        
        df['Desvío %'] = df['CCL'].apply(lambda x: f"{((x / ccl_m) - 1) * 100:+.2f}%")
        df['Señal'] = df.apply(asignar_señal, axis=1)
        df['CCL'] = df['CCL'].map("${:,.2f}".format)
        
    return df[['Activo', 'CCL', 'Clima', 'Señal', 'Desvío %', 'ARS']], ccl_m

df_m, ccl_m = fetch_market()
st.caption(f"CCL Mediano Sugerido: ${ccl_m:.2f}")

# Estilo de la tabla
def color_señal(val):
    if 'COMPRA' in str(val): return 'background-color: #004d00; color: white'
    if 'VENTA' in str(val): return 'background-color: #4d0000; color: white'
    return ''

st.dataframe(df_m.style.applymap(color_señal, subset=['Señal']), use_container_width=True, hide_index=True)

# --- AUTO-GUARDADO AL FINAL DE LA EJECUCIÓN ---
# Esto asegura que cada vez que alguien vea el monitor, el "update" se registre en el Excel.
guardar_progreso_auto(st.session_state.saldo, st.session_state.pos, st.session_state.hist)

st.caption(f"Última sincronización automática: {ahora_dt.strftime('%H:%M:%S')}")
