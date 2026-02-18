import streamlit as st
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
st.set_page_config(page_title="Simons GG v11 LIGHT", page_icon="🦅", layout="wide")

CAPITAL_INICIAL = 30000000.0

# --- PERSISTENCIA LOCAL (Se resetea si se recarga la página) ---
if 'saldo' not in st.session_state:
    st.session_state.saldo = 33362112.69
if 'pos' not in st.session_state:
    st.session_state.pos = {}

# --- INTERFAZ ---
st.title("🦅 Simons GG v11 - Modo Local 🤑")

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

@st.cache_data(ttl=120)
def fetch_market():
    datos, ccls = [], []
    for t, r in activos.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            # Pedimos menos datos para que sea más rápido y no falle
            h_usd = yf.download(t, period="1mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            if h_usd.empty or h_ars.empty: continue
            
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "USD": p_u, "ARS": p_a})
        except: continue
    
    df = pd.DataFrame(datos)
    ccl_m = np.median(ccls) if ccls else 0
    return df, ccl_m

df_res, ccl_m = fetch_market()

if not df_res.empty:
    st.caption(f"CCL Mediano: ${ccl_m:.2f}")
    
    def procesar(row):
        desvio = (row['CCL'] / ccl_m) - 1
        row['Desvío %'] = f"{desvio*100:+.2f}%"
        # Usamos tu lógica de la foto
        if desvio < -0.0065 and row['Clima'] == "🟢": row['Señal'] = "🟢 COMPRA"
        elif desvio > 0.0065: row['Señal'] = "🔴 VENTA"
        else: row['Señal'] = "⚖️ MANTENER"
        return row

    df_final = df_res.apply(procesar, axis=1)
    
    st.dataframe(
        df_final[['Activo', 'Señal', 'Desvío %', 'Clima', 'CCL', 'ARS', 'USD']]
        .style.applymap(lambda x: 'background-color: #004d00; color: white' if 'COMPRA' in str(x) else ('background-color: #4d0000; color: white' if 'VENTA' in str(x) else ''), subset=['Señal']), 
        use_container_width=True, hide_index=True
    )

    # --- PANEL MANUAL ---
    st.divider()
    st.subheader("🕹️ Ejecución Manual")
    col_sel, col_btns = st.columns([1, 2])
    
    with col_sel:
        activo_op = st.selectbox("Seleccionar activo", df_final['Activo'].unique())
        precio_ars = df_final[df_final['Activo'] == activo_op]['ARS'].values[0]
        st.info(f"Precio: AR$ {precio_ars:,.2f}")

    with col_btns:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button(f"🛒 Comprar {activo_op}", use_container_width=True):
                if st.session_state.saldo >= ticket_sugerido:
                    st.session_state.saldo -= ticket_sugerido
                    pos_actual = st.session_state.pos.get(activo_op, {'m': 0})
                    pos_actual['m'] = float(pos_actual['m']) + ticket_sugerido
                    st.session_state.pos[activo_op] = pos_actual
                    st.success(f"Comprado {activo_op}")
                    st.rerun()
        
        with btn_c2:
            if activo_op in st.session_state.pos:
                if st.button(f"💰 Vender TODO", use_container_width=True):
                    st.session_state.saldo += float(st.session_state.pos[activo_op]['m'])
                    del st.session_state.pos[activo_op]
                    st.warning(f"Vendido {activo_op}")
                    st.rerun()
else:
    st.warning("Cargando datos del mercado...")

# --- CARTERA ---
st.subheader("💼 Cartera Actual")
if st.session_state.pos:
    st.write(st.session_state.pos)
else:
    st.write("Sin posiciones abiertas.")
