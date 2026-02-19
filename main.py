import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import smtplib
import json
import os
from email.message import EmailMessage

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v12.2", page_icon="🦅", layout="wide")

# Auto-Refresh cada 5 minutos
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

activos_dict = {
    'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 
    'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 
    'AMZN':144, 'META':24, 'VIST':3, 'PAM':25
}

# --- PERSISTENCIA (Incluye notificados para evitar spam) ---
ARCHIVO_ESTADO = "simons_state.json"

def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, "r") as f:
                data = json.load(f)
                if "notificados" not in data: data["notificados"] = []
                return data
        except: pass
    return {"saldo": 33362112.69, "pos": {}, "notificados": []}

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

def guardar_estado():
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump({
            "saldo": st.session_state.saldo, 
            "pos": st.session_state.pos,
            "notificados": st.session_state.notificados
        }, f)

# --- FUNCIÓN DE MAIL (Anti-Spam) ---
def enviar_alerta_unica(asunto, cuerpo, op_id):
    # Solo envía si el ID de la operación no está en la lista de notificados
    if op_id not in st.session_state.notificados:
        MI_MAIL = "gustavoaaguiar99@gmail.com"
        CLAVE_APP = "oshrmhfqzvabekzt"
        
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
            # Registramos que ya avisamos
            st.session_state.notificados.append(op_id)
            if len(st.session_state.notificados) > 100: st.session_state.notificados.pop(0)
            return True
        except: return False
    return False

# --- CAPTURA DE MERCADO ---
@st.cache_data(ttl=290)
def fetch_market():
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
            datos.append({"Activo": t, "CCL": ccl, "Clima": clima, "ARS": p_a, "USD": p_u})
        except:
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD": 0})
    df = pd.DataFrame(datos)
    mediana = np.nanmedian(ccls) if ccls else None
    return df, mediana

df_m, ccl_m = fetch_market()

# --- CÁLCULOS PATRIMONIO ---
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    p_actual_fila = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    precio_hoy = p_actual_fila[0] if len(p_actual_fila) > 0 and p_actual_fila[0] > 0 else info['p']
    valor_cedears += (info['m'] / info['p']) * precio_hoy

patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento_total = ((patrimonio_total / 30000000.0) - 1) * 100

# --- LÓGICA DE TRADING (0.5%) ---
if ccl_m:
    for _, row in df_m.iterrows():
        if np.isnan(row['CCL']): continue
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        ahora_str = datetime.now().strftime("%Y%m%d_%H") # ID único por hora

        # COMPRA
        if desvio <= -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS']}
                enviar_alerta_unica(f"🦅 COMPRA: {activo}", f"Comprado {activo} a ${row['ARS']}\nCCL: {row['CCL']:.2f}\nDesvío: {desvio*100:.2f}%", f"buy_{activo}_{ahora_str}")
                guardar_estado()
                st.rerun()

        # VENTA
        elif desvio >= 0.005 and activo in st.session_state.pos:
            info_c = st.session_state.pos[activo]
            st.session_state.saldo += (info_c['m'] / info_c['p']) * row['ARS']
            del st.session_state.pos[activo]
            enviar_alerta_unica(f"🦅 VENTA: {activo}", f"Vendido {activo} a ${row['ARS']}\nDesvío: {desvio*100:.2f}%", f"sell_{activo}_{ahora_str}")
            guardar_estado()
            st.rerun()

# --- INTERFAZ (IGUAL A TU VERSIÓN PREFERIDA) ---
st.title("🦅 Simons GG v12.2 🤑")
c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo Disponible", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Valor Mercado Cedears", f"AR$ {valor_cedears:,.2f}")

st.divider()

if ccl_m:
    st.header(f"CCL Mercado: ${ccl_m:,.2f}")
    df_m['%'] = df_m['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if not np.isnan(x) else "S/D")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if not np.isnan(r['CCL']) and ((r['CCL']/ccl_m)-1) <= -0.005 and r['Clima'] == "🟢" else ("🔴 VENTA" if not np.isnan(r['CCL']) and ((r['CCL']/ccl_m)-1) >= 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_m[['Activo', '%', 'Clima', 'Señal', 'ARS', 'USD']], use_container_width=True, hide_index=True, height=530)

# --- SIDEBAR DETALLADO ---
st.sidebar.header("📂 Cartera y Ganancias")
for t, info in st.session_state.pos.items():
    p_actual_arr = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    p_act = p_actual_arr[0] if len(p_actual_arr) > 0 and p_actual_arr[0] > 0 else info['p']
    gan_ars = ((info['m'] / info['p']) * p_act) - info['m']
    gan_pct = ((p_act / info['p']) - 1) * 100
    color = "green" if gan_ars >= 0 else "red"
    with st.sidebar.expander(f"📦 {t}", expanded=True):
        st.write(f"**Ganancia:** :{color}[AR$ {gan_ars:,.2f} ({gan_pct:+.2f}%)]")
        st.write(f"Entrada: `${info['p']:,.2f}` | Actual: `${p_act:,.2f}`")
