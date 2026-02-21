import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, time
import smtplib
import json
import os
from email.message import EmailMessage
import pytz

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v12.9", page_icon="🦅", layout="wide")
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

# Configuración de Zona Horaria Argentina
arg_tz = pytz.timezone('America/Argentina/Buenos_Aires')
ahora_arg = datetime.now(arg_tz).time()

# --- PERSISTENCIA CON EL SALDO DE CIERRE ---
ARCHIVO_ESTADO = "simons_state.json"

def cargar_estado():
    # SALDO EXACTO SEGÚN CAPTURA v12.8
    SALDO_OBJETIVO = 33665259.87
    
    estado_inicial = {
        "saldo": SALDO_OBJETIVO, 
        "pos": {}, # Se inicia vacío porque el Valor Cedears en captura es 0.00
        "notificados": []
    }
    
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, "r") as f:
                data = json.load(f)
                # Forzamos la corrección del salto
                data["saldo"] = SALDO_OBJETIVO
                return data
        except: pass
    return estado_inicial

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

def guardar_estado():
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump({
            "saldo": st.session_state.saldo, 
            "pos": st.session_state.pos, 
            "notificados": st.session_state.notificados
        }, f)

def enviar_alerta_operacion(asunto, cuerpo, op_id, es_test=False):
    if es_test or op_id not in st.session_state.notificados:
        MI_MAIL, CLAVE_APP = "gustavoaaguiar99@gmail.com", "oshrmhfqzvabekzt"
        msg = EmailMessage()
        msg.set_content(cuerpo)
        msg['Subject'] = asunto
        msg['From'], msg['To'] = MI_MAIL, MI_MAIL
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(MI_MAIL, CLAVE_APP)
            server.send_message(msg)
            server.quit()
            if not es_test:
                st.session_state.notificados.append(op_id)
            return True
        except: return False
    return False

# --- CAPTURA DE MERCADO ---
activos_dict = {'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 'AMZN':144, 'META':24, 'VIST':3, 'PAM':25}

@st.cache_data(ttl=290)
def fetch_market():
    datos, ccls = [], []
    for t, r in activos_dict.items():
        try:
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
            ccl_i = (p_a * r) / p_u
            ccls.append(ccl_i)
            ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
            model = GaussianHMM(n_components=3, random_state=42).fit(ret)
            clima = "🟢" if model.predict(ret)[-1] == 0 else "🔴"
            datos.append({"Activo": t, "CCL": ccl_i, "Clima": clima, "ARS": p_a, "USD": p_u})
        except:
            datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD": 0})
    df = pd.DataFrame(datos)
    mediana = np.nanmedian(ccls) if ccls else None
    return df, mediana

df_m, ccl_m = fetch_market()

# --- LÓGICA HORARIA ---
HORA_LIMITE_COMPRA = time(16, 30)
HORA_CIERRE_TOTAL = time(16, 50)
puedo_comprar = ahora_arg < HORA_LIMITE_COMPRA
panico_sell = ahora_arg >= HORA_CIERRE_TOTAL

# --- CÁLCULOS & TRADING ---
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    p_actual_fila = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    precio_hoy = p_actual_fila[0] if len(p_actual_fila) > 0 and p_actual_fila[0] > 0 else info['p']
    valor_cedears += (info['m'] / info['p']) * precio_hoy

patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento_total = ((patrimonio_total / 30000000.0) - 1) * 100

if ccl_m:
    # CIERRE FORZADO 16:50
    if panico_sell and st.session_state.pos:
        for activo in list(st.session_state.pos.keys()):
            info_c = st.session_state.pos[activo]
            p_venta = df_m.loc[df_m['Activo'] == activo, 'ARS'].values[0]
            st.session_state.saldo += (info_c['m'] / info_c['p']) * p_venta
            del st.session_state.pos[activo]
            enviar_alerta_operacion(f"⚠️ CIERRE AUTOMÁTICO: {activo}", f"Venta 16:50.\nPrecio: ${p_venta}", f"panic_{activo}")
        guardar_estado()
        st.rerun()

    # TRADING NORMAL
    for _, row in df_m.iterrows():
        if np.isnan(row['CCL']): continue
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        ts_id = datetime.now().strftime("%Y%m%d_%H%M")

        if puedo_comprar and desvio <= -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_t = patrimonio_total * 0.08
            if st.session_state.saldo >= monto_t:
                st.session_state.saldo -= monto_t
                st.session_state.pos[activo] = {'m': monto_t, 'p': row['ARS']}
                enviar_alerta_operacion(f"🦅 COMPRA: {activo}", f"Precio: ${row['ARS']}\nCCL: ${row['CCL']:.2f}", f"buy_{activo}_{ts_id}")
                guardar_estado()
                st.rerun()
        elif not panico_sell and desvio >= 0.005 and activo in st.session_state.pos:
            info_c = st.session_state.pos[activo]
            st.session_state.saldo += (info_c['m'] / info_c['p']) * row['ARS']
            del st.session_state.pos[activo]
            enviar_alerta_operacion(f"🦅 VENTA: {activo}", f"Precio: ${row['ARS']}\nCCL: ${row['CCL']:.2f}", f"sell_{activo}_{ts_id}")
            guardar_estado()
            st.rerun()

# --- INTERFAZ ---
st.title("🦅 Simons GG v12.9 🤑")
estado_txt = "🟢 OPERANDO" if puedo_comprar else ("🟡 SOLO VENTAS" if ahora_arg < HORA_CIERRE_TOTAL else "🔴 CERRADO")
st.caption(f"Hora ARG: {ahora_arg.strftime('%H:%M:%S')} | Estado: {estado_txt}")

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Valor Cedears", f"AR$ {valor_cedears:,.2f}")

st.divider()

if ccl_m:
    df_m['%'] = df_m['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if not np.isnan(x) else "S/D")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if puedo_comprar and not np.isnan(r['CCL']) and ((r['CCL']/ccl_m)-1) <= -0.005 and r['Clima'] == "🟢" else ("🔴 VENTA" if not np.isnan(r['CCL']) and ((r['CCL']/ccl_m)-1) >= 0.005 else "⚖️ MANTENER"), axis=1)
    df_m['CCL_Display'] = df_m['CCL'].map(lambda x: f"${x:,.2f}")
    st.dataframe(df_m[['Activo', '%', 'Clima', 'Señal', 'CCL_Display', 'ARS', 'USD']], use_container_width=True, hide_index=True)

# --- SIDEBAR ---
st.sidebar.header("📂 Cartera y Ganancias")
if st.sidebar.button("🧪 Test Mail"):
    enviar_alerta_operacion("🦅 TEST SIMONS", "Prueba OK", "test", es_test=True)

st.sidebar.divider()
if st.session_state.pos:
    for t, info in st.session_state.pos.items():
        p_act = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0] if t in df_m['Activo'].values else info['p']
        gan_ars = ((info['m'] / info['p']) * p_act) - info['m']
        gan_pct = ((p_act / info['p']) - 1) * 100
        color = "green" if gan_ars >= 0 else "red"
        with st.sidebar.expander(f"📦 {t}", expanded=True):
            st.write(f"**Ganancia:** :{color}[AR$ {gan_ars:,.2f} ({gan_pct:+.2f}%)]")
            st.write(f"Inversión: AR$ {info['m']:,.2f}")
            st.write(f"Entrada: `${info['p']:,.2f}` | Actual: `${p_act:,.2f}`")
else:
    st.sidebar.info("Cartera vacía (Liquidez total)")
            
