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
import time as time_lib

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG v14.6", page_icon="🦅", layout="wide")
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

# 1. ZONA HORARIA Y ESTADO
try:
    arg_tz = pytz.timezone('America/Argentina/Buenos_Aires')
    ahora_arg_dt = datetime.now(arg_tz)
except:
    ahora_arg_dt = datetime.now()

ahora_arg_time = ahora_arg_dt.time()
dia_semana = ahora_arg_dt.weekday()

# 2. PERSISTENCIA Y SALDO
ARCHIVO_ESTADO = "simons_state.json"
SALDO_INICIAL_SISTEMA = 30000000.0  # Base para el cálculo del % de aumento total
SALDO_OBJETIVO = 34456041.58

def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, "r") as f:
                return json.load(f)
        except: pass
    return {"saldo": SALDO_OBJETIVO, "pos": {}, "notificados": []}

if 'saldo' not in st.session_state:
    st.session_state.update(cargar_estado())

def guardar_estado():
    with open(ARCHIVO_ESTADO, "w") as f:
        json.dump({k: st.session_state[k] for k in ["saldo", "pos", "notificados"]}, f)

# 3. COMUNICACIONES (MAIL CON % DE DIFERENCIA)
def enviar_alerta(asunto, cuerpo, op_id, es_test=False):
    if es_test or op_id not in st.session_state.notificados:
        MI_MAIL, CLAVE = "gustavoaaguiar99@gmail.com", "oshrmhfqzvabekzt"
        msg = EmailMessage()
        msg.set_content(cuerpo)
        msg['Subject'], msg['From'], msg['To'] = asunto, MI_MAIL, MI_MAIL
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(MI_MAIL, CLAVE)
                server.send_message(msg)
            if not es_test: 
                st.session_state.notificados.append(op_id)
                guardar_estado()
            return True
        except: return False
    return False

# 4. CAPTURA DE MERCADO
activos_dict = {'AAPL':20, 'TSLA':15, 'NVDA':24, 'MSFT':30, 'MELI':120, 'GGAL':10, 'YPF':1, 'BMA':10, 'CEPU':10, 'GOOGL':58, 'AMZN':144, 'META':24, 'VIST':3, 'PAM':25}

@st.cache_data(ttl=290)
def fetch_market_v14_6():
    datos, ccls = [], []
    for t, r in activos_dict.items():
        exito = False
        for intento in range(3):
            try:
                tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
                h_usd = yf.download(t, period="3mo", interval="1d", progress=False, timeout=15)
                h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False, timeout=15)
                if not h_usd.empty and not h_ars.empty:
                    p_u, p_a = float(h_usd.Close.iloc[-1]), float(h_ars.Close.iloc[-1])
                    ccl_i = (p_a * r) / p_u
                    ccls.append(ccl_i)
                    ret = np.diff(np.log(h_usd.Close.values.flatten().reshape(-1, 1)), axis=0)
                    clima = "🟢" if GaussianHMM(n_components=3, random_state=42).fit(ret).predict(ret)[-1] == 0 else "🔴"
                    datos.append({"Activo": t, "CCL": ccl_i, "Clima": clima, "ARS": p_a, "USD": p_u})
                    exito = True
                    break
            except: time_lib.sleep(1)
        if not exito: datos.append({"Activo": t, "CCL": np.nan, "Clima": "⚪", "ARS": 0, "USD": 0})
    return pd.DataFrame(datos), (np.nanmedian(ccls) if ccls else None)

df_m, ccl_m = fetch_market_v14_6()

# 5. LÓGICA DE TRADING
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    p_act = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    precio = p_act[0] if len(p_act) > 0 and p_act[0] > 0 else info['p']
    valor_cedears += (info['m'] / info['p']) * precio

patrimonio = st.session_state.saldo + valor_cedears
# Cálculo del aumento total respecto al inicio del sistema
aumento_total_pct = ((patrimonio / SALDO_INICIAL_SISTEMA) - 1) * 100

es_hora_op = (0 <= dia_semana <= 4 and time(11,0) <= ahora_arg_time < time(16,30))

if ccl_m and es_hora_op:
    for _, row in df_m.iterrows():
        desvio = (row['CCL'] / ccl_m) - 1 if pd.notnull(row['CCL']) else 0
        activo = row['Activo']
        
        # COMPRA
        if desvio <= -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto = min(st.session_state.saldo, patrimonio * 0.125)
            if monto > 100000:
                st.session_state.saldo -= monto
                st.session_state.pos[activo] = {'m': monto, 'p': row['ARS']}
                cuerpo_mail = f"Eagle Eye detectó oportunidad:\n\nActivo: {activo}\nMonto: AR$ {monto:,.2f}\nPrecio: ${row['ARS']}\nDesvío CCL: {desvio*100:.2f}%"
                enviar_alerta(f"🦅 COMPRA: {activo}", cuerpo_mail, f"b_{activo}_{ahora_arg_dt.day}")
                guardar_estado()
                st.rerun()

        # VENTA
        if desvio >= 0.005 and activo in st.session_state.pos:
            info_c = st.session_state.pos[activo]
            val_final = (info_c['m'] / info_c['p']) * row['ARS']
            ganancia_ars = val_final - info_c['m']
            ganancia_pct = ((row['ARS'] / info_c['p']) - 1) * 100
            
            st.session_state.saldo += val_final
            del st.session_state.pos[activo]
            
            cuerpo_mail = f"Cierre de posición:\n\nActivo: {activo}\nGanancia: AR$ {ganancia_ars:,.2f}\nRendimiento: {ganancia_pct:+.2f}%\nPrecio Venta: ${row['ARS']}"
            enviar_alerta(f"🦅 VENTA: {activo}", cuerpo_mail, f"s_{activo}_{ahora_arg_dt.day}")
            guardar_estado()
            st.rerun()

# 6. UI Y BARRA LATERAL
st.title("🦅 Simons GG v14.6 🤑")
m1, m2, m3 = st.columns(3)
# Aquí agregamos el porcentaje de aumento del total en la métrica
m1.metric("Patrimonio Total", f"AR$ {patrimonio:,.2f}", f"{aumento_total_pct:+.2f}%")
m2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
m3.metric("En CEDEARs", f"AR$ {valor_cedears:,.2f}")

st.divider()

if not df_m.empty and ccl_m:
    df_vis = df_m.copy()
    df_vis['% Desv'] = df_vis['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%" if pd.notnull(x) else "---")
    df_vis['Señal'] = df_vis.apply(lambda r: "🟢 COMPRA" if pd.notnull(r['CCL']) and ((r['CCL']/ccl_m)-1) <= -0.005 and r['Clima'] == "🟢" else ("🔴 VENTA" if pd.notnull(r['CCL']) and ((r['CCL']/ccl_m)-1) >= 0.005 else "⚖️ MANTENER"), axis=1)
    st.dataframe(df_vis[['Activo', '% Desv', 'Clima', 'Señal', 'ARS', 'USD', 'CCL']].rename(columns={'CCL': 'CCL Impl.'}), use_container_width=True, hide_index=True)

with st.sidebar:
    st.header("🦅 Panel de Control")
    if st.button("🧪 Probar Envío de Mail"):
        test_cuerpo = f"Test de sistema v14.6\n\nPatrimonio: AR$ {patrimonio:,.2f}\nRendimiento Total: {aumento_total_pct:+.2f}%"
        if enviar_alerta("🦅 TEST Simons", test_cuerpo, "test", True):
            st.success("Mail enviado!")
        else: st.error("Error al enviar")
    
    st.divider()
    st.subheader("📂 Mi Cartera")
    if st.session_state.pos:
        for t, info in list(st.session_state.pos.items()):
            p_act = df_m.loc[df_m['Activo']==t, 'ARS'].values[0] if t in df_m['Activo'].values else info['p']
            val_act = (info['m'] / info['p']) * p_act
            dif_ars = val_act - info['m']
            dif_pct = ((p_act / info['p']) - 1) * 100
            
            with st.expander(f"📦 {t} ({dif_pct:+.2f}%)", expanded=True):
                st.write(f"**Ganancia:** AR$ {dif_ars:,.2f}")
                st.write(f"**Rendimiento:** {dif_pct:+.2f}%")
                if st.button(f"Vender {t}", key=f"v_{t}"):
                    st.session_state.saldo += val_act
                    del st.session_state.pos[t]
                    guardar_estado()
                    st.rerun()
    else:
        st.info("No hay posiciones abiertas.")
