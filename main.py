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
st.set_page_config(page_title="Simons GG v13.8", page_icon="🦅", layout="wide")
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

# --- CORRECCIÓN DE ZONA HORARIA ---
try:
    arg_tz = pytz.timezone('America/Argentina/Buenos_Aires')
except:
    arg_tz = pytz.timezone('Etc/GMT+3') # Backup por si falla el string

ahora_arg_dt = datetime.now(arg_tz)
ahora_arg_time = ahora_arg_dt.time()
dia_semana = ahora_arg_dt.weekday()

# --- PERSISTENCIA ---
ARCHIVO_ESTADO = "simons_state.json"
SALDO_CORRECTO = 34130883.81

def cargar_estado():
    if os.path.exists(ARCHIVO_ESTADO):
        try:
            with open(ARCHIVO_ESTADO, "r") as f:
                data = json.load(f)
                # Si el saldo es viejo y no hay posiciones, reseteamos al valor actual
                if not data.get("pos"):
                    data["saldo"] = SALDO_CORRECTO
                return data
        except: pass
    return {"saldo": SALDO_CORRECTO, "pos": {}, "notificados": []}

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
            if not es_test: st.session_state.notificados.append(op_id)
            return True
        except: return False
    return False

# --- LÓGICA DE HORARIOS ---
es_dia_habil = 0 <= dia_semana <= 4
hora_apertura = time(11, 0)
hora_limite_compra = time(16, 30)
hora_cierre_total = time(16, 50)

mercado_abierto = es_dia_habil and (hora_apertura <= ahora_arg_time < hora_cierre_total)
puedo_comprar_auto = es_dia_habil and (hora_apertura <= ahora_arg_time < hora_limite_compra)
es_hora_de_cierre = es_dia_habil and (ahora_arg_time >= hora_cierre_total)

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

# --- CÁLCULOS Y TRADING ---
valor_cedears = 0.0
for t, info in st.session_state.pos.items():
    p_act_fila = df_m.loc[df_m['Activo'] == t, 'ARS'].values
    p_hoy = p_act_fila[0] if len(p_act_fila) > 0 and p_act_fila[0] > 0 else info['p']
    valor_cedears += (info['m'] / info['p']) * p_hoy

patrimonio_total = st.session_state.saldo + valor_cedears
rendimiento_total = ((patrimonio_total / 30000000.0) - 1) * 100

# CIERRE AUTOMÁTICO 16:50 OBLIGATORIO
if es_hora_de_cierre and st.session_state.pos:
    for activo in list(st.session_state.pos.keys()):
        info_c = st.session_state.pos[activo]
        p_act_f = df_m.loc[df_m['Activo'] == activo, 'ARS'].values
        p_venta = p_act_f[0] if len(p_act_f) > 0 else info_c['p']
        v_final = (info_c['m'] / info_c['p']) * p_venta
        st.session_state.saldo += v_final
        del st.session_state.pos[activo]
        enviar_alerta_operacion(f"⚠️ CIERRE 16:50: {activo}", f"Resultado final: AR$ {v_final - info_c['m']:,.2f}", f"panic_{activo}")
    guardar_estado()
    st.rerun()

# LÓGICA DE TRADING
if ccl_m and mercado_abierto:
    for _, row in df_m.iterrows():
        if np.isnan(row['CCL']): continue
        desvio = (row['CCL'] / ccl_m) - 1
        activo = row['Activo']
        
        if desvio >= 0.005 and activo in st.session_state.pos:
            info_c = st.session_state.pos[activo]
            v_final = (info_c['m'] / info_c['p']) * row['ARS']
            st.session_state.saldo += v_final
            del st.session_state.pos[activo]
            enviar_alerta_operacion(f"🦅 VENTA: {activo}", f"Ganancia: AR$ {v_final - info_c['m']:,.2f}", f"v_{activo}_{ahora_arg_dt.minute}")
            guardar_estado()
            st.rerun()
        
        if puedo_comprar_auto and desvio <= -0.005 and row['Clima'] == "🟢" and activo not in st.session_state.pos:
            monto_op = patrimonio_total * 0.125
            if st.session_state.saldo >= monto_op:
                st.session_state.saldo -= monto_op
                st.session_state.pos[activo] = {'m': monto_op, 'p': row['ARS']}
                enviar_alerta_operacion(f"🦅 COMPRA: {activo}", f"Precio: ${row['ARS']}", f"c_{activo}_{ahora_arg_dt.minute}")
                guardar_estado()
                st.rerun()

# --- INTERFAZ ---
st.title("🦅 Simons GG v13.8 🤑")

# Mostrar Estado
if not es_dia_habil: est = "🔴 FIN DE SEMANA"
elif ahora_arg_time < hora_apertura: est = f"🔴 CERRADO (Abre 11:00)"
elif es_hora_de_cierre: est = "🔴 CERRADO (Post-mercado)"
else: est = "🟢 OPERANDO"
st.caption(f"ARG: {ahora_arg_dt.strftime('%H:%M:%S')} | {est}")

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {patrimonio_total:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo", f"AR$ {st.session_state.saldo:,.2f}")
c3.metric("Valor Cedears", f"AR$ {valor_cedears:,.2f}")

st.divider()

if ccl_m:
    df_m['%'] = df_m['CCL'].apply(lambda x: f"{((x/ccl_m)-1)*100:+.2f}%")
    df_m['Señal'] = df_m.apply(lambda r: "🟢 COMPRA" if ((r['CCL']/ccl_m)-1) <= -0.005 and r['Clima'] == "🟢" else ("🔴 VENTA" if ((r['CCL']/ccl_m)-1) >= 0.005 else "⚖️ MANTENER"), axis=1)
    if not mercado_abierto: st.warning("⚠️ Modo Consulta activo.")
    st.dataframe(df_m[['Activo', '%', 'Clima', 'Señal', 'ARS', 'USD']], use_container_width=True, hide_index=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Cartera")
    if st.button("🧪 Test Mail"):
        enviar_alerta_operacion("🦅 TEST", "OK", "test", es_test=True)
    
    st.divider()
    if st.session_state.pos:
        for t, info in list(st.session_state.pos.items()):
            p_a = df_m.loc[df_m['Activo'] == t, 'ARS'].values[0] if t in df_m['Activo'].values else info['p']
            v_a = (info['m'] / info['p']) * p_a
            gan = v_a - info['m']
            with st.expander(f"📦 {t}", expanded=True):
                st.write(f"Ganancia: AR$ {gan:,.2f}")
                if st.button(f"🔴 Vender {t}", key=f"manual_{t}"):
                    st.session_state.saldo += v_a
                    del st.session_state.pos[t]
                    guardar_estado()
                    st.rerun()
    else:
        st.info("Cartera vacía.")
