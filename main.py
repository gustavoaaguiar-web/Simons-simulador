import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# --- CONFIGURACIÓN DE CORREO (RenTech Method) ---
MI_MAIL = "gustavoaaguiar99@gmail.com"
CLAVE_APP = "oshrmhfqzvabekzt" 

# --- CONFIGURACIÓN APP ---
st.set_page_config(page_title="Simons GG 11", page_icon="🦅", layout="wide")

CAPITAL_INICIAL = 30000000.0
SALDO_ACTUAL = 33362112.69 

def enviar_alerta_mail(asunto, cuerpo):
    msg = EmailMessage()
    msg.set_content(cuerpo)
    msg['Subject'] = asunto
    msg['From'] = MI_MAIL
    msg['To'] = MI_MAIL
    try:
        # Configuración para Gmail
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(MI_MAIL, CLAVE_APP)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.sidebar.error(f"Error de envío: {e}")
        return False

# --- INTERFAZ PRINCIPAL ---
st.title("🦅Simons GG 11 🤑")

# Cálculo de rendimiento de cartera
rendimiento_total = ((SALDO_ACTUAL / CAPITAL_INICIAL) - 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Patrimonio Total", f"AR$ {SALDO_ACTUAL:,.2f}", f"{rendimiento_total:+.2f}%")
c2.metric("Efectivo disponible", f"AR$ {SALDO_ACTUAL:,.2f}")
c3.metric("Ticket sugerido (8%)", f"AR$ {(SALDO_ACTUAL * 0.08):,.2f}")

st.divider()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("🛠 Panel de Control")

# BOTÓN DE TESTEO DE MAIL
if st.sidebar.button("🧪 ENVIAR MAIL DE TEST"):
    test_cuerpo = f"¡Conexión Exitosa!\n\nEl bot Simons GG 11 ya puede enviarte alertas a {MI_MAIL}.\nEstado de cartera: {rendimiento_total:+.2f}%"
    if enviar_alerta_mail("🦅 Test de Conexión Simons", test_cuerpo):
        st.sidebar.success("✅ Mail de prueba enviado.")
    else:
        st.sidebar.error("❌ Falló el envío.")

# --- MONITOR DE MERCADO (14 ACTIVOS) ---
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
            # Tickers Locales
            tk_ars = "YPFD.BA" if t=='YPF' else ("PAMP.BA" if t=='PAM' else f"{t}.BA")
            
            h_usd = yf.download(t, period="3mo", interval="1d", progress=False)
            h_ars = yf.download(tk_ars, period="1d", interval="1m", progress=False)
            
            if h_usd.empty or h_ars.empty: continue

            p_u = float(h_usd.Close.iloc[-1])
            p_a = float(h_ars.Close.iloc[-1])
            ccl = (p_a * r) / p_u
            ccls.append(ccl)
            
            # Algoritmo Markov
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
        if desvio < -0.0065 and row['Clima'] == "🟢": row['Señal'] = "🟢 COMPRA"
        elif desvio > 0.0065: row['Señal'] = "🔴 VENTA"
        else: row['Señal'] = "⚖️ MANTENER"
        return row

    df_final = df_res.apply(procesar, axis=1)
    
    st.dataframe(
        df_final[['Activo', 'CCL', 'Clima', 'Señal', 'Desvío %', 'ARS', 'USD']]
        .style.applymap(lambda x: 'background-color: #004d00; color: white' if 'COMPRA' in str(x) else ('background-color: #4d0000; color: white' if 'VENTA' in str(x) else ''), subset=['Señal']), 
        use_container_width=True, hide_index=True
    )
    
    # --- ALERTAS DE ARBITRAJE ---
    alertas = df_final[df_final['Señal'].str.contains("COMPRA|VENTA")]
    
    if not alertas.empty:
        st.sidebar.subheader("🚀 Alertas de Arbitraje")
        if st.sidebar.button("📧 ENVIAR SEÑALES AL MAIL"):
            cuerpo = f"🦅 INFORME DE ARBITRAJE - SIMONS GG 11\n"
            cuerpo += f"Rendimiento Cartera: {rendimiento_total:+.2f}%\n"
            cuerpo += "="*30 + "\n"
            for _, r in alertas.iterrows():
                cuerpo += f"ACTIVO: {r['Activo']} -> {r['Señal']}\n"
                cuerpo += f"Desvío: {r['Desvío %']} | Clima: {r['Clima']}\n"
                cuerpo += "-"*10 + "\n"
            
            if enviar_alerta_mail(f"🦅 Alerta: {len(alertas)} señales detectadas", cuerpo):
                st.sidebar.success("Informe enviado.")
else:
    st.warning("Conectando con el mercado...")
