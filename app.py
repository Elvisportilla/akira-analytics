import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



# ======================
# LOGIN SIMPLE - AKIRA
# ======================

USUARIOS = {
    "akira_admin": "akira2025",
    "direccion": "akira_dir"
}

def login():
    st.title("🔐 Acceso restringido – Akira Sistemas")

    usuario = st.text_input("Usuario")
    clave = st.text_input("Contraseña", type="password")

    if st.button("Ingresar"):
        if usuario in USUARIOS and USUARIOS[usuario] == clave:
            st.session_state["autenticado"] = True
            st.session_state["usuario"] = usuario
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos")

# Control de sesión
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

if not st.session_state["autenticado"]:
    login()
    st.stop()  # ⛔ BLOQUEA todo lo demás





st.set_page_config(
        page_title="Churn de Clientes – Akira Sistemas",
        layout="wide"
    )









# ✅ PEGAR AQUÍ — SOLO UNA VEZ
def preparar_dataset_modelo(df):
    df = df.copy()

    hoy = pd.Timestamp.today().normalize()

    # Variables base
    df["dias_a_vencer"] = (df["fecha_fin"] - hoy).dt.days
    df["antiguedad_meses"] = ((hoy - df["fecha_inicio"]).dt.days / 30).round(1)

    # 🔑 DATASET A NIVEL CLIENTE
    df_cliente = (
        df.groupby("cliente_id")
        .agg(
            dias_a_vencer_min=("dias_a_vencer", "min"),
            antiguedad_meses=("antiguedad_meses", "max"),
            precio_promedio=("precio_unitario", "mean"),
            dispositivos=("dispositivos", "sum"),
            tiene_vigente=("estado_analitico", lambda x: any(x.isin(["Activo", "Por vencer"]))),
            churn=("estado_analitico", lambda x: all(x.isin(["Vencido", "Inactivo"])))
        )
        .reset_index()
    )

    # Variable objetivo CORRECTA
    df_cliente["churn"] = df_cliente["churn"].astype(int)

    X = df_cliente[
        ["dias_a_vencer_min", "antiguedad_meses", "precio_promedio", "dispositivos"]
    ]
    y = df_cliente["churn"]

    return X, y, df_cliente






def calcular_estado_analitico(df, hoy):
    df = df.copy()

    # Días relativos
    df["dias_a_vencer"] = (df["fecha_fin"] - hoy).dt.days
    df["dias_post_vencimiento"] = (hoy - df["fecha_fin"]).dt.days

    condiciones = [
        # 1️⃣ Activo: dentro del periodo contractual
        (df["fecha_inicio"] <= hoy) & (df["fecha_fin"] >= hoy),

        # 2️⃣ Por vencer: faltan entre 0 y 3 días
        (df["dias_a_vencer"] >= 0) & (df["dias_a_vencer"] <= 3),

        # 3️⃣ Vencido: 1 a 5 días después del fin
        (df["dias_post_vencimiento"] >= 1) & (df["dias_post_vencimiento"] <= 5),

        # 4️⃣ Inactivo: más de 5 días vencido
        (df["dias_post_vencimiento"] > 5)
    ]

    estados = [
        "Activo",
        "Por vencer",
        "Vencido",
        "Inactivo"
    ]

    df["estado_analitico"] = np.select(
        condiciones,
        estados,
        default="Inactivo"
    )

    return df



# =============================
# 🔍 EXPLICACIÓN DE PREDICCIÓN (IA)
# =============================
def explicar_prediccion_cliente(X_cliente, modelo):
        coef = modelo.coef_[0]
        variables = X_cliente.columns

        explicacion = pd.DataFrame({
            "Variable": variables,
            "Valor actual": X_cliente.iloc[0].values,
            "Coeficiente modelo": coef,
            "Contribución": X_cliente.iloc[0].values * coef
        })

        explicacion["Impacto absoluto"] = explicacion["Contribución"].abs()

        return explicacion.sort_values("Impacto absoluto", ascending=False)










# =============================
# SIDEBAR
# =============================
st.sidebar.title("📊 Akira Sistemas")
st.sidebar.markdown("### Proyecto de Ciencia de Datos e IA")

menu = st.sidebar.radio(
    "Navegación",
    [
        "Inicio / Contexto",
        "Carga de Datos",
        "Visión General de Clientes",
        "Análisis de Riesgo",
        "Predicción de Deserción (IA)"
    ]
)

# =============================
# ESTADO GLOBAL
# =============================
if "df" not in st.session_state:
    st.session_state.df = None

# =============================
# INICIO
# =============================
if menu == "Inicio / Contexto":
    st.title("📌 Churn de Clientes – Akira Sistemas")

    st.markdown("""
    Este dashboard presenta el análisis progresivo del **comportamiento de clientes**
    en una empresa de servicios digitales por suscripción, culminando en la
    **predicción de deserción mediante Inteligencia Artificial**.
    """)

    st.divider()

    # =============================
    # LOGO CENTRAL (ESTÉTICO)
    # =============================
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <h1 style='text-align:center;'>Akira Sistemas</h1>
            <h3 style='text-align:center; color:gray;'>
            Ciencia de Datos e Inteligencia Artificial
            </h3>
            """,
        unsafe_allow_html=True
)

    st.divider()

    st.markdown(
        """
        ### 🎯 Objetivo del proyecto
        - Analizar el comportamiento histórico de clientes
        - Identificar **riesgos operativos de deserción**
        - Predecir la **probabilidad de churn a nivel cliente**
        - Apoyar la **toma de decisiones estratégicas**
        """
    )

# =============================
# CARGA DE DATOS
# =============================
elif menu == "Carga de Datos":
    st.title("📁 Carga de Datos")

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV o Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is None:
        st.warning("⬆️ Por favor, sube un archivo para continuar.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        st.stop()

    # Normalización
    df.columns = df.columns.str.lower().str.strip()
    for col in df.columns:
        if "fecha" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    st.session_state.df = df

    st.success("✅ Archivo cargado correctamente")
    st.dataframe(df.head(), use_container_width=True)

# =============================
# VISIÓN GENERAL
# =============================
elif menu == "Visión General de Clientes":
    st.title("👥 Visión General de Clientes")

    if st.session_state.df is None:
        st.warning("Primero debes cargar un archivo en la sección 'Carga de Datos'.")
        st.stop()

    df = st.session_state.df.copy()
    hoy = pd.Timestamp.today().normalize()
    df = calcular_estado_analitico(df, hoy)


    df["suscripcion_vigente"] = (
    (df["fecha_inicio"] <= hoy) &
    (df["fecha_fin"] >= hoy)
    )

    df["suscripcion_por_vencer"] = (
        (df["fecha_fin"] - hoy).dt.days.between(0, 3)
    )

    df["suscripcion_no_vigente"] = hoy > df["fecha_fin"]
    



    
  







    # =============================
    # 💰 CÁLCULO DE INGRESOS REALES
    # =============================

    df["meses_contratados"] = (
        (df["fecha_fin"] - df["fecha_inicio"]).dt.days / 30
    ).round().clip(lower=1)

    df["ingreso_total"] = df["precio_unitario"] * df["meses_contratados"]








    

    

    # =============================
    # VARIABLES DE TIEMPO
    # =============================
    df["año"] = df["fecha_inicio"].dt.year
    df["mes"] = df["fecha_inicio"].dt.month

    # =============================
    # FILTROS GLOBALES
    # =============================
    st.sidebar.markdown("### 🔎 Filtros de análisis")

    estado_sel = st.sidebar.selectbox(
        "Estado de suscripción",
        ["Todos", "Vigentes", "Por vencer", "No vigentes"]
    )

    año_sel = st.sidebar.selectbox(
        "Año",
        sorted(df["año"].dropna().unique())
    )

    mes_sel = st.sidebar.selectbox(
        "Mes",
        ["Todos"] + list(range(1, 13))
    )

    df = df[df["año"] == año_sel]
    if mes_sel != "Todos":
        df = df[df["mes"] == mes_sel]
    
    if estado_sel != "Todos":
        if estado_sel == "Vigentes":
            df = df[df["suscripcion_vigente"]]

        elif estado_sel == "Por vencer":
            df = df[df["suscripcion_por_vencer"]]

        elif estado_sel == "No vigentes":
            df = df[df["suscripcion_no_vigente"]]

    # =============================
    # 🔍 BÚSQUEDA DE CLIENTE
    # =============================
    st.sidebar.markdown("### 👤 Selección de cliente")

    clientes_lista = (
        df[["cliente_id", "nombre", "codigo_cliente"]]
        .drop_duplicates()
        .sort_values("nombre")
    )

    cliente_sel = st.sidebar.selectbox(
        "Cliente",
        ["Todos"] + clientes_lista["nombre"].tolist()
    )

    codigo_sel = st.sidebar.selectbox(
        "Código",
        ["Todos"] + clientes_lista["codigo_cliente"].dropna().tolist()
    )

    df_filtro = df.copy()

    if cliente_sel != "Todos":
        df_filtro = df_filtro[df_filtro["nombre"] == cliente_sel]

    if codigo_sel != "Todos":
        df_filtro = df_filtro[df_filtro["codigo_cliente"] == codigo_sel]


    # =============================
    # PERFIL DESCRIPTIVO DEL CLIENTE
    # =============================
        # =============================
    clientes_unicos = df_filtro["cliente_id"].nunique()

    if clientes_unicos == 1:
        cliente = df_filtro.iloc[0]   # 👈 CLAVE

        st.info("🔍 Análisis descriptivo del cliente seleccionado")

        c1, c2, c3 = st.columns(3)
        c1.metric("Cliente", cliente["nombre"])
        c2.metric("Código", cliente.get("codigo_cliente", "N/D"))
        c3.metric("Estado actual", cliente["estado_analitico"])


            # =============================
        # 📦 PRODUCTOS VIGENTES DEL CLIENTE
        # =============================
        productos_vigentes = (
            df_filtro
            .loc[df_filtro["suscripcion_vigente"], "producto"]
            .dropna()
            .unique()
            .tolist()
        )

        if productos_vigentes:
            productos_texto = ", ".join(productos_vigentes)
        else:
            productos_texto = "No tiene productos vigentes"





        precio_total_cliente = df_filtro["ingreso_total"].sum()

        st.markdown(f"""
        **Productos vigentes:** {productos_texto}  
        **Ingreso total del cliente:** S/ {precio_total_cliente:,.2f}  
        **Fecha de inicio más antigua:** {df_filtro["fecha_inicio"].min().date()}  
        **Fecha de fin más reciente:** {df_filtro["fecha_fin"].max().date()}  
        """)

    
    
    # =================================================
    # DATASET CONSOLIDADO A NIVEL CLIENTE (1 FILA = 1 CLIENTE)
    # =================================================
   

    df_cliente_filtro = (
        df_filtro
        .sort_values("fecha_fin", ascending=False)
        .groupby("cliente_id", as_index=False)
        .first()
    )




   # =============================
    # KPIs NUEVOS (CLIENTES vs SUSCRIPCIONES)
    # =============================

    # ---- KPIs DE CLIENTES ----
    clientes_estado = (
    df_filtro.groupby("cliente_id")
    .agg(tiene_vigente=("suscripcion_vigente", "any"))
    .reset_index()
    )

    total_clientes = clientes_estado["cliente_id"].nunique()
    clientes_activos = clientes_estado["tiene_vigente"].sum()
    clientes_inactivos = total_clientes - clientes_activos

    # ---- KPIs DE SUSCRIPCIONES ----
    total_suscripciones = len(df_filtro)
    suscripciones_vigentes = df_filtro["suscripcion_vigente"].sum()
    suscripciones_no_vigentes = df_filtro["suscripcion_no_vigente"].sum()
    suscripciones_por_vencer = df_filtro["suscripcion_por_vencer"].sum()

    # ---- INGRESOS SOLO VIGENTES ----
    if df_filtro["cliente_id"].nunique() == 1:
        ingresos_vigentes = df_filtro.loc[
            df_filtro["suscripcion_vigente"],
            "ingreso_total"
        ].sum()
    else:
        ingresos_vigentes = df_filtro.loc[
            df_filtro["suscripcion_vigente"],
            "ingreso_total"
        ].sum()

   


    # ---- KPIs EN UI ----
    k1, k2, k3 = st.columns(3)

    k1.metric("👤 Total Clientes", total_clientes)
    k2.metric("🟢 Clientes Activos", clientes_activos)
    k3.metric("🔴 Clientes Inactivos", clientes_inactivos)

    st.markdown("---")

    # =============================
    # KPIs – FILA 2 (SUSCRIPCIONES)
    # =============================
    k4, k5, k6, k7 = st.columns(4)

    k4.metric("📦 Total Suscripciones", total_suscripciones)
    k5.metric("✅ Vigentes", suscripciones_vigentes)
    k6.metric("❌ No vigentes", suscripciones_no_vigentes)
    k7.metric("⏳ Por vencer (≤3 días)", suscripciones_por_vencer)

    st.markdown("---")

    # =============================
    # KPIs – FILA 3 (INGRESOS)
    # =============================
    k8, _ = st.columns([1, 3])
    k8.metric("💰 Ingresos vigentes (S/)", f"{ingresos_vigentes:,.2f}")
    if df_filtro["cliente_id"].nunique() == 1 and ingresos_vigentes == 0:
        st.info("ℹ️ El cliente seleccionado no tiene suscripciones vigentes.")











    # =============================
    # ⏳ INDICADORES DE PERMANENCIA
    # =============================
    antiguedad_cliente = (
    df_filtro.groupby("cliente_id")["fecha_inicio"].min()
    )

    if not antiguedad_cliente.empty:
        antiguedad_promedio = (
            (hoy - antiguedad_cliente).dt.days / 30
        ).mean().round(1)
    else:
        antiguedad_promedio = 0


    if df_filtro["cliente_id"].nunique() == 1:
        st.caption("📌 Antigüedad calculada solo para el cliente seleccionado.")

    st.subheader("⏳ Indicadores de permanencia del cliente")

    if df_filtro["cliente_id"].nunique() == 1 and len(df_filtro) == 1:
        st.info("ℹ️ La antigüedad promedio no es representativa para un único contrato.")
    else:
        c_ant_1, c_ant_2 = st.columns([1, 4])

        with c_ant_1:
            st.metric(
                "Antigüedad promedio",
                f"{antiguedad_promedio:.1f} meses"
            )

        with c_ant_2:
            st.caption(
                "Tiempo promedio que los clientes permanecen desde el inicio "
                "de su suscripción. Un valor bajo suele estar asociado a mayor churn."
            )

    st.divider()

    # =============================
    # 📊 GRÁFICOS PRINCIPALES
    # =============================
    col_left, col_right = st.columns([1, 1], gap="large")

    # ==================================================
    # 🔴 IZQUIERDA: DISTRIBUCIÓN DE SUSCRIPCIONES POR ESTADO
    # ==================================================
    with col_left:
    # =============================
# 📌 DISTRIBUCIÓN DE SUSCRIPCIONES POR ESTADO
# =============================

        st.subheader("📌 Distribución de suscripciones por estado")

        estado_suscripciones = (
        df_filtro.assign(
            Estado=np.select(
                [
                    df_filtro["suscripcion_vigente"],
                    df_filtro["suscripcion_por_vencer"],
                    df_filtro["suscripcion_no_vigente"]
                ],
                ["Vigentes", "Por vencer", "No vigentes"],
                default="No vigentes"
            )
        )
        .groupby("Estado")
        .size()
        .reset_index(name="Cantidad")
        )

        total_suscripciones_estado = estado_suscripciones["Cantidad"].sum()

        if total_suscripciones_estado > 0:

            # 1️⃣ CREAR FIGURA
            fig_pie = px.pie(
                estado_suscripciones,
                names="Estado",
                values="Cantidad",
                hole=0.45,
                color="Estado",
                color_discrete_map={
                    "Vigentes": "#2ECC71",
                    "Por vencer": "#F1C40F",
                    "No vigentes": "#E74C3C"
                }
            )

            # 2️⃣ AJUSTES DE TRAZOS
            fig_pie.update_traces(
                textinfo="percent+value",
                textposition="inside"
            )

            # 3️⃣ AJUSTES DE LAYOUT (LEYENDA BIEN UBICADA)
            fig_pie.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=0.62,
                    font=dict(size=12)
                )
            )

            # 4️⃣ MOSTRAR
            st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.info("No hay suscripciones para mostrar.")

    # ==================================================
    # 🟢 DERECHA: RANKING DE PRODUCTOS POR SUSCRIPCIONES
    # ==================================================
    with col_right:
        st.subheader("💰 Ranking de productos por suscripciones contratadas")

    # 🔑 CLAVE: 1 fila = 1 suscripción
    ranking_suscripciones = (
        df_filtro
        .groupby("producto")
        .size()
        .reset_index(name="cantidad_suscripciones")
        .sort_values("cantidad_suscripciones", ascending=False)
    )

    if ranking_suscripciones.empty:
        st.info("ℹ️ No hay datos suficientes para el ranking de productos.")
    else:
        fig_bar = px.bar(
            ranking_suscripciones,
            x="producto",
            y="cantidad_suscripciones",
            text="cantidad_suscripciones",
            labels={
                "producto": "Producto",
                "cantidad_suscripciones": "Cantidad de suscripciones"
            },
            color="producto",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=380,
            showlegend=False,
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig_bar, use_container_width=True)


        # =============================
    # 📈 EVOLUCIÓN TEMPORAL
    # =============================


    


    st.subheader("💰 Evolución mensual de ingresos (análisis económico real)")

    if df_filtro["cliente_id"].nunique() == 1:
        st.info("ℹ️ La evolución mensual se muestra solo para análisis agregados.")
    else:
        ingresos_mensuales = (
            df_filtro
            .groupby(pd.Grouper(key="fecha_inicio", freq="M"))
            .agg(
                ingreso_total=("ingreso_total", "sum"),
                clientes_activos=("cliente_id", "nunique"),
                precio_unitario_prom=("precio_unitario", "mean")
            )
        )

        ingresos_mensuales["arpu"] = (
            ingresos_mensuales["ingreso_total"] /
            ingresos_mensuales["clientes_activos"].replace(0, np.nan)
        ).round(2)

        fig = px.line(
            ingresos_mensuales,
            y=["ingreso_total", "arpu", "precio_unitario_prom"],
            markers=True,
            labels={
                "value": "Monto (S/)",
                "variable": "Indicador"
            }
        )

        fig.update_layout(
            height=450,
            legend_title_text="Indicador",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)



    st.caption(
        "El ingreso total considera el valor completo de los contratos. "
        "El ARPU refleja el ingreso promedio por cliente activo. "
        "Las anotaciones señalan picos explicados por concentración de ventas."
    )




    # =============================
    # 📋 TABLA FINAL
    # =============================
    st.subheader("📋 Vista resumida de clientes")

    st.dataframe(
    df_filtro[
        [
            "cliente_id",
            "codigo_cliente",
            "nombre",
            "producto",
            "estado_analitico",
            "precio_unitario",
            "ingreso_total",      # 👈 NUEVO
            "fecha_inicio",
            "fecha_fin"
        ]
    ]
    .rename(columns={
        "precio_unitario": "Precio unitario (S/)",
        "ingreso_total": "Precio total (S/)",
        "fecha_inicio": "Fecha inicio",
        "fecha_fin": "Fecha fin"
    })
    .sort_values("Fecha inicio"),
    use_container_width=True
    )

# =============================
# ANÁLISIS DE RIESGO
# =============================
elif menu == "Análisis de Riesgo":
    st.title("⚠️ Análisis de Riesgo de Deserción (Pre-IA)")

    if st.session_state.df is None:
        st.warning("Primero debes cargar datos.")
        st.stop()

    df = st.session_state.df.copy()
    hoy = pd.Timestamp.today().normalize()
    df = calcular_estado_analitico(df, hoy)

    # 🔑 CLAVE ABSOLUTA
    

    df["dias_a_vencer"] = (df["fecha_fin"] - hoy).dt.days
    df["antiguedad_meses"] = ((hoy - df["fecha_inicio"]).dt.days / 30).round(1)

  










        # =============================
    # CLASIFICACIÓN DE RIESGO (PRE-IA)
    # =============================
    def riesgo_churn(row):
        if row["estado_analitico"] in ["Vencido", "Inactivo"]:
            return "Alto"
        if row["dias_a_vencer"] <= 7:
            return "Alto"
        if row["antiguedad_meses"] <= 2:
            return "Medio"
        return "Bajo"

    df["nivel_riesgo"] = df.apply(riesgo_churn, axis=1)

    df["orden_riesgo"] = df["nivel_riesgo"].map({
    "Alto": 1,
    "Medio": 2,
    "Bajo": 3
    })



    # =============================
    # 🔎 FILTROS – ANÁLISIS DE RIESGO
    # =============================
    st.sidebar.markdown("### 🔎 Filtros de riesgo")

    riesgo_sel = st.sidebar.multiselect(
        "Tipo de riesgo",
        ["Alto", "Medio", "Bajo"],
        default=["Alto", "Medio", "Bajo"]
    )

    clientes_lista = (
        df[["cliente_id", "nombre", "codigo_cliente"]]
        .drop_duplicates()
        .sort_values("nombre")
    )

    nombre_sel = st.sidebar.selectbox(
        "Cliente",
        ["Todos"] + clientes_lista["nombre"].tolist()
    )

    codigo_sel = st.sidebar.selectbox(
        "Código",
        ["Todos"] + clientes_lista["codigo_cliente"].dropna().tolist()
    )

    df_riesgo = df[df["nivel_riesgo"].isin(riesgo_sel)]

    if nombre_sel != "Todos":
        df_riesgo = df_riesgo[df_riesgo["nombre"] == nombre_sel]

    if codigo_sel != "Todos":
        df_riesgo = df_riesgo[df_riesgo["codigo_cliente"] == codigo_sel]











    # =============================
    # KPIs DE RIESGO
    # =============================
    total = df_riesgo["cliente_id"].nunique()
    alto = df_riesgo[df_riesgo["nivel_riesgo"] == "Alto"]["cliente_id"].nunique()
    medio = df_riesgo[df_riesgo["nivel_riesgo"] == "Medio"]["cliente_id"].nunique()
    bajo = df_riesgo[df_riesgo["nivel_riesgo"] == "Bajo"]["cliente_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes evaluados", total)
    c2.metric("Riesgo alto", alto)
    c3.metric("Riesgo medio", medio)
    c4.metric("Riesgo bajo", bajo)

    st.divider()

    # =============================
    # DISTRIBUCIÓN DE RIESGO
    # =============================
    st.subheader("📊 Distribución de clientes por nivel de riesgo")

    riesgo_dist = (
        df_riesgo.groupby("nivel_riesgo")["cliente_id"]
        .nunique()
        .reset_index(name="clientes")
    )

    fig_riesgo = px.pie(
        riesgo_dist,
        names="nivel_riesgo",
        values="clientes",
        hole=0.45,
        color_discrete_map={
            "Alto": "#E74C3C",
            "Medio": "#F1C40F",
            "Bajo": "#2ECC71"
        }
    )

    fig_riesgo.update_traces(textinfo="percent+value")
    st.plotly_chart(fig_riesgo, use_container_width=True)

    st.divider()

    # =============================
    # RIESGO VS ANTIGÜEDAD
    # =============================
    st.subheader("⏳ Riesgo de churn según antigüedad")

    fig_ant = px.box(
        df_riesgo,
        x="nivel_riesgo",
        y="antiguedad_meses",
        color="nivel_riesgo",
        color_discrete_map={
            "Alto": "#E74C3C",
            "Medio": "#F1C40F",
            "Bajo": "#2ECC71"
        }
    )

    st.plotly_chart(fig_ant, use_container_width=True)

    st.caption(
        "El gráfico muestra la relación entre la antigüedad del cliente y su nivel de riesgo de churn. "
        "Clientes con menor permanencia presentan mayor dispersión y riesgo elevado."
    )

    st.divider()

    # =============================
    # RIESGO POR PRODUCTO
    # =============================
    st.subheader("🧩 Productos con mayor concentración de riesgo")

    riesgo_producto = (
        df_riesgo[df_riesgo["nivel_riesgo"] == "Alto"]
        .groupby("producto")["cliente_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="clientes_en_riesgo")
    )

    fig_prod = px.bar(
        riesgo_producto,
        x="producto",
        y="clientes_en_riesgo",
        text="clientes_en_riesgo",
        color="producto"
    )

    fig_prod.update_traces(textposition="outside")
    fig_prod.update_layout(showlegend=False)

    st.plotly_chart(fig_prod, use_container_width=True)

    st.divider()

    # =============================
    # TABLA DE PRIORIZACIÓN
    # =============================
    st.subheader("📋 Clientes prioritarios para intervención")

    tabla_riesgo = (
        df_riesgo[
            [
                "cliente_id",
                "nombre",
                "producto",
                "estado_analitico",
                "fecha_inicio",     # 👈 NUEVO
                "fecha_fin",        # 👈 NUEVO
                "dias_a_vencer",
                "antiguedad_meses",
                "nivel_riesgo",
                "orden_riesgo"
            ]
        ]
    .sort_values(
        by=["orden_riesgo", "dias_a_vencer"],
        ascending=[True, True]
    )
)


    st.dataframe(tabla_riesgo, use_container_width=True)

    st.caption(
        "Este análisis heurístico identifica clientes prioritarios antes de aplicar "
        "el modelo predictivo de Inteligencia Artificial."
    )

# =============================
# PREDICCIÓN DE DESERCIÓN (IA)
# =============================
elif menu == "Predicción de Deserción (IA)":
    st.title("🤖 Predicción de Deserción (Machine Learning)")

    if st.session_state.df is None:
        st.warning("Primero debes cargar datos.")
        st.stop()

    # =============================
    # PREPARACIÓN DE DATOS
    # =============================
    df = st.session_state.df.copy()
    hoy = pd.Timestamp.today().normalize()
    df = calcular_estado_analitico(df, hoy)

    df["dias_a_vencer"] = (df["fecha_fin"] - hoy).dt.days
    df["antiguedad_meses"] = ((hoy - df["fecha_inicio"]).dt.days / 30).round(1)

    # =============================
    # DATASET A NIVEL CLIENTE
    # =============================
    X, y, df_modelo = preparar_dataset_modelo(df)

    # =============================
    # ENTRENAMIENTO DEL MODELO
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # =============================
    # MÉTRICAS DEL MODELO
    # =============================
    st.success("Modelo entrenado correctamente")

    c1, c2 = st.columns(2)
    c1.metric("Clientes evaluados", len(df_modelo))
    c2.metric("AUC del modelo", f"{auc:.3f}")

    st.divider()

    # =============================
    # IMPORTANCIA DE VARIABLES
    # =============================
    coeficientes = pd.DataFrame({
        "Variable": X.columns,
        "Impacto": modelo.coef_[0]
    }).sort_values("Impacto", ascending=False)

    st.subheader("📊 Variables que explican la deserción")

    fig_coef = px.bar(
        coeficientes,
        x="Impacto",
        y="Variable",
        orientation="h",
        color="Impacto",
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # =============================
    # FILTROS – PREDICCIÓN INDIVIDUAL
    # =============================
    st.sidebar.markdown("### 👤 Predicción individual")

    clientes_ordenados = (
        df[["cliente_id", "nombre", "codigo_cliente"]]
        .drop_duplicates()
        .sort_values("nombre")
    )

    nombre_sel = st.sidebar.selectbox(
        "Selecciona cliente",
        ["Todos"] + clientes_ordenados["nombre"].tolist()
    )

    codigo_sel = st.sidebar.selectbox(
        "Código del cliente",
        ["Todos"] + clientes_ordenados["codigo_cliente"].dropna().tolist()
    )

    df_pred = df.copy()

    if nombre_sel != "Todos":
        df_pred = df_pred[df_pred["nombre"] == nombre_sel]

    if codigo_sel != "Todos":
        df_pred = df_pred[df_pred["codigo_cliente"] == codigo_sel]

    # =============================
    # PREDICCIÓN INDIVIDUAL (CORRECTA)
    # =============================
    if df_pred["cliente_id"].nunique() == 1:

        cliente_nombre = df_pred.iloc[0]["nombre"]

        df_cliente = (
            df_pred
            .groupby("cliente_id")
            .agg(
                dias_a_vencer_min=("dias_a_vencer", "min"),
                antiguedad_meses=("antiguedad_meses", "max"),
                precio_promedio=("precio_unitario", "mean"),
                dispositivos=("dispositivos", "sum")
            )
        )

        X_cliente_scaled = scaler.transform(df_cliente)
        prob = modelo.predict_proba(X_cliente_scaled)[0][1]

        st.subheader("🎯 Resultado de predicción individual")

        if prob < 0.30:
            nivel_riesgo_ml = "🟢 Riesgo Bajo"
        elif prob < 0.60:
            nivel_riesgo_ml = "🟡 Riesgo Medio"
        else:
            nivel_riesgo_ml = "🔴 Riesgo Alto"

        c1, c2, c3 = st.columns([2, 1.5, 1.5])

        with c1:
            st.metric("Cliente", cliente_nombre)

        with c2:
            st.metric("Probabilidad de churn", f"{prob*100:.1f}%")

        with c3:
            st.metric("Nivel de riesgo estimado", nivel_riesgo_ml)

        st.caption(
            "🔎 **Umbrales definidos por negocio:** "
            "Bajo < 30% · Medio 30–60% · Alto > 60%. "
            "La probabilidad se estima a partir del comportamiento agregado del cliente."
        )

        # =============================
        # 📋 VARIABLES PREDICTIVAS (EXPLICABLE)
        # =============================
        st.subheader("📋 Variables predictivas utilizadas por el modelo")

        X_cliente_df = pd.DataFrame(
            df_cliente.values,
            columns=X.columns
        )

        explicacion = explicar_prediccion_cliente(X_cliente_df, modelo)

        st.dataframe(
            explicacion[
                ["Variable", "Valor actual", "Coeficiente modelo", "Contribución"]
            ].round(3),
            use_container_width=True
        )

        st.caption(
            "Cada variable muestra su contribución positiva o negativa "
            "a la probabilidad de deserción. Vista explicable para jurado."
        )

    else:
        st.info("Selecciona un único cliente para obtener la predicción.")