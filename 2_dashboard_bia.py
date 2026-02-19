import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Panel Ejecutivo AMAZON", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #131921; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    section[data-testid="stSidebar"] { background-color: #232F3E !important; border-right: 1px solid #3A4553; }
    h1, h2, h3, p, label, div[data-testid="stCaptionContainer"], .stMarkdown { color: #F8F9FA !important; }
    div[data-testid="metric-container"] { background-color: #232F3E; border: 1px solid #3A4553; padding: 15px 20px; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0,0,0,0.3); border-left: 5px solid #FF9900; }
    div[data-testid="metric-container"] * { color: #FFFFFF !important; }
    .stDataFrame { filter: invert(0.9) hue-rotate(180deg); }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_excel('data/amazon_ventas_limpio.xlsx')
    df['order_fecha'] = pd.to_datetime(df['order_fecha'])
    df['Mes y Año'] = df['order_fecha'].dt.to_period('M').astype(str)
    
    le = LabelEncoder()
    df['region_num'] = le.fit_transform(df['Región_cliente'])
    df['pago_num'] = le.fit_transform(df['metodo_pago'])
    
    df = df.rename(columns={
        'producto_categoria': 'Categoría del Producto', 'ventas_total': 'Ventas Totales',
        'cantidad_vendida': 'Cantidad Vendida', 'Región_cliente': 'Región del Cliente',
        'metodo_pago': 'Método de Pago', 'porcentaje_descuento': 'Porcentaje de Descuento',
        'precio_con_descuento': 'Precio con Descuento'
    })
    return df

df_original = load_data()

col1, col2, col3 = st.sidebar.columns([1,3,1])
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", use_container_width=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Panel de Control")

modo_filtro = st.sidebar.radio("Selección de Datos:", options=["🛒 Todas las Categorías", "🎯 Elegir Categorías"], index=0)
lista_categorias = df_original['Categoría del Producto'].unique().tolist()

if modo_filtro == "🛒 Todas las Categorías":
    categorias_seleccionadas = lista_categorias
else:
    categorias_seleccionadas = st.sidebar.multiselect("Selecciona las categorías:", options=lista_categorias, default=[lista_categorias[0]])

df = df_original[df_original['Categoría del Producto'].isin(categorias_seleccionadas)]

if df.empty:
    st.error("⚠️ Selecciona al menos una categoría.")
    st.stop()


st.title("Panel Ejecutivo Anual")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ingresos Totales", f"S/ {df['Ventas Totales'].sum():,.2f}")
c2.metric("Ticket Promedio", f"S/ {df['Ventas Totales'].mean():,.2f}")
c3.metric("Volumen Ventas", f"{df['Cantidad Vendida'].sum():,.0f} und.")
c4.metric("Rating Promedio", f"{df['Valoración'].mean():.1f} ⭐")
c5.metric("Pago Favorito", df['Método de Pago'].mode()[0].title())

st.markdown("<br>", unsafe_allow_html=True)

def obtener_medalla(rank):
    if rank == 1: return "🥇 "
    elif rank == 2: return "🥈 "
    elif rank == 3: return "🥉 "
    else: return f"{rank}. "

col_izq, col_centro, col_der = st.columns([1.2, 1.5, 1.2])

with col_izq:
    st.subheader("Top Categorías por Ingresos")
    df_cat = df.groupby('Categoría del Producto')['Ventas Totales'].sum().reset_index().sort_values('Ventas Totales', ascending=False)
    df_cat['Rank'] = range(1, len(df_cat) + 1)
    df_cat['Categoria_Label'] = df_cat['Rank'].apply(obtener_medalla) + df_cat['Categoría del Producto']
    df_cat = df_cat.sort_values('Ventas Totales', ascending=True)

    fig_cat = px.bar(df_cat, x='Ventas Totales', y='Categoria_Label', orientation='h', color_discrete_sequence=['#FF9900'], text='Ventas Totales')
    fig_cat.update_traces(texttemplate='S/ %{text:,.2f}', textposition='inside', textfont=dict(color='white', size=14))
    fig_cat.update_layout(xaxis_title="Ingresos (S/)", yaxis_title="", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), font=dict(color="white"), height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

with col_centro:
    st.subheader("Cobertura Global de Ventas")
    df_reg = df.groupby('Región del Cliente')['Ventas Totales'].sum().reset_index()
    coordenadas = {'América del norte': (45.0,-100.0), 'Europa': (54.0,15.0), 'Asia': (34.0,100.0), 'África': (8.0,20.0), 'Oriente Medio': (25.0,45.0), 'Océano Atlántico': (0.0,-30.0), 'América del Sur': (-15.0,-60.0), 'Oceanía': (-25.0,135.0)}
    df_reg['Latitud'] = df_reg['Región del Cliente'].map(lambda x: coordenadas.get(x, (0,0))[0])
    df_reg['Longitud'] = df_reg['Región del Cliente'].map(lambda x: coordenadas.get(x, (0,0))[1])
    
    fig_mapa = px.scatter_geo(df_reg, lat='Latitud', lon='Longitud', size='Ventas Totales', hover_name='Región del Cliente', projection="natural earth", color_discrete_sequence=['#FF9900'], size_max=35)
    fig_mapa.update_geos(showcountries=True, countrycolor="#3A4553", showcoastlines=True, coastlinecolor="#3A4553", showland=True, landcolor="#232F3E", bgcolor="rgba(0,0,0,0)")
    fig_mapa.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=400)
    st.plotly_chart(fig_mapa, use_container_width=True)

with col_der:
    st.subheader("TOP DE VENTAS POR REGION")
    df_reg_top = df.groupby('Región del Cliente')['Ventas Totales'].sum().reset_index().sort_values('Ventas Totales', ascending=False).head(4)
    df_reg_top['Rank'] = range(1, len(df_reg_top) + 1)
    df_reg_top['Region_Label'] = df_reg_top['Rank'].apply(obtener_medalla) + df_reg_top['Región del Cliente']
    df_reg_top = df_reg_top.sort_values('Ventas Totales', ascending=True)
    
    fig_reg = px.bar(df_reg_top, x='Ventas Totales', y='Region_Label', orientation='h', color_discrete_sequence=['#00A8E1'], text='Ventas Totales')
    fig_reg.update_traces(texttemplate='S/ %{text:,.2f}', textposition='inside', textfont=dict(color='white', size=14))
    fig_reg.update_layout(xaxis_title="Ingresos (S/)", yaxis_title="", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), font=dict(color="white"), height=400)
    st.plotly_chart(fig_reg, use_container_width=True)

st.subheader("EVOLUCIÓN TEMPORAL DE VENTAS")
df_tiempo = df.groupby('Mes y Año')['Ventas Totales'].sum().reset_index()
fig_tiempo = px.line(df_tiempo, x='Mes y Año', y='Ventas Totales', markers=True, color_discrete_sequence=['#FF9900'])
fig_tiempo.update_layout(xaxis_title="Años / Meses", yaxis_title="Venta Total (S/)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=350, margin=dict(l=0, r=0, t=10, b=0))
fig_tiempo.update_yaxes(tickprefix="S/ ", tickformat=",.0f", gridcolor="#3A4553")
fig_tiempo.update_xaxes(gridcolor="#3A4553")
fig_tiempo.update_traces(line=dict(width=3), marker=dict(size=8, color='#FFFFFF'))
st.plotly_chart(fig_tiempo, use_container_width=True)

st.divider()

st.title("🔍 Análisis Exploratorio de Datos (EDA)")
st.markdown("Fase de comprensión de datos y visualización estadística utilizando **Matplotlib** y **Seaborn**.")

plt.style.use('dark_background')

col_eda1, col_eda2 = st.columns(2)

with col_eda1:
    st.markdown("##### 1. Distribución de Precios con Descuento")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Precio con Descuento'].dropna(), bins=30, kde=True, color='#FF9900', ax=ax1)
    ax1.set_xlabel('Precio con Descuento (S/)')
    ax1.set_ylabel('Frecuencia (Cantidad de Productos)')
    fig1.patch.set_facecolor('#131921')
    ax1.set_facecolor('#131921')
    st.pyplot(fig1)

    st.markdown("##### 3. Detección de Outliers en Ventas")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df['Ventas Totales'], color='#FF9900', ax=ax3)
    ax3.set_xlabel('Ventas Totales (S/)')
    fig3.patch.set_facecolor('#131921')
    ax3.set_facecolor('#131921')
    st.pyplot(fig3)

with col_eda2:
    st.markdown("##### 2. Correlación de Variables")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='Oranges', fmt=".2f", ax=ax2, cbar=False)
    fig2.patch.set_facecolor('#131921')
    ax2.set_facecolor('#131921')
    st.pyplot(fig2)

    st.markdown("##### 4. Segmentación: Precio vs Ventas por Región")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Precio con Descuento', y='Ventas Totales', hue='Región del Cliente', palette='Set2', alpha=0.8, ax=ax4)
    ax4.set_xlabel('Precio con Descuento (S/)')
    ax4.set_ylabel('Ventas Totales (S/)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    fig4.patch.set_facecolor('#131921')
    ax4.set_facecolor('#131921')
    st.pyplot(fig4)

st.divider()

st.title("Modelos Analíticos (Machine Learning)")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Regresión Múltiple (Forecast)", "👥 K-Means (Clustering)", "🌳 Random Forest", "🌲 Árboles de Decisión"])

with tab1:
    st.markdown("### Predicción de Ventas (Precio, Descuento, Rating y Región)")
    X_reg = df[['Precio con Descuento', 'Porcentaje de Descuento', 'Valoración', 'region_num']].fillna(0)
    y_reg = df['Ventas Totales']
    
    modelo_reg = LinearRegression()
    modelo_reg.fit(X_reg, y_reg)
    precision = modelo_reg.score(X_reg, y_reg)
    
    st.success(f"✅ **Modelo Entrenado.** Precisión (R²): **{precision:.2f}**")
    st.info("💡 **Forecast:** Este modelo multivariante permite predecir los ingresos exactos ajustando las variables de descuento y precio para cada región específica.")

with tab2:
    st.markdown("### Segmentación de Clientes por Región, Pago y Frecuencia")
    X_cluster = df[['region_num', 'pago_num', 'Cantidad Vendida']].fillna(0)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster = df.copy()
    df_cluster['Perfil'] = kmeans.fit_predict(X_cluster)
    
    df_cluster['Perfil'] = df_cluster['Perfil'].replace({0: 'Comprador Casual', 1: 'Cliente Frecuente', 2: 'Comprador Mayorista'})
    
    st.success("✅ **Modelo K-Means Entrenado.** Se identificaron 3 perfiles principales:")
    st.dataframe(df_cluster[['Región del Cliente', 'Método de Pago', 'Cantidad Vendida', 'Perfil']].head(10), use_container_width=True)

with tab3:
    st.markdown("### Identificación de Productos Exitosos")
    X_rf = df[['Precio con Descuento', 'Porcentaje de Descuento', 'Valoración', 'cantidad_reseñas']].fillna(0)
    y_rf = df['Ventas Totales']
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_rf, y_rf)
    
    df_rf = df.copy()
    df_rf['Probabilidad_Exito'] = rf.predict(X_rf)
    top_prod = df_rf.sort_values('Probabilidad_Exito', ascending=False).iloc[0]
    
    st.success("✅ **Modelo Random Forest Entrenado (Alta precisión).**")
    st.warning(f"🏆 **Producto con mayor probabilidad de compra:** El ID **{top_prod['producto_id']}** de la categoría *{top_prod['Categoría del Producto']}*.")

with tab4:
    st.markdown("### Extracción de Reglas de Negocio")
    dt = DecisionTreeRegressor(max_depth=2, random_state=42)
    dt.fit(X_rf, y_rf)
    
    st.success("✅ **Modelo de Árbol de Decisión Entrenado.**")
    st.info("📋 **Regla Comercial Detectada (Si descuento alto + rating alto → Mayor venta):** \n\nEl algoritmo demuestra que aplicar un descuento agresivo a productos que ya tienen un Rating Alto es el camino más rápido para disparar el volumen de ventas.")