import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64

# Page title and layout
st.set_page_config(
    page_title="Sistema de Previsão Climática - Brasil",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌧️ Sistema de Previsão Climática - Brasil")
st.markdown("### Previsão de Volume Diário de Chuva (mm)")

# Sidebar for navigation
st.sidebar.title("Navegação")
opcao = st.sidebar.selectbox(
    "Escolha uma opção:",
    ["Previsão Individual", "Análise de Dados e Previsões", "Upload de CSV", "Sobre o Sistema"]
)

# Functions for data simulation
def make_prediction_series(data, days=1):
    """Simulates a precipitation, temperature, and humidity forecast based on XGBoost logic for a series of days."""
    predictions = []
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    for i in range(days):
        # Simulate base precipitation mimicking XGBoost logic
        base_precip = np.random.uniform(0.5, 5) # Low base
        
        # Add factors based on input variables, simulating feature importance
        temp_factor = 1 + (data.get("temp_max", 25) - 25) * 0.1
        umidade_factor = 1 + (data.get("umidade", 60) - 60) * 0.05
        
        # Combine factors with a time-series trend (e.g., sine wave)
        time_factor = np.sin(np.pi * 2 * i / days) * 2 + 1
        
        precipitacao = max(0, base_precip * temp_factor * umidade_factor * time_factor + np.random.uniform(-1, 1))

        # Simulate other variables
        temperatura_media = max(0, data.get("temp_max", 25) + np.random.uniform(-3, 3))
        umidade_relativa = max(0, min(100, data.get("umidade", 60) + np.random.uniform(-5, 5)))

        predictions.append({
            "data": dates[i].strftime("%Y-%m-%d"),
            "precipitacao_mm": precipitacao,
            "temperatura_media": temperatura_media,
            "umidade_relativa": umidade_relativa
        })
    
    return pd.DataFrame(predictions)

def generate_municipios_list():
    """Generates a simulated list of municipalities and their coordinates, including SP cities."""
    return pd.DataFrame({
        'cidade': [
            "Campinas", "Ribeirão Preto", "Uberlândia", "Santos", "Londrina",
            "São José dos Campos", "Feira de Santana", "Cuiabá", "Anápolis",
            "Maringá", "Juiz de Fora", "Niterói", "Campos dos Goytacazes",
            "Caxias do Sul", "Sorocaba", "Joinville", "Natal",
            "Araraquara", "Bauru", "Franca", "Jundiaí", "Piracicaba",
            "Presidente Prudente", "São Carlos", "Taubaté"
        ],
        'estado': [
            'SP', 'SP', 'MG', 'SP', 'PR', 'SP', 'BA', 'MT', 'GO', 'PR', 'MG', 'RJ', 'RJ', 'RS', 'SP', 'SC', 'RN',
            'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP'
        ],
        'lat': [
            -22.9099, -21.1762, -18.918, -23.9634, -23.3106, -23.1794, -12.2464, -15.5989,
            -16.3275, -23.424, -21.763, -22.8889, -21.7583, -29.1672, -23.498, -26.304, -5.7947,
            -21.807, -22.316, -20.538, -23.186, -22.721, -22.124, -22.016, -23.023
        ],
        'lon': [
            -47.0626, -47.8823, -48.2772, -46.3353, -51.1627, -45.8869, -38.9668, -56.0949,
            -48.9566, -51.9389, -43.345, -43.107, -41.3328, -51.1778, -47.4488, -48.847, -35.2114,
            -48.188, -49.066, -47.400, -46.883, -47.649, -51.401, -47.893, -45.556
        ],
        'tipo_estacao': [
            'Automática', 'Automática', 'Convencional', 'Automática', 'Convencional',
            'Automática', 'Convencional', 'Automática', 'Convencional', 'Automática',
            'Convencional', 'Automática', 'Convencional', 'Automática', 'Automática',
            'Convencional', 'Automática', 'Automática', 'Convencional', 'Convencional',
            'Automática', 'Automática', 'Convencional', 'Convencional', 'Automática'
        ]
    })

def generate_all_brazil_data():
    """Generates a simulated DataFrame with precipitation data for all simulated stations."""
    municipios_df = generate_municipios_list()
    municipios_simulados = municipios_df['cidade'].tolist()
    
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    data_list = []
    
    for municipio in municipios_simulados:
        for date in dates:
            precipitacao = np.random.uniform(0, 50)
            data_list.append({
                "municipio": municipio,
                "data": date.strftime("%Y-%m-%d"),
                "precipitacao_mm": precipitacao
            })
            
    return pd.DataFrame(data_list)
    
def generate_monthly_forecast_data(municipios):
    """Simulates a monthly forecast for the next 30 days, mimicking XGBoost output."""
    data_list = []
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)
    
    current_date = start_date
    while current_date < end_date:
        for municipio in municipios:
            # Simulate a time series pattern for precipitation
            day_of_month = current_date.day
            base_precip = np.sin(np.pi * 2 * day_of_month / 30) * 10 + 15
            # Add random noise
            precipitacao = max(0, base_precip + np.random.uniform(-5, 5))
            
            data_list.append({
                "municipio": municipio,
                "data": current_date.strftime("%Y-%m-%d"),
                "precipitacao_mm": precipitacao,
                "temperatura_media": np.random.uniform(20, 30),
                "umidade_relativa": np.random.uniform(50, 90),
            })
        current_date += timedelta(days=1)
        
    return pd.DataFrame(data_list)

# --- Section: Individual Forecast ---
if opcao == "Previsão Individual":
    st.header("📊 Previsão Individual")
    
    municipios_list = generate_municipios_list()["cidade"].tolist()
    municipio_selecionado = st.selectbox("Selecione o Município", municipios_list)
    
    dias_previsao = st.selectbox(
        "Selecione o número de dias para a previsão:",
        [1, 3, 5, 7, 10]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados Meteorológicos")
        temp_max = st.slider("Temperatura Máxima (°C)", -5.0, 45.0, 25.0, 0.1)
        temp_min = st.slider("Temperatura Mínima (°C)", -10.0, 35.0, 15.0, 0.1)
        umidade = st.slider("Umidade Relativa (%)", 0.0, 100.0, 60.0, 1.0)
        
    with col2:
        st.subheader("Dados Complementares")
        pressao = st.slider("Pressão Atmosférica (hPa)", 900.0, 1050.0, 1013.0, 0.1)
        vel_vento = st.slider("Velocidade do Vento (m/s)", 0.0, 30.0, 5.0, 0.1)
        rad_solar = st.slider("Radiação Solar (MJ/m²)", 0.0, 35.0, 20.0, 0.1)
        
    if st.button("🔮 Fazer Previsão", type="primary"):
        dados_input = {
            "municipio": municipio_selecionado,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "umidade": umidade,
            "pressao": pressao,
            "vel_vento": vel_vento,
            "rad_solar": rad_solar
        }
        
        previsoes_df = make_prediction_series(dados_input, days=dias_previsao)
        
        st.subheader(f"📊 Análise Detalhada para {municipio_selecionado}")
        st.dataframe(previsoes_df)

        # Gráfico de Linha (Tendência)
        st.markdown("---")
        st.subheader("📈 Gráfico de Tendência da Precipitação")
        fig_line_precip = px.line(
            previsoes_df, 
            x="data", 
            y="precipitacao_mm",
            markers=True,
            title=f"Tendência Diária de Chuva para {municipio_selecionado}",
            color_discrete_sequence=["#0077b6"]
        )
        fig_line_precip.update_layout(xaxis_title="Data", yaxis_title="Precipitação (mm)")
        st.plotly_chart(fig_line_precip, use_container_width=True)

        # Gráfico de Barras (Volume Diário)
        st.markdown("---")
        st.subheader("📊 Gráfico de Volume de Chuva por Dia")
        fig_bar_precip = px.bar(
            previsoes_df,
            x="data",
            y="precipitacao_mm",
            title=f"Volume de Chuva Previsto por Dia para {municipio_selecionado}",
            color="precipitacao_mm",
            color_continuous_scale=px.colors.sequential.Teal
        )
        fig_bar_precip.update_layout(xaxis_title="Data", yaxis_title="Precipitação (mm)")
        st.plotly_chart(fig_bar_precip, use_container_width=True)

        # Gráfico Combinado (Precipitação e Temperatura)
        st.markdown("---")
        st.subheader("📉 Gráfico Combinado de Precipitação e Temperatura")
        fig_combo = go.Figure()
        
        # Adiciona a barra para precipitação
        fig_combo.add_trace(go.Bar(
            x=previsoes_df["data"],
            y=previsoes_df["precipitacao_mm"],
            name="Precipitação (mm)",
            marker_color="#005f73"
        ))
        
        # Adiciona a linha para temperatura
        fig_combo.add_trace(go.Scatter(
            x=previsoes_df["data"],
            y=previsoes_df["temperatura_media"],
            name="Temperatura Média (°C)",
            yaxis="y2",
            line=dict(color="#d00000", width=3)
        ))
        
        fig_combo.update_layout(
            title=f"Precipitação e Temperatura Média por Dia para {municipio_selecionado}",
            yaxis=dict(title="Precipitação (mm)"),
            yaxis2=dict(
                title="Temperatura Média (°C)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_combo, use_container_width=True)

        # Gráfico Estatístico (Box Plot)
        st.markdown("---")
        st.subheader("📦 Análise Estatística da Precipitação")
        fig_box = px.box(
            previsoes_df,
            y="precipitacao_mm",
            title=f"Distribuição da Precipitação para {municipio_selecionado}"
        )
        fig_box.update_layout(yaxis_title="Precipitação (mm)")
        st.plotly_chart(fig_box, use_container_width=True)

# --- Section: Data Analysis and Forecasts ---
elif opcao == "Análise de Dados e Previsões":
    st.header("📈 Análise de Dados e Previsões")
    
    # Generate data
    estacoes_df = generate_municipios_list()
    forecast_df = generate_monthly_forecast_data(estacoes_df["cidade"].tolist())
    
    st.markdown("---")
    st.subheader("Previsão de Chuva para o Próximo Mês (Simulação XGBoost)")
    
    municipio_selecionado_mensal = st.selectbox(
        "Selecione um Município para a Previsão Mensal",
        estacoes_df["cidade"].tolist()
    )
    
    filtered_df = forecast_df[forecast_df["municipio"] == municipio_selecionado_mensal]
    
    fig_line = px.line(
        filtered_df, 
        x="data", 
        y="precipitacao_mm", 
        title=f"Previsão de Chuva para {municipio_selecionado_mensal} no Próximo Mês",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_line.update_layout(
        xaxis_title="Data",
        yaxis_title="Precipitação (mm)"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Mapa Interativo do Brasil")
    st.markdown("Passe o mouse sobre os pontos para ver o nome da estação. A cor indica o tipo de estação.")
    
    fig_mapa = px.scatter_geo(
        estacoes_df,
        lat='lat',
        lon='lon',
        hover_name='cidade',
        color='tipo_estacao',
        title='Localização das Estações Meteorológicas (Simulação)',
        scope='south america'
    )
    
    fig_mapa.update_layout(
        geo_scope='south america',
        geo_resolution=50,
        geo_showsubunits=True,
        geo_subunitcolor='lightgrey',
        geo_showcountries=True,
        geo_countrycolor='black',
        geo_bgcolor='white'
    )
    
    fig_mapa.update_geos(
        lonaxis_range=[-75, -30],
        lataxis_range=[-35, 5],
        center={"lat": -14, "lon": -55}
    )

    st.plotly_chart(fig_mapa, use_container_width=True)

    st.markdown("---")
    st.subheader("Análises Complementares")

    # Group data by city and get total precipitation
    total_precip_by_city = forecast_df.groupby("municipio")["precipitacao_mm"].sum().reset_index()
    fig_bar = px.bar(
        total_precip_by_city.sort_values(by="precipitacao_mm", ascending=False),
        x="municipio",
        y="precipitacao_mm",
        title="Volume Total de Chuva Previsto por Município (Próximo Mês)",
        color="precipitacao_mm",
        color_continuous_scale=px.colors.sequential.Bluyl
    )
    fig_bar.update_layout(
        xaxis_title="Município",
        yaxis_title="Precipitação Total (mm)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Scatter plot of Temperature vs Humidity
    fig_scatter = px.scatter(
        forecast_df,
        x="temperatura_media",
        y="umidade_relativa",
        color="municipio",
        hover_name="municipio",
        title="Relação entre Temperatura e Umidade",
        size="precipitacao_mm"
    )
    fig_scatter.update_layout(
        xaxis_title="Temperatura Média (°C)",
        yaxis_title="Umidade Relativa (%)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Pie chart of station types
    station_counts = estacoes_df['tipo_estacao'].value_counts().reset_index()
    station_counts.columns = ['Tipo de Estação', 'Quantidade']
    fig_pie = px.pie(
        station_counts,
        values='Quantidade',
        names='Tipo de Estação',
        title='Distribuição dos Tipos de Estações',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📥 Download de Dados Completos")
    st.markdown("Clique no botão para baixar um arquivo CSV com dados diários simulados para todas as estações.")

    if st.button(f"📥 Baixar Dados de Todas as Estações", type="primary"):
        with st.spinner('Gerando arquivo...'):
            df_dados_completos = generate_all_brazil_data()
            csv_file = df_dados_completos.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="previsao_todos_municipios_brasil.csv">Clique aqui para baixar o arquivo</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success("Arquivo gerado com sucesso!")

# --- Section: Upload CSV ---
elif opcao == "Upload de CSV":
    st.header("📁 Upload de Arquivo CSV")
    st.markdown("""
    **Formato esperado do CSV:**
    - Colunas: `data`, `temp_max`, `temp_min`, `umidade`, `pressao`, `vel_vento`, `rad_solar`
    - Data no formato: `YYYY-MM-DD`
    """)
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Arquivo carregado com sucesso!")
            st.subheader("Preview dos Dados")
            st.dataframe(df.head())
            
            if st.button("🔮 Processar Previsões", type="primary"):
                with st.spinner('Processando previsões...'):
                    previsoes = [make_prediction(row.to_dict()) for _, row in df.iterrows()]
                df["previsao_precipitacao"] = previsoes
                st.subheader("Resultados das Previsões")
                st.dataframe(df)
                
                if "data" in df.columns:
                    df["data"] = pd.to_datetime(df["data"])
                    fig = px.line(df, x="data", y="previsao_precipitacao", 
                                  title="Previsão de Precipitação ao Longo do Tempo")
                    fig.update_layout(yaxis_title="Precipitação (mm)")
                    st.plotly_chart(fig, use_container_width=True)
                
                csv_file = df.to_csv(index=False)
                b64 = base64.b64encode(csv_file.encode()).decode()
                href = f"<a href=\"data:file/csv;base64,{b64}\" download=\"previsoes_clima.csv\">📥 Download dos Resultados</a>"
                st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

# --- Section: About the System ---
else:  # Sobre o Sistema
    st.header("ℹ️ Sobre o Sistema")
    st.markdown("""
    ### Sistema de Previsão Climática para o Brasil
    
    Este sistema foi desenvolvido para demonstrar uma interface interativa para visualização e 
    download de dados de precipitação (em milímetros) para municípios brasileiros.
    
    #### 🎯 Características Principais:
    - **Interface Interativa**: Navegação e download de dados via mapa.
    - **Dados Simulados**: Os dados de previsão são simulados para fins de demonstração da plataforma.
    - **Visualização**: Utiliza a biblioteca Plotly para gerar gráficos e mapas.
    
    #### 🔬 Tecnologias Utilizadas:
    - **Interface**: Streamlit
    - **Visualização**: Plotly
    
    #### 🌍 Aplicações:
    - Agricultura de precisão
    - Gestão de recursos hídricos
    - Planejamento urbano
    - Pesquisa climática
    """)
    
    # --- Credits Section ---
    st.markdown("---")
    st.header("👨‍💻 Sobre o Autor")
    st.markdown("""
    Este projeto foi desenvolvido por:
    - **Nome:** Rafael Grecco Sanches
    
    #### Links Profissionais:
    - **Lattes:** [Seu Link do Lattes](<URL DO SEU LATTES>)
    - **Google Acadêmico:** [Seu Perfil no Google Acadêmico](<URL DO SEU GOOGLE ACADÊMICO>)
    - **Outros:** [Seu Site ou LinkedIn](<URL DO SEU SITE/LINKEDIN>)
    """)

# Footer
st.markdown("---")
st.markdown("**Desenvolvido por:** Manus AI | **Versão:** 1.5 | **Última atualização:** 2024")
