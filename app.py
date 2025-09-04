import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64

# Título do aplicativo e configuração da página
st.set_page_config(
    page_title="Sistema de Previsão Climática - Brasil",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌧️ Sistema de Previsão Climática - Brasil")
st.markdown("### Previsão de Volume Diário de Chuva (mm)")

# Sidebar para navegação
st.sidebar.title("Navegação")
opcao = st.sidebar.selectbox(
    "Escolha uma opção:",
    ["Previsão Individual", "Mapa e Download de Dados", "Upload de CSV", "Sobre o Sistema"]
)

# Funções de simulação de dados
def make_prediction(data):
    """Simula uma previsão de precipitação com base nos dados de entrada."""
    base_precip = np.random.uniform(0, 15)
    if data.get("temp_max", 25) > 30:
        base_precip *= 1.5
    if data.get("umidade", 50) > 70:
        base_precip *= 1.3
    
    municipio = data.get("municipio", "Itirapina")
    if municipio == "São Paulo":
        base_precip *= 1.2
    elif municipio == "Rio de Janeiro":
        base_precip *= 1.1
    
    return max(0, base_precip)

def generate_municipios_list():
    """Gera uma lista simulada de municípios e suas coordenadas, incluindo cidades de SP."""
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
    """Gera um DataFrame simulado com dados de precipitação para todas as estações simuladas."""
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

# --- Seção: Previsão Individual ---
if opcao == "Previsão Individual":
    st.header("📊 Previsão Individual")
    
    municipios_list = generate_municipios_list()["cidade"].tolist()
    municipio_selecionado = st.selectbox("Selecione o Município", municipios_list)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados Meteorológicos")
        temp_max = st.slider("Temperatura Máxima (°C)", -5.0, 45.0, 25.0, 0.1)
        temp_min = st.slider("Temperatura Mínima (°C)", -10.0, 35.0, 15.0, 0.1)
        umidade = st.slider("Umidade Relativa (%)", 0.0, 100.0, 60.0, 1.0)
        pressao = st.slider("Pressão Atmosférica (hPa)", 900.0, 1050.0, 1013.0, 0.1)
        
    with col2:
        st.subheader("Dados Complementares")
        vel_vento = st.slider("Velocidade do Vento (m/s)", 0.0, 30.0, 5.0, 0.1)
        rad_solar = st.slider("Radiação Solar (MJ/m²)", 0.0, 35.0, 20.0, 0.1)
        data_previsao = st.date_input("Data da Previsão", datetime.now())
        
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
        
        previsao = make_prediction(dados_input)
        
        st.success(f"🌧️ Previsão de Precipitação para {municipio_selecionado}: **{previsao:.2f} mm**")
        
        if previsao < 1:
            st.info("☀️ Dia seco - Precipitação muito baixa")
        elif previsao < 5:
            st.info("🌤️ Chuva leve - Precipitação baixa")
        elif previsao < 15:
            st.warning("🌦️ Chuva moderada - Precipitação moderada")
        else:
            st.error("⛈️ Chuva intensa - Precipitação alta")
            
        fig = go.Figure(data=[
            go.Bar(x=["Previsão"], y=[previsao], marker_color="lightblue")
        ])
        fig.update_layout(
            title="Volume de Chuva Previsto",
            yaxis_title="Precipitação (mm)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Seção: Mapa e Download de Dados ---
elif opcao == "Mapa e Download de Dados":
    st.header("🗺️ Mapa Interativo do Brasil")
    st.markdown("Passe o mouse sobre os pontos para ver o nome da estação. A cor indica o tipo de estação.")

    estacoes_df = generate_municipios_list()
    
    # Adiciona a visualização de pontos de cidades de médio porte em um mapa do Brasil
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
    
    # Foca o mapa no Brasil
    fig_mapa.update_geos(
        lonaxis_range=[-75, -30],
        lataxis_range=[-35, 5],
        center={"lat": -14, "lon": -55}
    )

    st.plotly_chart(fig_mapa, use_container_width=True)
    
    # Barra de rolagem para os municípios de São Paulo
    st.header("Municípios de São Paulo")
    sp_municipios = estacoes_df[estacoes_df['estado'] == 'SP']['cidade'].tolist()
    st.selectbox("Selecione um município para mais detalhes (simulação)", sp_municipios)


    st.markdown("---")
    st.header("📥 Download de Dados Completos")
    st.markdown("Clique no botão para baixar um arquivo CSV com dados diários simulados para todas as estações.")

    if st.button(f"📥 Baixar Dados de Todas as Estações", type="primary"):
        with st.spinner('Gerando arquivo...'):
            df_dados_completos = generate_all_brazil_data()
            csv_file = df_dados_completos.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="previsao_todos_municipios_brasil.csv">Clique aqui para baixar o arquivo</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success("Arquivo gerado com sucesso!")

# --- Seção: Upload de CSV ---
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

# --- Seção: Sobre o Sistema ---
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
    
    # --- Seção de Créditos ---
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
st.markdown("**Desenvolvido por:** Manus AI | **Versão:** 1.0 | **Última atualização:** 2024")
