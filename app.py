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

def make_prediction(data):
    """Simula uma previsão de precipitação com base nos dados de entrada."""
    base_precip = np.random.uniform(0, 15)
    if data.get("temp_max", 25) > 30:
        base_precip *= 1.5
    if data.get("umidade", 50) > 70:
        base_precip *= 1.3
    
    # Simula variações por local
    municipio = data.get("municipio", "Itirapina")
    if municipio == "São Paulo":
        base_precip *= 1.2
    elif municipio == "Rio de Janeiro":
        base_precip *= 1.1
    
    return max(0, base_precip)

def generate_all_brazil_data():
    """Gera um DataFrame simulado com dados de precipitação para todos os municípios do Brasil."""
    # Simula uma lista de municípios brasileiros
    municipios_simulados = [
        "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Fortaleza", 
        "Curitiba", "Manaus", "Recife", "Porto Alegre", "Brasília", "Campinas",
        "Goiânia", "Belém", "Guarulhos", "São Luís", "São Gonçalo", "Maceió", "Teresina",
        "Campo Grande", "Natal", "Duque de Caxias", "Nova Iguaçu", "São Bernardo do Campo",
        "João Pessoa", "Santo André", "Osasco", "Jaboatão dos Guararapes", "Contagem",
        "Uberlândia", "Ribeirão Preto", "Sorocaba", "Londrina", "Aracaju", "Joinville",
        "Cuiabá", "Ananindeua", "Juiz de Fora", "Niterói", "Campos dos Goytacazes",
        "Caxias do Sul", "Santos", "Mauá", "Vila Velha", "Aparecida de Goiânia"
    ]
    
    # Cria uma lista de datas para 30 dias
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    data_list = []
    
    for municipio in municipios_simulados:
        for date in dates:
            # Simula um valor de precipitação
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
    
    municipios_list = generate_all_brazil_data()["municipio"].unique().tolist()
    municipio_selecionado = st.selectbox("Selecione o Município (com barra de rolagem)", municipios_list)
    
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
    st.markdown("Passe o mouse sobre os estados para visualizar. Clique no botão para baixar dados de todos os municípios.")

    # URL pública do GeoJSON para os estados do Brasil
    brazil_geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click-that-hood/master/geojson/brazil-states.geojson'
    
    # Cria um DataFrame com dados simulados para colorir o mapa
    brazil_df = pd.DataFrame({
        "Estado": ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"],
        "Simulacao_Precipitacao": np.random.uniform(5, 25, 27) # Dados variados para colorir
    })
    
    fig_mapa = px.choropleth(
        brazil_df,
        geojson=brazil_geojson_url,
        locations="Estado",
        locationmode="geojson-id",
        title="Previsão de Chuva por Estado (Simulação)",
        color="Simulacao_Precipitacao",
        color_continuous_scale="Viridis",
        labels={'Simulacao_Precipitacao':'Simulação (mm)'},
        hover_name="Estado"
    )
    fig_mapa.update_geos(fitbounds="locations", visible=False)
    fig_mapa.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

    st.markdown("---")
    st.header("📥 Download de Dados Completos")
    st.markdown("Clique no botão para baixar um arquivo CSV com dados diários simulados para todos os municípios.")

    if st.button(f"📥 Baixar Dados de Todos os Municípios", type="primary"):
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
