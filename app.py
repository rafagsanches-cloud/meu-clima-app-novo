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
st.markdown("### Mapa de Previsão de Chuva")

# Sidebar para navegação
st.sidebar.title("Navegação")
opcao = st.sidebar.selectbox(
    "Escolha uma opção:",
    ["Mapa e Download de Dados", "Sobre o Sistema"]
)

def generate_all_brazil_data():
    """Gera um DataFrame simulado com dados de precipitação para todos os municípios do Brasil."""
    # Simula uma lista de municípios brasileiros
    # Em uma aplicação real, esta lista seria carregada de um banco de dados
    municipios_simulados = [
        "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Fortaleza", 
        "Curitiba", "Manaus", "Recife", "Porto Alegre", "Brasília"
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

if opcao == "Mapa e Download de Dados":
    st.header("🗺️ Mapa do Brasil")

    # URL pública do GeoJSON para os estados do Brasil
    brazil_geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click-that-hood/master/geojson/brazil-states.geojson'
    
    # Cria um DataFrame vazio para desenhar o mapa.
    # Isso garante que o mapa seja exibido de forma limpa, sem dados.
    brazil_df = pd.DataFrame({
        "Estado": ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"],
        "Valor": [0] * 27
    })
    
    fig_mapa = px.choropleth(
        brazil_df,
        geojson=brazil_geojson_url,
        locations="Estado",
        locationmode="geojson-id",
        title="Mapa do Brasil (Estados)",
        color="Valor",
        color_continuous_scale="Viridis",
        labels={'Valor':'Simulação'},
        hover_name="Estado"
    )
    fig_mapa.update_geos(fitbounds="locations", visible=False)
    fig_mapa.update_layout(
        coloraxis_showscale=False,  # Remove a barra de cores
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    st.plotly_chart(fig_mapa, use_container_width=True)
    
    st.markdown("---")
    st.header("📥 Download de Dados")
    
    st.markdown("Clique no botão abaixo para baixar um arquivo CSV com dados diários simulados para todos os municípios do Brasil.")
    
    if st.button(f"📥 Baixar Dados de Todos os Municípios", type="primary"):
        with st.spinner('Gerando arquivo...'):
            df_dados_completos = generate_all_brazil_data()
            csv_file = df_dados_completos.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="previsao_todos_municipios_brasil.csv">Clique aqui para baixar o arquivo</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success("Arquivo gerado com sucesso!")

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
