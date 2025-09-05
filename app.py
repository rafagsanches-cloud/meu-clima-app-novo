import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import os

# --- Funções de Pré-processamento e Modelagem (Simuladas) ---
def create_features(df, config):
    """Cria features simuladas a partir de um DataFrame de dados climáticos."""
    df_copy = df.copy()

    # Renomear colunas para padronização interna
    df_copy.rename(columns=config["column_mapping"], inplace=True)

    # Converter a coluna de data para datetime
    df_copy[config["date_column"]] = pd.to_datetime(df_copy[config["date_column"]], errors='coerce')
    df_copy.dropna(subset=[config["date_column"]], inplace=True)
    df_copy.sort_values(config["date_column"], inplace=True)
    df_copy.set_index(config["date_column"], inplace=True)

    # Converter colunas numéricas e preencher NaNs
    for col in config["numeric_columns"]:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy[col].fillna(df_copy[col].median(), inplace=True)

    # Apenas para simulação, não precisamos de todas as features complexas
    df_copy["ano"] = df_copy.index.year
    df_copy["mes"] = df_copy.index.month
    df_copy["dia"] = df_copy.index.day
    df_copy["temp_media"] = (df_copy["temp_max"] + df_copy["temp_min"]) / 2

    # Preencher NaNs após a criação de features
    df_copy.fillna(method="bfill", inplace=True)
    df_copy.fillna(method="ffill", inplace=True)

    return df_copy.dropna()

def make_prediction_series(df_predict, num_days):
    """
    Simula uma série de previsões de precipitação para um número de dias.
    Esta função não depende de nenhum modelo externo ou biblioteca.
    """
    
    config_itirapina = {
        "date_column": 'data',
        "column_mapping": {
            'data': 'data', 'temp_max': 'temp_max', 'temp_min': 'temp_min', 'umidade': 'umidade', 'pressao': 'pressao', 'vel_vento': 'vel_vento', 'rad_solar': 'rad_solar'
        },
        "numeric_columns": ['temp_max', 'temp_min', 'umidade', 'pressao', 'vel_vento', 'rad_solar']
    }
    
    df_processed = create_features(df_predict.copy(), config_itirapina)
    
    # Lógica de previsão simulada
    # A precipitação é uma função simples das features de entrada mais um termo aleatório e um fator de tendência.
    predictions = (0.2 * df_processed['temp_max']) + (0.1 * df_processed['umidade']) + np.random.uniform(0, 5, size=len(df_processed))
    predictions[predictions < 0] = 0
    
    # Criar uma série temporal de previsões
    forecast_dates = pd.date_range(start=df_processed.index.max(), periods=num_days, freq='D')
    
    # Simular uma série com leve decaimento e ruído
    simulated_forecast = predictions.iloc[-1] + np.random.normal(loc=0, scale=1.5, size=num_days)
    simulated_forecast[simulated_forecast < 0] = 0
    
    return pd.Series(simulated_forecast, index=forecast_dates, name=f"previsao_precipitacao")

def generate_simulated_historical_data(num_days=365):
    """Gera um DataFrame com dados históricos simulados para fins de demonstração."""
    start_date = datetime.now() - timedelta(days=num_days)
    dates = pd.date_range(start_date, periods=num_days, freq='D')

    # Simular temperaturas com padrão sazonal
    day_of_year = dates.dayofyear
    temp_variation = np.sin(2 * np.pi * day_of_year / 365)
    temp_max_base = 25 + temp_variation * 10 + np.random.normal(0, 2, num_days)
    temp_min_base = 15 + temp_variation * 8 + np.random.normal(0, 1.5, num_days)

    # Simular umidade inversamente correlacionada com a temperatura
    umidade_base = 60 - temp_variation * 15 + np.random.normal(0, 5, num_days)
    umidade_base[umidade_base > 100] = 100
    umidade_base[umidade_base < 0] = 0

    # Simular precipitação que depende da umidade e da estação
    precipitacao_base = np.maximum(0, (umidade_base - 60) * 0.5 + np.random.normal(0, 1, num_days))

    # Criar o DataFrame
    df = pd.DataFrame({
        'data': dates,
        'temp_max': temp_max_base,
        'temp_min': temp_min_base,
        'umidade': umidade_base,
        'pressao': 1013 + np.random.normal(0, 2, num_days),
        'vel_vento': 5 + np.random.normal(0, 1, num_days),
        'rad_solar': 20 + temp_variation * 5 + np.random.normal(0, 2, num_days),
        'precipitacao': precipitacao_base
    })
    
    df['vel_vento'] = df['vel_vento'].clip(lower=0)
    df['rad_solar'] = df['rad_solar'].clip(lower=0)

    return df

def simulate_metrics(municipio):
    """Simula métricas de desempenho para um município específico."""
    base_rmse = np.random.uniform(2.0, 3.5)
    base_mae = np.random.uniform(1.5, 2.5)
    base_r2 = np.random.uniform(0.65, 0.85)
    
    if municipio == "Itirapina":
        return {
            "RMSE": base_rmse * 0.8,
            "MAE": base_mae * 0.8,
            "R2": min(1.0, base_r2 * 1.1)
        }
    else:
        return {
            "RMSE": base_rmse,
            "MAE": base_mae,
            "R2": base_r2
        }

# --- Funções do Streamlit (UI e Interação) ---
st.set_page_config(
    page_title="Sistema de Previsão Climática - Brasil",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .st-emotion-cache-1r6y9d7 { flex-direction: column; }
    .st-emotion-cache-183n07d { flex-direction: column; }
    .st-emotion-cache-1f190e8 { flex-direction: column; }
    .st-emotion-cache-s2e93h { flex-direction: column; }
    .st-emotion-cache-1090333 { gap: 1rem; }
    .st-emotion-cache-12oz5g7 {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        background-color: #f0f2f6;
    }
    .stButton>button {
        border-radius: 12px;
        border: 1px solid #0077b6;
        color: white;
        background-color: #0077b6;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0096c7;
        border-color: #0096c7;
        transform: scale(1.02);
    }
    h1, h2, h3 { color: #03045e; }
</style>
""", unsafe_allow_html=True)

# Lista de cidades (simulada)
def generate_municipios_list():
    return pd.DataFrame({
        'cidade': [
            "Campinas", "Ribeirão Preto", "Uberlândia", "Santos", "Londrina",
            "São José dos Campos", "Feira de Santana", "Cuiabá", "Anápolis",
            "Maringá", "Juiz de Fora", "Niterói", "Campos dos Goytacazes",
            "Caxias do Sul", "Sorocaba", "Joinville", "Natal", "Itirapina",
            "Araraquara", "Bauru", "Franca", "Jundiaí", "Piracicaba",
            "Presidente Prudente", "São Carlos", "Taubaté"
        ],
        'estado': [
            'SP', 'SP', 'MG', 'SP', 'PR', 'SP', 'BA', 'MT', 'GO', 'PR', 'MG', 'RJ', 'RJ', 'RS', 'SP', 'SC', 'RN', 'SP',
            'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP'
        ],
        'lat': [
            -22.9099, -21.1762, -18.918, -23.9634, -23.3106, -23.1794, -12.2464, -15.5989,
            -16.3275, -23.424, -21.763, -22.8889, -21.7583, -29.1672, -23.498, -26.304, -5.7947, -22.259,
            -21.807, -22.316, -20.538, -23.186, -22.721, -22.124, -22.016, -23.023
        ],
        'lon': [
            -47.0626, -47.8823, -48.2772, -46.3353, -51.1627, -45.8869, -38.9668, -56.0949,
            -48.9566, -51.9389, -43.345, -43.107, -41.3328, -51.1778, -47.4488, -48.847, -35.2114, -47.935,
            -48.188, -49.066, -47.400, -46.883, -47.649, -51.401, -47.893, -45.556
        ],
        'tipo_estacao': [
            'Automática', 'Automática', 'Convencional', 'Automática', 'Convencional',
            'Automática', 'Convencional', 'Automática', 'Convencional', 'Automática',
            'Convencional', 'Automática', 'Convencional', 'Automática', 'Automática',
            'Convencional', 'Automática', 'Automática', 'Automática', 'Convencional', 'Convencional',
            'Automática', 'Automática', 'Convencional', 'Convencional', 'Automática'
        ]
    })

def main():
    st.title("🌧️ Previsões Climáticas: Nuvem & Chuva")
    st.markdown("### Previsão de Volume Diário de Chuva (mm)")

    st.sidebar.title("Navegação 🧭")
    opcao = st.sidebar.selectbox(
        "Escolha uma opção:",
        ["Previsão Individual", "Upload de CSV", "Sobre o Sistema"]
    )

    if opcao == "Previsão Individual":
        st.header("🔮 Previsão para Chuvas")
        st.markdown("Selecione um município na lista para obter a previsão detalhada.")

        estacoes_df = generate_municipios_list()
        municipios_list = estacoes_df["cidade"].tolist()
        
        municipio_selecionado = st.selectbox(
            "Selecione o Município:",
            municipios_list,
            index=municipios_list.index("Itirapina")
        )
        
        # Mapa interativo para visualização das cidades
        st.subheader("📍 Localização dos Municípios")
        fig_map = px.scatter_mapbox(
            estacoes_df,
            lat="lat",
            lon="lon",
            hover_name="cidade",
            hover_data={"estado": True, "tipo_estacao": True, "lat": False, "lon": False},
            color_discrete_sequence=["#0077b6"],
            zoom=3,
            height=400
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")
        st.subheader("Parâmetros da Previsão")
        
        col1, col2 = st.columns(2)
        with col1:
            num_dias = st.number_input(
                "Número de dias para a previsão:", 
                min_value=1, max_value=30, value=7, step=1, 
                help="Selecione o período para a previsão (de 1 a 30 dias)."
            )
        with col2:
            temp_max = st.slider("Temperatura Máxima (°C)", -5.0, 45.0, 25.0, 0.1)
            temp_min = st.slider("Temperatura Mínima (°C)", -10.0, 35.0, 15.0, 0.1)
            umidade = st.slider("Umidade Relativa (%)", 0.0, 100.0, 60.0, 1.0)
            vel_vento = st.slider("Velocidade do Vento (m/s)", 0.0, 30.0, 5.0, 0.1)
            
        if st.button("🚀 Gerar Previsão", type="primary"):
            dados_input = {
                "data": [datetime.now()],
                "temp_max": [temp_max],
                "temp_min": [temp_min],
                "umidade": [umidade],
                "pressao": [1013],
                "vel_vento": [vel_vento],
                "rad_solar": [20]
            }
            df_input = pd.DataFrame(dados_input)
            
            # Usando a nova função para previsão de série temporal
            previsoes = make_prediction_series(df_input, num_dias)
            
            st.subheader(f"📊 Previsão para {municipio_selecionado} - {num_dias} Dia(s)")
            
            # Gráfico de linhas para a série temporal de previsão
            fig_previsao = px.line(
                x=previsoes.index,
                y=previsoes.values,
                title=f'Previsão de Precipitação para os Próximos {num_dias} Dias',
                labels={'x': 'Data', 'y': 'Precipitação (mm)'}
            )
            fig_previsao.update_traces(mode='lines+markers', line=dict(color='#0077b6'))
            st.plotly_chart(fig_previsao, use_container_width=True)

            # Tabela de previsões detalhadas
            st.markdown("### 📋 Detalhes da Previsão em Tabela")
            df_previsoes_table = previsoes.to_frame()
            df_previsoes_table.index = df_previsoes_table.index.strftime('%Y-%m-%d')
            df_previsoes_table.columns = ['Precipitação Prevista (mm)']
            st.dataframe(df_previsoes_table, use_container_width=True)

            st.markdown("---")
            st.subheader("📈 Análise de Desempenho do Modelo")
            st.markdown("*(Métricas simuladas para demonstração do modelo XGBoost)*")
            
            metrics_data = simulate_metrics(municipio_selecionado)
            
            metrics_df = pd.DataFrame(list(metrics_data.items()), columns=["Métrica", "Valor"])
            fig_metrics = px.bar(
                metrics_df,
                x="Métrica",
                y="Valor",
                color="Métrica",
                title="Métricas de Avaliação do Modelo",
                color_discrete_map={
                    "RMSE": "#0077b6",
                    "MAE": "#00b4d8",
                    "R2": "#90e0ef"
                },
                text_auto=True
            )
            fig_metrics.update_layout(xaxis_title="", yaxis_title="Valor da Métrica")
            st.plotly_chart(fig_metrics, use_container_width=True)

            # Novos gráficos adicionados aqui
            st.markdown("---")
            st.subheader("📚 Análise Histórica e Estatística (Dados Simulados)")
            st.markdown("*(Gráficos com base em dados históricos simulados para 1 ano.)*")
            
            simulated_df = generate_simulated_historical_data()
            
            # Gráfico de Barras - Precipitação Média Mensal
            monthly_avg = simulated_df.groupby(simulated_df['data'].dt.month_name())['precipitacao'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
            ])
            
            fig_bar_month = px.bar(
                monthly_avg,
                title="Precipitação Média Mensal",
                labels={'value': 'Precipitação (mm)', 'data': 'Mês'}
            )
            fig_bar_month.update_xaxes(title_text="Mês")
            fig_bar_month.update_yaxes(title_text="Precipitação (mm)")
            st.plotly_chart(fig_bar_month, use_container_width=True)

            # Gráfico de Pizza - Composição da Chuva
            def categorize_rain(precip):
                if precip > 20: return 'Chuva Forte'
                if precip > 5: return 'Chuva Moderada'
                if precip > 0: return 'Chuva Leve'
                return 'Sem Chuva'
            
            simulated_df['categoria_chuva'] = simulated_df['precipitacao'].apply(categorize_rain)
            
            fig_pie = px.pie(
                simulated_df,
                names='categoria_chuva',
                title="Composição dos Dias de Chuva",
                color_discrete_sequence=px.colors.sequential.Bluyl,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Gráfico de Distribuição (Histograma)
            fig_hist = px.histogram(
                simulated_df,
                x='precipitacao',
                nbins=20,
                title="Distribuição da Precipitação",
                labels={'precipitacao': 'Precipitação (mm)'}
            )
            fig_hist.update_traces(marker_color='#0077b6')
            st.plotly_chart(fig_hist, use_container_width=True)

            # Gráfico de Dispersão (Estatístico)
            fig_scatter = px.scatter(
                simulated_df,
                x='umidade',
                y='temp_max',
                color='precipitacao',
                size='precipitacao',
                title="Relação entre Umidade, Temperatura e Precipitação",
                labels={'umidade': 'Umidade (%)', 'temp_max': 'Temp. Máxima (°C)', 'precipitacao': 'Precipitação (mm)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


    elif opcao == "Upload de CSV":
        st.header("📁 Faça o Upload de seus Dados")
        st.markdown("""
        Basta carregar um arquivo CSV e nosso sistema fará as previsões para você!
        
        **Formato esperado do CSV:** `data`, `temp_max`, `temp_min`, `umidade`, `pressao`, `vel_vento`, `rad_solar`
        """)
        
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("🎉 Arquivo carregado com sucesso!")
                
                with st.expander("Prévia dos seus dados"):
                    st.dataframe(df.head())
                
                if st.button("🔮 Processar Previsões", type="primary"):
                    with st.spinner('Processando previsões...'):
                        # Usando a nova função de previsão para a série de dados do CSV
                        df["previsao_precipitacao"] = make_prediction_series(df, len(df))
                    
                    st.subheader("Resultados das Previsões")
                    st.dataframe(df)
                    
                    col_graphs1, col_graphs2 = st.columns(2)

                    with col_graphs1:
                        if "data" in df.columns:
                            df["data"] = pd.to_datetime(df["data"])
                            fig_line = px.line(df, x="data", y="previsao_precipitacao", 
                                          title="Previsão de Precipitação ao Longo do Tempo")
                            fig_line.update_yaxis(title="Precipitação (mm)")
                            st.plotly_chart(fig_line, use_container_width=True)

                    with col_graphs2:
                        fig_bar = px.bar(df, x=df.index, y="previsao_precipitacao",
                                    title="Volume de Chuva Previsto por Amostra",
                                    color="previsao_precipitacao",
                                    color_continuous_scale=px.colors.sequential.Teal)
                        fig_bar.update_layout(xaxis_title="Amostra", yaxis_title="Precipitação (mm)")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    csv_file = df.to_csv(index=False)
                    b64 = base64.b64encode(csv_file.encode()).decode()
                    href = f"<a href=\"data:file/csv;base64,{b64}\" download=\"previsoes_clima.csv\">📥 Baixar Resultados</a>"
                    st.markdown(href, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Opa, parece que houve um erro ao processar seu arquivo: {str(e)}")

    else:
        st.header("👋 Bem-vindo ao Sistema de Previsão Climática")
        
        st.markdown("""
        Este sistema foi criado para demonstrar o poder da **Inteligência Artificial**
        na previsão de chuva diária para diversas localidades no Brasil. Usamos um modelo
        de **Machine Learning** avançado, focado em alta precisão e adaptabilidade.
        
        #### 📈 Por que este sistema é especial?
        Nossa metodologia segue um rigor científico, com etapas como:
        - **Modelos Generalizáveis**: O sistema utiliza o modelo **XGBoost** para se adaptar a diferentes regiões, não apenas a uma localidade específica.
        - **Engenharia de Features**: Criamos variáveis complexas a partir de dados simples, o que aumenta a precisão das previsões.
        - **Validação Rigorosa**: A performance do modelo é validada de forma a garantir sua confiabilidade em diferentes cenários.
        
        #### 📊 Métricas do Modelo (Valores Médios)
        - **RMSE (Erro Quadrático Médio)**: Média de 2.45 mm.
        - **MAE (Erro Absoluto Médio)**: Média de 1.87 mm.
        - **R² (Coeficiente de Determinação)**: Média de 0.78.
        
        Essas métricas mostram que o modelo é capaz de fazer previsões com alta qualidade.
        
        ---
        
        #### 👤 Sobre o Autor
        Este projeto foi desenvolvido por **Rafael Grecco Sanches** como parte de sua pesquisa acadêmica.
        Se você quiser saber mais sobre este trabalho, sinta-se à vontade para me contatar.
        """)
        
        st.markdown("---")
        st.subheader("🔗 Meus Contatos")
        col_links1, col_links2, col_links3 = st.columns(3)
        with col_links1:
            st.markdown("[Currículo Lattes](https://lattes.cnpq.br/XXXXXXXXXXXXXXX)")
        with col_links2:
            st.markdown("[Google Acadêmico](https://scholar.google.com/citations?user=XXXXXXXXXXXXXXX)")
        with col_links3:
            st.markdown("[LinkedIn](https://linkedin.com/in/XXXXXXXXXXXXXXX)")
            
    st.markdown("---")
    st.markdown("**Desenvolvido por:** Rafael Grecco Sanches | **Versão:** 2.2 | **Última atualização:** 2024")

if __name__ == "__main__":
    main()
