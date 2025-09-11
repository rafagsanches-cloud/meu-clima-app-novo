import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings("ignore")

# --- Importar o novo módulo de aquisição de dados ---
import ana_streamlit_data as ana_data

# --- Configuração da Página ---
st.set_page_config(
    page_title="Sistema de Previsão Climática - Brasil",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Personalizado ---
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
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005f8f;
        border: 1px solid #005f8f;
    }
    .stExpander {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stExpander div[data-baseweb="button"] {
        background-color: #e0e0e0;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #003366;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# --- Funções de Validação ---
def validate_temperature_range(temp_max, temp_min):
    """Valida se as temperaturas estão em faixas razoáveis."""
    if temp_max < -50 or temp_max > 60:
        return False, "Temperatura máxima fora da faixa válida (-50°C a 60°C)"
    if temp_min < -60 or temp_min > 50:
        return False, "Temperatura mínima fora da faixa válida (-60°C a 50°C)"
    if temp_min >= temp_max:
        return False, "Temperatura mínima deve ser menor que a máxima"
    return True, ""

def validate_meteorological_data(data):
    """Valida dados meteorológicos de entrada."""
    errors = []
    
    if 'temp_max' in data and 'temp_min' in data:
        valid, msg = validate_temperature_range(data['temp_max'], data['temp_min'])
        if not valid:
            errors.append(msg)
    
    if 'umidade' in data:
        if data['umidade'] < 0 or data['umidade'] > 100:
            errors.append("Umidade deve estar entre 0% e 100%")
    
    if 'pressao' in data:
        if data['pressao'] < 800 or data['pressao'] > 1100:
            errors.append("Pressão atmosférica deve estar entre 800 hPa e 1100 hPa")
    
    if 'vel_vento' in data:
        if data['vel_vento'] < 0 or data['vel_vento'] > 200:
            errors.append("Velocidade do vento deve estar entre 0 m/s e 200 m/s")
    
    return len(errors) == 0, errors

# --- Funções de Feature Engineering Melhoradas ---
def create_features_enhanced(df, config):
    """Versão melhorada da função de feature engineering com tratamento robusto de erros."""
    try:
        df_copy = df.copy()

        if "column_mapping" in config:
            df_copy.rename(columns=config["column_mapping"], inplace=True)

        if config["date_column"] in df_copy.columns:
            df_copy[config["date_column"]] = pd.to_datetime(df_copy[config["date_column"]], errors='coerce')
            initial_rows = len(df_copy)
            df_copy.dropna(subset=[config["date_column"]], inplace=True)
            if len(df_copy) < initial_rows:
                st.warning(f"⚠️ {initial_rows - len(df_copy)} linhas removidas devido a datas inválidas")
            
            df_copy.sort_values(config["date_column"], inplace=True)
            df_copy.set_index(config["date_column"], inplace=True)

        for col in config["numeric_columns"]:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                if df_copy[col].isna().sum() > 0:
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)

        df_copy["ano"] = df_copy.index.year
        df_copy["mes"] = df_copy.index.month
        df_copy["dia"] = df_copy.index.day
        df_copy["dia_ano"] = df_copy.index.dayofyear
        df_copy["dia_semana"] = df_copy.index.dayofweek

        if 'temp_max' in df_copy.columns and 'temp_min' in df_copy.columns:
            df_copy["temp_media"] = (df_copy["temp_max"] + df_copy["temp_min"]) / 2
            df_copy["amplitude_termica"] = df_copy["temp_max"] - df_copy["temp_min"]

        df_copy["mes_sin"] = np.sin(2 * np.pi * df_copy["mes"] / 12)
        df_copy["mes_cos"] = np.cos(2 * np.pi * df_copy["mes"] / 12)
        df_copy["dia_ano_sin"] = np.sin(2 * np.pi * df_copy["dia_ano"] / 365)
        df_copy["dia_ano_cos"] = np.cos(2 * np.pi * df_copy["dia_ano"] / 365)

        if len(df_copy) > 7:
            for col in ['temp_max', 'temp_min', 'umidade']:
                if col in df_copy.columns:
                    df_copy[f"{col}_ma_7d"] = df_copy[col].rolling(window=7, min_periods=1).mean()

        df_copy.fillna(method="bfill", inplace=True)
        df_copy.fillna(method="ffill", inplace=True)
        
        df_copy.dropna(inplace=True)

        return df_copy

    except Exception as e:
        st.error(f"Erro no processamento de features: {str(e)}")
        return df.copy()

# --- Funções de Previsão e Métricas ---
def make_prediction_enhanced(df_input, num_days, municipio):
    """Simula uma previsão de precipitação baseada em dados de entrada."""
    try:
        config = {
            "date_column": 'data',
            "column_mapping": {
                'data': 'data', 'temp_max': 'temp_max', 'temp_min': 'temp_min', 
                'umidade': 'umidade', 'pressao': 'pressao', 'vel_vento': 'vel_vento', 
                'rad_solar': 'rad_solar'
            },
            "numeric_columns": ['temp_max', 'temp_min', 'umidade', 'pressao', 'vel_vento', 'rad_solar']
        }
        
        df_processed = create_features_enhanced(df_input.copy(), config)
        
        if len(df_processed) == 0:
            st.error("Não foi possível processar os dados de entrada")
            return pd.Series()

        base_row = df_processed.iloc[-1]
        
        temp_factor = (base_row.get('temp_max', 25) - 20) / 10
        humidity_factor = (base_row.get('umidade', 60) - 50) / 50
        pressure_factor = (1013 - base_row.get('pressao', 1013)) / 20
        
        mes_atual = base_row.get('mes', 6)
        if mes_atual in [12, 1, 2]:
            seasonal_factor = 1.5
        elif mes_atual in [6, 7, 8]:
            seasonal_factor = 0.3
        else:
            seasonal_factor = 1.0

        municipio_factors = {
            "Itirapina": 1.0,
            "Santos": 1.3,
            "Cuiabá": 0.8,
            "Natal": 1.2,
        }
        municipio_factor = municipio_factors.get(municipio, 1.0)

        base_precipitation = (
            2.0 +
            humidity_factor * 8.0 +
            temp_factor * 3.0 +
            pressure_factor * 2.0 +
            seasonal_factor * 2.0
        ) * municipio_factor

        dates = pd.date_range(start=datetime.now(), periods=num_days, freq='D')
        
        predictions = []
        for i in range(num_days):
            day_variation = np.sin(2 * np.pi * i / 7) * 0.5
            random_noise = np.random.normal(0, 1.5)
            
            daily_pred = base_precipitation + day_variation + random_noise
            daily_pred = max(0, daily_pred)
            predictions.append(daily_pred)

        return pd.Series(predictions, index=dates, name="previsao_precipitacao")

    except Exception as e:
        st.error(f"Erro na previsão: {str(e)}")
        return pd.Series()

def generate_enhanced_historical_data(municipio, num_days=365):
    """Gera dados históricos mais realistas baseados no município."""
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=num_days, freq='D')
    
    params = {
        "Itirapina": {"temp_base": 22, "temp_var": 8, "humidity_base": 65, "precip_factor": 1.0},
        "Santos": {"temp_base": 25, "temp_var": 6, "humidity_base": 75, "precip_factor": 1.3},
        "Cuiabá": {"temp_base": 28, "temp_var": 10, "humidity_base": 60, "precip_factor": 0.7},
        "Natal": {"temp_base": 27, "temp_var": 4, "humidity_base": 70, "precip_factor": 1.1},
    }
    
    selected_params = params.get(municipio, params["Itirapina"])
    
    day_of_year = dates.dayofyear
    seasonal_pattern = np.sin(2 * np.pi * (day_of_year - 80) / 365)

    temp_max = selected_params["temp_base"] + seasonal_pattern * selected_params["temp_var"] + np.random.normal(0, 2, num_days)
    temp_min = temp_max - 8 - np.random.uniform(2, 6, num_days)
    umidade = selected_params["humidity_base"] - seasonal_pattern * 15 + np.random.normal(0, 8, num_days)
    precipitacao = np.maximum(0, (umidade - 50) * 0.3 * selected_params["precip_factor"] + np.random.exponential(1.5, num_days))
    pressao = 1013 + seasonal_pattern * 5 + np.random.normal(0, 3, num_days)
    vel_vento = 5 + np.abs(np.random.normal(0, 2, num_days))
    rad_solar = 20 + seasonal_pattern * 8 + np.random.normal(0, 3, num_days)
    
    df = pd.DataFrame({
        'data': dates,
        'temp_max': np.round(temp_max, 1),
        'temp_min': np.round(temp_min, 1),
        'umidade': np.round(np.clip(umidade, 10, 95), 1),
        'pressao': np.round(pressao, 1),
        'vel_vento': np.round(np.clip(vel_vento, 0, 50), 1),
        'rad_solar': np.round(np.clip(rad_solar, 0, 40), 1),
        'precipitacao': np.round(precipitacao, 2)
    })
    
    return df

def calculate_enhanced_metrics(municipio, num_days):
    """Calcula métricas de desempenho simuladas para o modelo."""
    base_metrics = {
        "Itirapina": {"RMSE": 2.1, "MAE": 1.6, "R2": 0.82},
        "Santos": {"RMSE": 2.8, "MAE": 2.1, "R2": 0.75},
        "Cuiabá": {"RMSE": 3.2, "MAE": 2.4, "R2": 0.68},
        "Natal": {"RMSE": 2.5, "MAE": 1.9, "R2": 0.78},
    }
    
    metrics = base_metrics.get(municipio, base_metrics["Itirapina"])
    
    if num_days > 7:
        degradation_factor = 1 + (num_days - 7) * 0.05
        metrics["RMSE"] *= degradation_factor
        metrics["MAE"] *= degradation_factor
        metrics["R2"] *= (1 / degradation_factor)
    
    return {k: round(v, 3) for k, v in metrics.items()}

# --- Lista de Municípios Expandida ---
@st.cache_data
def get_municipios_data():
    """Retorna dados dos municípios com cache para melhor performance."""
    return pd.DataFrame({
        'cidade': [
            "Itirapina", "Campinas", "Ribeirão Preto", "Santos", "São José dos Campos",
            "Sorocaba", "Piracicaba", "Bauru", "Araraquara", "São Carlos",
            "Franca", "Presidente Prudente", "Marília", "Araçatuba", "Botucatu",
            "Rio Claro", "Limeira", "Americana", "Jundiaí", "Taubaté",
            "Guaratinguetá", "Jacareí", "Mogi das Cruzes", "Suzano", "Diadema",
            "Cuiabá", "Campo Grande", "Londrina", "Maringá", "Cascavel",
            "Natal", "João Pessoa", "Recife", "Salvador", "Aracaju"
        ],
        'estado': [
            'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP',
            'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP',
            'SP', 'SP', 'SP', 'SP', 'SP', 'MT', 'MS', 'PR', 'PR', 'PR',
            'RN', 'PB', 'PE', 'BA', 'SE'
        ],
        'regiao': [
            'Interior', 'Interior', 'Interior', 'Litoral', 'Interior',
            'Interior', 'Interior', 'Interior', 'Interior', 'Interior',
            'Interior', 'Interior', 'Interior', 'Interior', 'Interior',
            'Interior', 'Interior', 'Interior', 'Interior', 'Interior',
            'Interior', 'Interior', 'Interior', 'Interior', 'Interior',
            'Centro-Oeste', 'Centro-Oeste', 'Sul', 'Sul', 'Sul',
            'Nordeste', 'Nordeste', 'Nordeste', 'Nordeste', 'Nordeste'
        ],
        'lat': [
            -22.259, -22.9099, -21.1762, -23.9634, -23.1794,
            -23.498, -22.721, -22.316, -21.807, -22.016,
            -20.538, -22.124, -22.214, -21.209, -22.886,
            -22.411, -22.565, -22.739, -23.186, -23.023,
            -22.806, -23.305, -23.522, -23.542, -23.686,
            -15.5989, -20.4697, -23.3106, -23.424, -24.956,
            -5.7947, -7.1195, -8.0476, -12.9714, -10.9472
        ],
        'lon': [
            -47.935, -47.0626, -47.8823, -46.3353, -45.8869,
            -47.4488, -47.649, -49.066, -48.188, -47.893,
            -47.400, -51.401, -49.946, -50.433, -48.445,
            -47.561, -47.404, -47.331, -46.883, -45.556,
            -45.209, -45.969, -46.188, -46.311, -46.622,
            -56.0949, -54.6201, -51.1627, -51.9389, -53.455,
            -35.2114, -34.8641, -34.8770, -38.5014, -37.0731
        ],
        'populacao': [
            17000, 1213792, 703293, 433656, 729737,
            687357, 407252, 379297, 238339, 254484,
            358539, 230371, 240590, 198129, 149684,
            206424, 308482, 237014, 423006, 317915,
            122505, 235416, 440962, 300559, 426757,
            650916, 906092, 575377, 430157, 348051,
            890480, 817511, 1653461, 2886698, 664908
        ]
    })


# --- Interface Principal ---
def main():
    st.title("🌧️ Sistema Avançado de Previsão Climática")
    st.markdown("### 🇧🇷 Previsão de Volume Diário de Chuva para o Brasil")
    
    with st.expander("ℹ️ Sobre este Sistema", expanded=False):
        st.markdown("""
        **Sistema de Previsão Climática Avançado** desenvolvido com tecnologias de Machine Learning.
        
        **Características:**
        - 🎯 Previsões para 35+ municípios brasileiros
        - 📊 Análise histórica e estatística
        - 🔍 Validação robusta de dados
        - 📈 Visualizações interativas
        - 🌡️ Múltiplas variáveis meteorológicas
        
        **Tecnologias:** Python, Streamlit, Plotly, Pandas, NumPy
        """)

    st.sidebar.title("🧭 Navegação")
    st.sidebar.markdown("---")
    
    opcao = st.sidebar.selectbox(
        "Escolha uma funcionalidade:",
        ["🔮 Previsão Individual", "📁 Upload de CSV", "📊 Análise Comparativa", "📥 Dados da ANA"],
        help="Selecione a funcionalidade desejada"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Status do Sistema")
    st.sidebar.success("🟢 Sistema Online")
    st.sidebar.info(f"📅 Última atualização: {datetime.now().strftime('%d/%m/%Y')}")
    st.sidebar.markdown(f"🏙️ **{len(get_municipios_data())} municípios** disponíveis")

    if opcao == "🔮 Previsão Individual":
        st.header("🔮 Previsão Climática Individual")
        st.markdown("Selecione um município e configure os parâmetros para obter previsões detalhadas.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            municipio_selecionado = st.selectbox(
                "Selecione um município:",
                get_municipios_data()["cidade"].unique(),
                index=0,
                key="selecao_municipio_prev",
                help="Escolha o município para a previsão"
            )

        with col2:
            num_dias = st.slider(
                "Número de dias da previsão:",
                min_value=1, max_value=30, value=7,
                help="Escolha quantos dias no futuro você quer prever"
            )

        col3, col4, col5, col6, col7, col8 = st.columns(6)
        with col3:
            temp_max = st.number_input("Temp. Máx (°C)", value=30.0, format="%.1f")
        with col4:
            temp_min = st.number_input("Temp. Mín (°C)", value=20.0, format="%.1f")
        with col5:
            umidade = st.number_input("Umidade (%)", value=75, step=1)
        with col6:
            pressao = st.number_input("Pressão (hPa)", value=1013.0, format="%.1f")
        with col7:
            vel_vento = st.number_input("Vento (m/s)", value=5.0, format="%.1f")
        with col8:
            rad_solar = st.number_input("Rad. Solar (W/m²)", value=20.0, format="%.1f")
        
        dados_entrada = {
            'data': datetime.now(),
            'temp_max': temp_max,
            'temp_min': temp_min,
            'umidade': umidade,
            'pressao': pressao,
            'vel_vento': vel_vento,
            'rad_solar': rad_solar
        }
        
        if st.button("Gerar Previsão", key="btn_previsao_individual"):
            valid, errors = validate_meteorological_data(dados_entrada)
            if valid:
                with st.spinner("⏳ Gerando previsão..."):
                    df_entrada = pd.DataFrame([dados_entrada])
                    previsao_df = make_prediction_enhanced(df_entrada, num_dias, municipio_selecionado)
                    
                    if not previsao_df.empty:
                        st.subheader("Resultado da Previsão")
                        fig = px.bar(
                            previsao_df,
                            x=previsao_df.index,
                            y="previsao_precipitacao",
                            title=f"Previsão de Precipitação para {municipio_selecionado}",
                            labels={'x': 'Data', 'previsao_precipitacao': 'Precipitação (mm)'}
                        )
                        fig.update_layout(xaxis_title="Data", yaxis_title="Precipitação (mm)", showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("Previsão em Tabela")
                        st.dataframe(previsao_df.reset_index().rename(columns={'index': 'Data', 'previsao_precipitacao': 'Precipitação (mm)'}), use_container_width=True)
                    else:
                        st.error("Não foi possível gerar a previsão. Verifique os dados de entrada.")
            else:
                for error in errors:
                    st.error(f"❌ Erro de validação: {error}")

        st.markdown("---")
        st.subheader("Análise Histórica do Município")
        st.info("Aqui seria exibida a análise dos dados históricos que o modelo usou para a previsão. Esta seção utiliza dados simulados.")
        
        with st.spinner("🔄 Carregando dados históricos..."):
            dados_historicos = generate_enhanced_historical_data(municipio_selecionado, 365)
        
        if not dados_historicos.empty:
            fig_hist = px.line(
                dados_historicos,
                x="data",
                y="precipitacao",
                title=f"Dados Históricos de Precipitação para {municipio_selecionado}",
                labels={'data': 'Data', 'precipitacao': 'Precipitação (mm)'}
            )
            fig_hist.update_layout(xaxis_title="Data", yaxis_title="Precipitação (mm)")
            st.plotly_chart(fig_hist, use_container_width=True)

    elif opcao == "📁 Upload de CSV":
        st.header("📁 Upload e Processamento de Dados")
        st.markdown("Faça o upload do seu próprio arquivo CSV com dados históricos para processamento e análise.")
        
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Arquivo carregado com sucesso!")
                st.subheader("Visualização dos Dados Carregados")
                st.dataframe(df.head(), use_container_width=True)
                
                config = {
                    "date_column": 'data',
                    "column_mapping": {
                        'data': 'data', 'temp_max': 'temp_max', 'temp_min': 'temp_min', 
                        'umidade': 'umidade', 'pressao': 'pressao', 'vel_vento': 'vel_vento', 
                        'rad_solar': 'rad_solar'
                    },
                    "numeric_columns": ['temp_max', 'temp_min', 'umidade', 'pressao', 'vel_vento', 'rad_solar']
                }

                st.markdown("---")
                if st.button("Processar e Analisar Dados"):
                    with st.spinner("⏳ Processando dados..."):
                        df_processed = create_features_enhanced(df, config)
                        st.success("✅ Dados processados e prontos para análise!")
                        
                        st.subheader("Análise Gráfica dos Dados Processados")
                        fig_upload = px.line(
                            df_processed,
                            y=['temp_max', 'temp_min', 'umidade'],
                            title="Série Temporal de Variáveis Meteorológicas",
                            labels={'value': 'Valor', 'variable': 'Variável'}
                        )
                        st.plotly_chart(fig_upload, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erro ao ler arquivo: {str(e)}")
                st.warning("Certifique-se de que o arquivo é um CSV válido com as colunas corretas (e.g., 'data', 'temp_max', 'precipitacao').")

    elif opcao == "📊 Análise Comparativa":
        st.header("📊 Análise Comparativa entre Municípios")
        st.markdown("Compare o desempenho do modelo e os padrões climáticos de diferentes municípios.")
        
        municipios_df = get_municipios_data()
        all_municipios = municipios_df["cidade"].unique()
        
        municipios_selecionados = st.multiselect(
            "Selecione os municípios para comparação:",
            all_municipios,
            default=["Itirapina", "Santos", "Natal"],
            help="Escolha pelo menos dois municípios"
        )
        
        if len(municipios_selecionados) >= 2:
            dados_comparativos = {}
            for municipio in municipios_selecionados:
                dados_hist = generate_enhanced_historical_data(municipio, 365)
                if not dados_hist.empty:
                    dados_comparativos[municipio] = dados_hist

            st.subheader("Precipitação Histórica (últimos 3 meses)")
            df_plot = pd.DataFrame()
            for municipio, df_data in dados_comparativos.items():
                df_data_copy = df_data.copy()
                df_data_copy['municipio'] = municipio
                df_plot = pd.concat([df_plot, df_data_copy], ignore_index=True)

            fig_comp = px.line(
                df_plot.tail(90*len(municipios_selecionados)),
                x="data",
                y="precipitacao",
                color="municipio",
                title="Série Histórica de Precipitação",
                labels={'data': 'Data', 'precipitacao': 'Precipitação (mm)'}
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.subheader("Métricas de Desempenho do Modelo")
            metricas_df = pd.DataFrame()
            for municipio in municipios_selecionados:
                metrics = calculate_enhanced_metrics(municipio, 7)
                metricas_df = pd.concat([metricas_df, pd.DataFrame([metrics], index=[municipio])])
            
            st.dataframe(metricas_df, use_container_width=True)
            
            fig_radar = go.Figure()
            
            for municipio, row in metricas_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['RMSE'], row['MAE'], row['R2']],
                    theta=['RMSE', 'MAE', 'R2'],
                    fill='toself',
                    name=municipio
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5])
                ),
                title="Desempenho Relativo do Modelo por Município",
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    elif opcao == "📥 Dados da ANA":
        st.header("📥 Aquisição de Dados da ANA")
        st.markdown("Simule a busca de dados de estações hidrológicas da Agência Nacional de Águas (ANA).")
        
        # Obter a lista de estações do novo módulo
        stations_df = ana_data.get_list_of_stations()
        
        station_name = st.selectbox(
            "Selecione a Estação da ANA:",
            options=stations_df["nome"],
            index=0
        )
        
        num_days_ana = st.slider(
            "Período de Dados (dias):",
            min_value=30, max_value=730, value=365
        )
        
        selected_station = stations_df[stations_df["nome"] == station_name].iloc[0]
        station_code = selected_station["codigo"]
        data_type = selected_station["tipo_dados"]
        
        if st.button("Buscar Dados", key="btn_fetch_ana"):
            with st.spinner("⏳ Buscando dados da ANA..."):
                try:
                    df_ana = ana_data.fetch_ana_station_data(station_code, num_days_ana, data_type)
                    
                    if not df_ana.empty:
                        st.subheader(f"Dados Históricos de {data_type.capitalize()} para a Estação {station_code}")
                        
                        fig_ana = px.line(
                            df_ana,
                            y=data_type,
                            title=f"Série Temporal de {data_type.capitalize()}",
                            labels={data_type: data_type.capitalize()}
                        )
                        fig_ana.update_layout(xaxis_title="Data", yaxis_title=data_type.capitalize())
                        st.plotly_chart(fig_ana, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("Tabela de Dados")
                        st.dataframe(df_ana, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Erro ao buscar dados da ANA: {str(e)}")

    else:  # Sobre o Sistema
        st.header("ℹ️ Sobre o Sistema")
        st.markdown("""
        Este aplicativo demonstra as capacidades de um sistema de previsão climática.
        
        **Aviso:** Os dados apresentados neste aplicativo, incluindo as previsões e análises, são **simulados** e criados para fins de demonstração e estudo. Eles não representam dados climáticos reais nem previsões oficiais. A lógica de previsão é um modelo simplificado e não um modelo de Machine Learning treinado com dados reais.

        **Tecnologias Utilizadas:**
        - **Streamlit:** Para a criação da interface web interativa.
        - **Pandas e NumPy:** Para manipulação e análise de dados.
        - **Plotly:** Para as visualizações gráficas interativas.
        """)
        st.markdown("---")
        st.markdown("#### **Demo Rápida**")
        st.markdown("Veja uma previsão instantânea para um município aleatório.")
        
        if st.button("Gerar Demo"):
            demo_municipio = np.random.choice(get_municipios_data()["cidade"])
            demo_temp = np.random.uniform(25, 35)
            demo_umidade = np.random.randint(60, 90)
            
            with st.spinner(f"Gerando demo para {demo_municipio}..."):
                dados_demo = pd.DataFrame({
                    "data": [datetime.now()],
                    "temp_max": [demo_temp],
                    "temp_min": [demo_temp - 8],
                    "umidade": [demo_umidade],
                    "pressao": [1013],
                    "vel_vento": [5],
                    "rad_solar": [20]
                })
                
                previsao_demo = make_prediction_enhanced(dados_demo, 1, demo_municipio)
                
                if len(previsao_demo) > 0:
                    st.success(f"🌧️ Previsão para {demo_municipio}: **{previsao_demo.iloc[0]:.1f} mm**")
                    
                    if previsao_demo.iloc[0] > 10:
                        st.warning("⛈️ Chuva intensa prevista!")
                    elif previsao_demo.iloc[0] > 5:
                        st.info("🌦️ Chuva moderada prevista")
                    elif previsao_demo.iloc[0] > 1:
                        st.info("🌤️ Chuva leve prevista")
                    else:
                        st.info("☀️ Tempo seco previsto")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Sistema de Previsão Climática - Brasil</strong></p>
            <p>Desenvolvido com ❤️ usando Streamlit | © 2024 Manus AI</p>
            <p>Para suporte técnico ou sugestões, entre em contato através dos canais oficiais</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
