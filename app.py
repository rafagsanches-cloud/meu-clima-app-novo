import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import os
import warnings
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import time

warnings.filterwarnings("ignore")

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
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0096c7;
        border-color: #0096c7;
        transform: scale(1.02);
    }
    h1, h2, h3 { color: #03045e; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
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
    
    # Validar temperaturas
    if 'temp_max' in data and 'temp_min' in data:
        valid, msg = validate_temperature_range(data['temp_max'], data['temp_min'])
        if not valid:
            errors.append(msg)
    
    # Validar umidade
    if 'umidade' in data:
        if data['umidade'] < 0 or data['umidade'] > 100:
            errors.append("Umidade deve estar entre 0% e 100%")
    
    # Validar pressão
    if 'pressao' in data:
        if data['pressao'] < 800 or data['pressao'] > 1100:
            errors.append("Pressão atmosférica deve estar entre 800 hPa e 1100 hPa")
    
    # Validar velocidade do vento
    if 'vel_vento' in data:
        if data['vel_vento'] < 0 or data['vel_vento'] > 200:
            errors.append("Velocidade do vento deve estar entre 0 m/s e 200 m/s")
    
    return len(errors) == 0, errors

# --- Funções de Feature Engineering Melhoradas ---
def create_features_enhanced(df, config):
    """Versão melhorada da função de feature engineering com tratamento robusto de erros."""
    try:
        df_copy = df.copy()

        # Renomear colunas para padronização interna
        if "column_mapping" in config:
            df_copy.rename(columns=config["column_mapping"], inplace=True)

        # Converter a coluna de data para datetime com tratamento de erro
        if config["date_column"] in df_copy.columns:
            df_copy[config["date_column"]] = pd.to_datetime(df_copy[config["date_column"]], errors='coerce')
            # Remover linhas com datas inválidas
            initial_rows = len(df_copy)
            df_copy.dropna(subset=[config["date_column"]], inplace=True)
            if len(df_copy) < initial_rows:
                st.warning(f"⚠️ {initial_rows - len(df_copy)} linhas removidas devido a datas inválidas")
            
            df_copy.sort_values(config["date_column"], inplace=True)
            df_copy.set_index(config["date_column"], inplace=True)

        # Converter colunas numéricas com tratamento robusto
        for col in config["numeric_columns"]:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                # Preencher NaNs com mediana (mais robusto que média)
                if df_copy[col].isna().sum() > 0:
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)

        # Features básicas temporais
        df_copy["ano"] = df_copy.index.year
        df_copy["mes"] = df_copy.index.month
        df_copy["dia"] = df_copy.index.day
        df_copy["dia_ano"] = df_copy.index.dayofyear
        df_copy["dia_semana"] = df_copy.index.dayofweek

        # Features derivadas
        if 'temp_max' in df_copy.columns and 'temp_min' in df_copy.columns:
            df_copy["temp_media"] = (df_copy["temp_max"] + df_copy["temp_min"]) / 2
            df_copy["amplitude_termica"] = df_copy["temp_max"] - df_copy["temp_min"]

        # Features cíclicas para capturar sazonalidade
        df_copy["mes_sin"] = np.sin(2 * np.pi * df_copy["mes"] / 12)
        df_copy["mes_cos"] = np.cos(2 * np.pi * df_copy["mes"] / 12)
        df_copy["dia_ano_sin"] = np.sin(2 * np.pi * df_copy["dia_ano"] / 365)
        df_copy["dia_ano_cos"] = np.cos(2 * np.pi * df_copy["dia_ano"] / 365)

        # Médias móveis para capturar tendências
        if len(df_copy) > 7:  # Só calcular se tiver dados suficientes
            for col in ['temp_max', 'temp_min', 'umidade']:
                if col in df_copy.columns:
                    df_copy[f"{col}_ma_7d"] = df_copy[col].rolling(window=7, min_periods=1).mean()

        # Preencher NaNs restantes
        df_copy.fillna(method="bfill", inplace=True)
        df_copy.fillna(method="ffill", inplace=True)
        
        # Remover linhas que ainda tenham NaNs
        df_copy.dropna(inplace=True)

        return df_copy

    except Exception as e:
        st.error(f"Erro no processamento de features: {str(e)}")
        return df.copy()

# --- Função de Previsão Melhorada ---
def make_prediction_enhanced(df_input, num_days, municipio):
    """
    Função de previsão melhorada com lógica mais sofisticada.
    Simula um modelo mais realista baseado em padrões climáticos.
    """
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
        
        # Processar features
        df_processed = create_features_enhanced(df_input.copy(), config)
        
        if len(df_processed) == 0:
            st.error("Não foi possível processar os dados de entrada")
            return pd.Series()

        # Lógica de previsão mais sofisticada
        base_row = df_processed.iloc[-1]
        
        # Fatores climáticos baseados em conhecimento meteorológico
        temp_factor = (base_row.get('temp_max', 25) - 20) / 10  # Normalizado
        humidity_factor = (base_row.get('umidade', 60) - 50) / 50  # Normalizado
        pressure_factor = (1013 - base_row.get('pressao', 1013)) / 20  # Normalizado
        
        # Sazonalidade (baseada no mês)
        mes_atual = base_row.get('mes', 6)
        if mes_atual in [12, 1, 2]:  # Verão
            seasonal_factor = 1.5
        elif mes_atual in [6, 7, 8]:  # Inverno
            seasonal_factor = 0.3
        else:  # Outono/Primavera
            seasonal_factor = 1.0

        # Fator específico do município (simulado)
        municipio_factors = {
            "Itirapina": 1.0,
            "Santos": 1.3,  # Litoral, mais chuva
            "Cuiabá": 0.8,   # Centro-oeste, mais seco
            "Natal": 1.2,    # Nordeste litorâneo,
            "Campinas": 1.1,
            "Ribeirão Preto": 0.9,
            "São José dos Campos": 1.0,
            "Sorocaba": 1.0,
            "Piracicaba": 1.0,
            "Bauru": 0.8,
            "Araraquara": 0.9,
            "São Carlos": 1.0,
            "Franca": 0.9,
            "Presidente Prudente": 0.8,
            "Marília": 0.9,
            "Araçatuba": 0.8,
            "Botucatu": 0.9,
            "Rio Claro": 1.0,
            "Limeira": 1.0,
            "Americana": 1.0,
            "Jundiaí": 1.0,
            "Taubaté": 1.0,
            "Guaratinguetá": 1.0,
            "Jacareí": 1.0,
            "Mogi das Cruzes": 1.0,
            "Suzano": 1.1,
            "Diadema": 1.1,
            "Cuiabá": 0.8,
            "Campo Grande": 0.9,
            "Londrina": 1.0,
            "Maringá": 1.0,
            "Cascavel": 1.0,
            "Natal": 1.2,
            "João Pessoa": 1.2,
            "Recife": 1.3,
            "Salvador": 1.2,
            "Aracaju": 1.2
        }
        municipio_factor = municipio_factors.get(municipio, 1.0)

        # Calcular previsão base
        base_precipitation = (
            2.0 +  # Base mínima
            humidity_factor * 8.0 +  # Umidade é o fator mais importante
            temp_factor * 3.0 +      # Temperatura
            pressure_factor * 2.0 +  # Pressão
            seasonal_factor * 2.0    # Sazonalidade
        ) * municipio_factor

        # Gerar série temporal com variação realista
        dates = pd.date_range(start=datetime.now(), periods=num_days, freq='D')
        
        predictions = []
        for i in range(num_days):
            # Adicionar variação temporal e ruído
            day_variation = np.sin(2 * np.pi * i / 7) * 0.5  # Variação semanal
            random_noise = np.random.normal(0, 1.5)  # Ruído aleatório
            
            daily_pred = base_precipitation + day_variation + random_noise
            daily_pred = max(0, daily_pred)  # Não pode ser negativo
            predictions.append(daily_pred)

        return pd.Series(predictions, index=dates, name="previsao_precipitacao")

    except Exception as e:
        st.error(f"Erro na previsão: {str(e)}")
        return pd.Series()

# --- Função para Gerar Dados Históricos Melhorados ---
def generate_enhanced_historical_data(municipio, num_days=365):
    """Gera dados históricos mais realistas baseados no município."""
    try:
        start_date = datetime.now() - timedelta(days=num_days)
        dates = pd.date_range(start_date, periods=num_days, freq='D')

        # Parâmetros específicos por município
        municipio_params = {
            "Itirapina": {"temp_base": 22, "temp_var": 8, "humidity_base": 65, "precip_factor": 1.0},
            "Santos": {"temp_base": 25, "temp_var": 6, "humidity_base": 75, "precip_factor": 1.3},
            "Cuiabá": {"temp_base": 28, "temp_var": 10, "humidity_base": 60, "precip_factor": 0.7},
            "Natal": {"temp_base": 27, "temp_var": 4, "humidity_base": 70, "precip_factor": 1.1},
            "Campinas": {"temp_base": 23, "temp_var": 7, "humidity_base": 68, "precip_factor": 1.0},
            "Ribeirão Preto": {"temp_base": 26, "temp_var": 8, "humidity_base": 62, "precip_factor": 0.9},
            "São José dos Campos": {"temp_base": 22, "temp_var": 7, "humidity_base": 70, "precip_factor": 1.0},
            "Sorocaba": {"temp_base": 23, "temp_var": 7, "humidity_base": 69, "precip_factor": 1.0},
            "Piracicaba": {"temp_base": 24, "temp_var": 8, "humidity_base": 67, "precip_factor": 1.0},
            "Bauru": {"temp_base": 25, "temp_var": 9, "humidity_base": 63, "precip_factor": 0.8},
            "Araraquara": {"temp_base": 24, "temp_var": 8, "humidity_base": 65, "precip_factor": 0.9},
            "São Carlos": {"temp_base": 23, "temp_var": 7, "humidity_base": 68, "precip_factor": 1.0},
            "Franca": {"temp_base": 23, "temp_var": 8, "humidity_base": 64, "precip_factor": 0.9},
            "Presidente Prudente": {"temp_base": 26, "temp_var": 9, "humidity_base": 61, "precip_factor": 0.8},
            "Marília": {"temp_base": 24, "temp_var": 8, "humidity_base": 65, "precip_factor": 0.9},
            "Araçatuba": {"temp_base": 27, "temp_var": 9, "humidity_base": 60, "precip_factor": 0.8},
            "Botucatu": {"temp_base": 23, "temp_var": 8, "humidity_base": 66, "precip_factor": 0.9},
            "Rio Claro": {"temp_base": 23, "temp_var": 7, "humidity_base": 67, "precip_factor": 1.0},
            "Limeira": {"temp_base": 24, "temp_var": 7, "humidity_base": 68, "precip_factor": 1.0},
            "Americana": {"temp_base": 24, "temp_var": 7, "humidity_base": 68, "precip_factor": 1.0},
            "Jundiaí": {"temp_base": 23, "temp_var": 7, "humidity_base": 69, "precip_factor": 1.0},
            "Taubaté": {"temp
