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

    # CORREÇÃO APLICADA AQUI: usar np.clip para garantir que os valores fiquem entre 0 e 100
    umidade_base = np.clip(umidade_base, 0, 100)

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
            "
