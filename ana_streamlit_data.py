# ana_streamlit_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_ana_station_data(station_code: str, num_days: int, data_type: str = "precipitacao"):
    """
    Simula a busca de dados de uma estação da ANA (Agência Nacional de Águas).

    Args:
        station_code (str): Código da estação.
        num_days (int): Número de dias de dados a serem gerados.
        data_type (str): Tipo de dado ("precipitacao", "nivel", "descarga").

    Returns:
        pd.DataFrame: Um DataFrame com dados simulados da estação.
    """
    if data_type not in ["precipitacao", "nivel", "descarga"]:
        raise ValueError("Tipo de dado inválido. Use 'precipitacao', 'nivel' ou 'descarga'.")

    print(f"Simulando busca de dados da ANA para a estação {station_code}...")
    
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=num_days, freq='D')
    
    # Parâmetros simulados para diferentes tipos de dados
    if data_type == "precipitacao":
        # Chuva é mais esporádica e com picos
        base_value = 0.5
        noise_factor = 3.0
        data_values = np.random.exponential(scale=base_value, size=num_days) + np.random.normal(0, noise_factor, num_days)
        data_values = np.maximum(0, data_values)
    elif data_type == "nivel":
        # Nível do rio com variação sazonal
        base_value = 5.0
        seasonal_variation = np.sin(np.linspace(0, 2 * np.pi, num_days)) * 2.0
        data_values = base_value + seasonal_variation + np.random.normal(0, 0.5, num_days)
        data_values = np.maximum(0, data_values)
    elif data_type == "descarga":
        # Vazão com picos após chuvas
        base_value = 20.0
        data_values = np.random.lognormal(mean=np.log(base_value), sigma=0.4, size=num_days)
    
    df = pd.DataFrame({
        'data': dates,
        data_type: np.round(data_values, 2)
    })
    
    df.set_index('data', inplace=True)
    return df

def get_list_of_stations():
    """Simula a obtenção de uma lista de estações da ANA, com foco em SP."""
    return pd.DataFrame({
        'codigo': ['35520000', '35520002', '35520004', '35520005', '35520006',
                   '45010001', '45010002', '45010003', '46020004', '46020005',
                   '47030006', '47030007', '48040008', '48040009', '49050010',
                   '49050011', '50060012', '50060013', '51070014', '51070015',
                   '52080016', '52080017', '53090018', '53090019', '54100020'],
        'nome': ['Estação A - Campinas', 'Estação B - São Paulo', 'Estação C - Santos', 'Estação D - Ribeirão Preto', 'Estação E - São José do Rio Preto',
                 'Estação F - Bauru', 'Estação G - Presidente Prudente', 'Estação H - Piracicaba', 'Estação I - Jundiaí', 'Estação J - Sorocaba',
                 'Estação K - São Carlos', 'Estação L - Araraquara', 'Estação M - Marília', 'Estação N - Campinas', 'Estação O - Taubaté',
                 'Estação P - Mogi das Cruzes', 'Estação Q - São Vicente', 'Estação R - Guarulhos', 'Estação S - Botucatu', 'Estação T - Franca',
                 'Estação U - Barretos', 'Estação V - Assis', 'Estação W - Lins', 'Estação X - Araçatuba', 'Estação Y - Rio Claro'],
        'tipo_dados': ['precipitacao', 'nivel', 'precipitacao', 'descarga', 'nivel',
                       'precipitacao', 'nivel', 'precipitacao', 'descarga', 'nivel',
                       'precipitacao', 'nivel', 'precipitacao', 'descarga', 'nivel',
                       'precipitacao', 'nivel', 'precipitacao', 'descarga', 'nivel',
                       'precipitacao', 'nivel', 'precipitacao', 'descarga', 'nivel']
    })
