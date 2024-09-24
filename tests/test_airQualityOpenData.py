import requests
import logging
from datetime import datetime, timedelta

# URL base de la API de calidad del aire en Euskadi
air_quality_base_url = "https://api.euskadi.eus/"
hourly_measurements_endpoint = "air-quality/measurements/hourly/stations/{station_code}/from/{start_time}/to/{end_time}?lang=SPANISH"

# Función para calcular el rango de tiempo de las últimas X horas
def calculate_time_range(hours: int = 2):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")
    return start_time_str, end_time_str

# Función para hacer la llamada a la API
def fetch_air_quality_data(station_code: str, hours: int = 6):
    start_time, end_time = calculate_time_range(hours)
    url = f"{air_quality_base_url}{hourly_measurements_endpoint.format(station_code=station_code, start_time=start_time, end_time=end_time)}"
    
    logging.info(f"Fetching air quality data from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Data successfully retrieved for station ID {station_code}")
        return data
    except requests.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return None
    except requests.RequestException as e:
        logging.error(f"Request error occurred: {e}")
        return None

# Parámetros de prueba
station_code = "68"  # 60 MAZARREDO - 68 HERNANI
hours = 6  # Últimas 6 horas

# Ejecutar la prueba
logging.basicConfig(level=logging.INFO)
data = fetch_air_quality_data(station_code, hours)

if data:
    print(f"Datos de calidad del aire recibidos: {data}")
else:
    print("No se recibieron datos de calidad del aire o hubo un error.")
