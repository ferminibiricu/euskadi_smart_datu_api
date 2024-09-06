import yaml
import os
import logging
from datetime import datetime, timedelta

# Ruta al archivo config.yaml
config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Leer el archivo YAML
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Acceso a las configuraciones
air_quality_base_url = config['air_quality']['base_url']
stations_endpoint = config['air_quality']['stations_endpoint']
hourly_measurements_endpoint = config['air_quality']['hourly_measurements_endpoint']
exclusion_threshold = config['air_quality']['exclusion_threshold']

logging_level = config['logging']['level']
logging_file = config['logging']['file']

# Creación de la carpeta log si no existe
log_dir = os.path.dirname(logging_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configurar logging
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logging_file),
        logging.StreamHandler()
    ]
)

# Función para obtener la URL de mediciones horarias formateada
def get_hourly_measurements_url(station_code: str, hours: int):
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M')

    return air_quality_base_url + hourly_measurements_endpoint.format(
        station_code=station_code,
        start_date=start_date_str,
        end_date=end_date_str
    )
