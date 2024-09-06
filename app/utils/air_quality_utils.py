import logging
import requests
from geopy.distance import geodesic
from config.config import air_quality_base_url, stations_endpoint


def get_nearest_station(lat, lon):
    """
    Encuentra la estación de monitoreo de calidad del aire más cercana a las coordenadas dadas.

    :param lat: Latitud de la ubicación.
    :param lon: Longitud de la ubicación.
    :return: Diccionario con la información de la estación más cercana.
    """
    url = f"{air_quality_base_url}{stations_endpoint}"

    
    logging.info(f"Fetching station data from {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        stations = response.json()["features"]
        nearest_station = None
        min_distance = float('inf')

        logging.info(f"Processing {len(stations)} stations to find the nearest one.")
        for station in stations:
            station_lat = station["geometry"]["coordinates"][1]
            station_lon = station["geometry"]["coordinates"][0]
            
            # Calcula la distancia geodésica entre las coordenadas proporcionadas y la estación
            distance = geodesic((lat, lon), (station_lat, station_lon)).km
            logging.debug(f"Station {station['properties']['name']} (ID: {station['properties']['id']}) is {distance:.2f} km away.")

            # Si la distancia es menor que la mínima encontrada hasta ahora, actualiza la estación más cercana
            if distance < min_distance:
                min_distance = distance
                nearest_station = station

        if nearest_station:
            logging.info(f"Nearest station found: {nearest_station['properties']['name']} (ID: {nearest_station['properties']['id']}) at a distance of {min_distance:.2f} km.")
        else:
            logging.warning("No nearby station found.")

        return nearest_station
    else:
        logging.error(f"Failed to fetch station data. HTTP status code: {response.status_code}")
        return None
