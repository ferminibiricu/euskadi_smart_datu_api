import logging
import requests
from geopy.distance import geodesic
from config.config import air_quality_base_url, stations_endpoint

def get_nearest_station(lat, lon):
    """
    Encuentra la estación de monitoreo de calidad del aire más cercana a las coordenadas dadas.

    :param lat: Latitud de la ubicación.
    :param lon: Longitud de la ubicación.
    :return: Diccionario con la información de la estación más cercana y la distancia.
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
            # Añadir la distancia al diccionario de la estación
            nearest_station['distance'] = round(min_distance, 2)
        else:
            logging.warning("No nearby station found.")

        return nearest_station
    else:
        logging.error(f"Failed to fetch station data. HTTP status code: {response.status_code}")
        return None

def get_all_stations_with_distance(lat, lon):
    """
    Obtiene una lista de todas las estaciones con sus distancias desde las coordenadas proporcionadas.

    :param lat: Latitud de la ubicación.
    :param lon: Longitud de la ubicación.
    :return: Lista de diccionarios con código, nombre, latitud, longitud y distancia de cada estación.
    """
    url = f"{air_quality_base_url}{stations_endpoint}"

    logging.info(f"Fetching station data from {url}")
    response = requests.get(url)

    if response.status_code == 200:
        stations = response.json()["features"]
        station_distances = []

        logging.info(f"Processing {len(stations)} stations to calculate distances.")
        for station in stations:
            station_lat = station["geometry"]["coordinates"][1]
            station_lon = station["geometry"]["coordinates"][0]
            
            # Calcula la distancia geodésica entre las coordenadas proporcionadas y la estación
            distance = geodesic((lat, lon), (station_lat, station_lon)).km
            station_info = {
                "station_code": station["properties"]["id"],
                "station_name": station["properties"]["name"],
                "station_lat": station_lat,  # Añadir latitud de la estación
                "station_lon": station_lon,  # Añadir longitud de la estación
                "distance": round(distance, 2)
            }
            station_distances.append(station_info)

        # Ordenar las estaciones por distancia
        station_distances = sorted(station_distances, key=lambda x: x["distance"])

        logging.info("Stations sorted by distance.")
        return station_distances
    else:
        logging.error(f"Failed to fetch station data. HTTP status code: {response.status_code}")
        return []

def get_all_stations():
    """
    Obtiene una lista de todas las estaciones de calidad del aire.

    :return: Diccionario con las estaciones.
    """
    url = f"{air_quality_base_url}{stations_endpoint}"

    logging.info(f"Fetching all station data from {url}")
    response = requests.get(url)

    if response.status_code == 200:
        stations = response.json()
        return stations
    else:
        logging.error(f"Failed to fetch station data. HTTP status code: {response.status_code}")
        return None

def get_station_by_id(station_id):
    """
    Obtiene la información de una estación por su ID.

    :param station_id: ID de la estación.
    :return: Diccionario con la información de la estación o None si no se encuentra.
    """
    stations = get_all_stations()
    if stations and "features" in stations:
        for station in stations["features"]:
            if station["properties"]["id"] == str(station_id):
                return station
    logging.warning(f"Station with ID {station_id} not found.")
    return None
