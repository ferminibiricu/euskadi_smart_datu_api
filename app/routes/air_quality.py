from fastapi import APIRouter, HTTPException
from typing import Optional
from app.utils.air_quality_utils import (
    get_nearest_station,
    get_all_stations_with_distance,
    get_all_stations,
    get_station_by_id,
)
from app.services.air_quality_service import (
    get_air_quality_data,
    predict_air_quality,
    calculate_air_quality_summary,
)
import logging

router = APIRouter()

@router.get("/stations_by_distance")
def get_stations_by_distance(lat: float, lon: float):
    logging.info(f"Received request to get stations ordered by distance from lat: {lat}, lon: {lon}.")
    stations = get_all_stations_with_distance(lat, lon)
    
    if not stations:
        logging.warning("No stations found or unable to fetch station data.")
        raise HTTPException(status_code=404, detail="Unable to fetch station data.")
    
    return stations

@router.get("/nearest_station")
def get_nearest_station_by_location(lat: float, lon: float):
    logging.info(f"Received request for nearest air quality station at lat: {lat}, lon: {lon}.")
    station = get_nearest_station(lat, lon)
    if not station:
        logging.warning("No nearby station found.")
        raise HTTPException(status_code=404, detail="No nearby station found.")
    
    nearest_station_code = station["properties"]["id"]
    nearest_station_name = station["properties"]["name"]
    nearest_station_lat = station["geometry"]["coordinates"][1]
    nearest_station_lon = station["geometry"]["coordinates"][0]
    distance = station["distance"]
    
    logging.info(f"Nearest station found: {nearest_station_name} (ID: {nearest_station_code}) at a distance of {distance:.2f} km.")
    return {
        "nearest_station_code": nearest_station_code,
        "nearest_station_name": nearest_station_name,
        "nearest_station_lat": nearest_station_lat,
        "nearest_station_lon": nearest_station_lon,
        "distance": round(distance, 2),
    }

@router.get("/air_quality")
def get_air_quality_by_location(lat: float, lon: float):
    logging.info(f"Received request for air quality data at lat: {lat}, lon: {lon}.")

    summary_hours = 6
    prediction_hours = 24

    try:
        # Obtener la estación más cercana
        station = get_nearest_station(lat, lon)
        if not station:
            logging.warning("No nearby station found.")
            raise HTTPException(status_code=404, detail="No nearby station found.")

        nearest_station_code = station["properties"]["id"]
        nearest_station_name = station["properties"]["name"]
        nearest_station_lat = station["geometry"]["coordinates"][1]
        nearest_station_lon = station["geometry"]["coordinates"][0]
        distance = station["distance"]
        logging.info(f"Nearest station found: {nearest_station_name} (ID: {nearest_station_code})")

        # Obtener datos para el resumen actual
        data_for_summary = get_air_quality_data(nearest_station_code, summary_hours)
        if not data_for_summary:
            logging.warning(f"Air quality data not found for the nearest station ID {nearest_station_code}.")
            current_air_quality_summary = "Datos no disponibles"
        else:
            current_air_quality_summary = calculate_air_quality_summary(data_for_summary, summary_hours)
            logging.debug(f"Current air quality summary: {current_air_quality_summary}")

        # Obtener datos para la predicción
        data_for_prediction = get_air_quality_data(nearest_station_code, prediction_hours)
        if not data_for_prediction:
            logging.warning(f"Air quality data not found for prediction for station ID {nearest_station_code}.")
            predicted_air_quality_summary = "Predicción no disponible"
        else:
            try:
                logging.debug(f"Data for prediction (station {nearest_station_code}): {data_for_prediction}")
                predicted_air_quality_summary = predict_air_quality(data_for_prediction)
                logging.debug(f"Predicted air quality summary: {predicted_air_quality_summary}")
            except Exception as e:
                logging.error(f"Error in prediction: {str(e)}", exc_info=True)
                predicted_air_quality_summary = "Predicción no disponible"

        logging.info(f"Returning air quality data and predictions for station {nearest_station_name} (ID: {nearest_station_code})")
        return {
            "lat": lat,
            "lon": lon,
            "nearest_station_code": nearest_station_code,
            "nearest_station_name": nearest_station_name,
            "nearest_station_lat": nearest_station_lat,
            "nearest_station_lon": nearest_station_lon,
            "distance": round(distance, 2),
            "current_air_quality_summary": current_air_quality_summary,
            "predicted_air_quality_summary": predicted_air_quality_summary,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"An error occurred while processing air quality data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing air quality data.")

@router.get("/air_quality_all")
def get_air_quality_for_all_stations(id: Optional[int] = None):
    logging.info(f"Received request for air quality data for all stations or for station ID: {id}.")

    summary_hours = 6
    prediction_hours = 24

    if id is not None:
        # Procesar solo la estación con el ID proporcionado
        station = get_station_by_id(id)
        if not station:
            logging.warning(f"No station found with ID {id}.")
            raise HTTPException(status_code=404, detail=f"No station found with ID {id}.")

        station_code = station["properties"]["id"]
        station_name = station["properties"]["name"]
        station_lat = station["geometry"]["coordinates"][1]
        station_lon = station["geometry"]["coordinates"][0]

        logging.info(f"Processing station: {station_name} (ID: {station_code})")

        try:
            # Obtener datos para el resumen actual
            data_for_summary = get_air_quality_data(station_code, summary_hours)
            logging.info(f"data_for_summary: station: {station_name} (summary_hours: {summary_hours})")
            if not data_for_summary:
                logging.warning(f"Air quality data not found for station ID {station_code}.")
                current_air_quality_summary = "Datos no disponibles"
            else:
                current_air_quality_summary = calculate_air_quality_summary(data_for_summary, summary_hours)
                logging.debug(f"Current air quality summary for station {station_code}: {current_air_quality_summary}")

            # Obtener datos para la predicción
            data_for_prediction = get_air_quality_data(station_code, prediction_hours)
            logging.info(f"data_for_prediction: station: {station_name} (summary_hours: {prediction_hours})")
            if not data_for_prediction:
                logging.warning(f"Air quality data not found for prediction for station ID {station_code}.")
                predicted_air_quality_summary = "Predicción no disponible"
            else:
                try:
                    logging.debug(f"Data for prediction (station {station_code}): {data_for_prediction}")
                    predicted_air_quality_summary = predict_air_quality(data_for_prediction)
                    logging.debug(f"Predicted air quality summary for station {station_code}: {predicted_air_quality_summary}")
                except Exception as e:
                    logging.error(f"Error in prediction for station {station_code}: {str(e)}", exc_info=True)
                    predicted_air_quality_summary = "Predicción no disponible"

            return {
                "station_code": station_code,
                "station_name": station_name,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "current_air_quality_summary": current_air_quality_summary,
                "predicted_air_quality_summary": predicted_air_quality_summary,
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            logging.error(f"An error occurred while processing station {station_code}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing station {station_code}.")
    else:
        # Procesar todas las estaciones
        stations = get_all_stations()
        if not stations:
            logging.warning("No stations found or unable to fetch station data.")
            raise HTTPException(status_code=404, detail="Unable to fetch station data.")

        results = []

        for station in stations["features"]:
            station_code = station["properties"]["id"]
            station_name = station["properties"]["name"]
            station_lat = station["geometry"]["coordinates"][1]
            station_lon = station["geometry"]["coordinates"][0]

            logging.info(f"Processing station: {station_name} (ID: {station_code})")

            try:
                # Obtener datos para el resumen actual
                data_for_summary = get_air_quality_data(station_code, summary_hours)
                if not data_for_summary:
                    logging.warning(f"Air quality data not found for station ID {station_code}.")
                    current_air_quality_summary = "Datos no disponibles"
                else:
                    current_air_quality_summary = calculate_air_quality_summary(data_for_summary, summary_hours)
                    logging.debug(f"Current air quality summary for station {station_code}: {current_air_quality_summary}")

                # Obtener datos para la predicción
                data_for_prediction = get_air_quality_data(station_code, prediction_hours)
                if not data_for_prediction:
                    logging.warning(f"Air quality data not found for prediction for station ID {station_code}.")
                    predicted_air_quality_summary = "Predicción no disponible"
                else:
                    try:
                        logging.debug(f"Data for prediction (station {station_code}): {data_for_prediction}")
                        predicted_air_quality_summary = predict_air_quality(data_for_prediction)
                        logging.debug(f"Predicted air quality summary for station {station_code}: {predicted_air_quality_summary}")
                    except Exception as e:
                        logging.error(f"Error in prediction for station {station_code}: {str(e)}", exc_info=True)
                        predicted_air_quality_summary = "Predicción no disponible"

                station_result = {
                    "station_code": station_code,
                    "station_name": station_name,
                    "station_lat": station_lat,
                    "station_lon": station_lon,
                    "current_air_quality_summary": current_air_quality_summary,
                    "predicted_air_quality_summary": predicted_air_quality_summary,
                }
                results.append(station_result)
            except HTTPException as he:
                raise he
            except Exception as e:
                logging.error(f"An error occurred while processing station {station_code}: {str(e)}", exc_info=True)
                # Continuar con la siguiente estación
                continue

        return results
