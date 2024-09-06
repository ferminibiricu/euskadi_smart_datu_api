import unittest
from app.utils.air_quality_utils import get_nearest_station


class TestAirQualityUtils(unittest.TestCase):

    def test_get_nearest_station(self):
        
        lat = 43.263012  # Latitud para Bilbao
        lon = -2.934985  # Longitud para Bilbao

        lat = 43.2387833 # Hernani
        lon = -1.9442591 # Hernani

        # Llamada a la función con las coordenadas de prueba
        station = get_nearest_station(lat, lon)

        # Verificar que se encontró una estación cercana
        self.assertIsNotNone(station, "No station found")
        
        # Verificar que la estación tiene los campos esperados
        self.assertIn("properties", station, "Station data malformed")
        self.assertIn("name", station["properties"], "Station does not have a name")
        self.assertIn("id", station["properties"], "Station does not have an ID")

        print(f"Nearest station: {station['properties']['name']} (ID: {station['properties']['id']})")

if __name__ == '__main__':
    unittest.main()
