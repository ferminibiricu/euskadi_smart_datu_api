import logging
from fastapi import FastAPI
from app.routes import air_quality, transport

# Configuración básica de logging
logging.basicConfig(
    filename='app.log',  # Nombre del archivo de log
    level=logging.INFO,  # Nivel de log, puede ser DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(title="Euskadi Smart Datu API")

# Incluir los routers para las rutas de calidad del aire y transporte
app.include_router(air_quality.router, prefix="/api/v1/environment")
app.include_router(transport.router, prefix="/api/v1/public_transport")

@app.get("/")
def read_root():
    logging.info("Root endpoint accessed")  # Log para acceso a la raíz
    return {"message": "Welcome to Euskadi Smart Datu API"}

# Puedes agregar logs similares en otros endpoints o funciones según lo necesites.
