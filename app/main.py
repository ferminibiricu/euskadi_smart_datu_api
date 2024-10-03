import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.routes import air_quality

# Configuración básica de logging
logging.basicConfig(
    filename='app.log',  # Nombre del archivo de log
    level=logging.INFO,  # Nivel de log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(title="Euskadi Smart Datu API")

# Incluir los routers para las rutas de calidad del aire
app.include_router(air_quality.router, prefix="/api/v1/environment")

@app.get("/", response_class=HTMLResponse)
def read_root():
    logging.info("Root endpoint accessed")  # Log para acceso a la raíz
    html_content = """
    <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Euskadi Smart Datu API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f7f7f7;
                    margin: 0;
                    padding: 0;
                }
                header {
                    background-color: #00448C;
                    color: white;
                    padding: 20px 0;
                    text-align: center;
                }
                header h1 {
                    margin: 0;
                    font-size: 2.5em;
                }
                .content {
                    padding: 20px;
                    text-align: center;
                }
                .content p {
                    font-size: 1.2em;
                    color: #333;
                }
                .buttons {
                    margin-top: 30px;
                }
                .buttons a {
                    display: inline-block;
                    padding: 15px 25px;
                    margin: 10px;
                    font-size: 1em;
                    color: white;
                    background-color: #005BB5;
                    text-decoration: none;
                    border-radius: 5px;
                    transition: background-color 0.3s ease;
                }
                .buttons a:hover {
                    background-color: #00448C;
                }
                footer {
                    background-color: #00448C;
                    color: white;
                    padding: 10px;
                    text-align: center;
                    position: fixed;
                    width: 100%;
                    bottom: 0;
                }
            </style>
        </head>
        <body>
            <header>
                <h1>Euskadi Smart Datu API</h1>
            </header>
            <div class="content">
                <p>Ongi etorri / Bienvenid@s / Benvinguts / Benvidos / Welcome / Bienvenue / Bem-vindos / Willkommen / Benvenuti</p>
                <div class="buttons">
                    <a href="/docs">Swagger UI Docs</a>
                    <a href="/redoc">Redoc API Docs</a>
                </div>
            </div>
            <footer>
                <p>&copy; 2024 Euskadi Smart Datu API.</p>
            </footer>
        </body>
    </html>
    """
    return html_content
