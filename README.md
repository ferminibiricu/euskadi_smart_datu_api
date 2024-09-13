1. Endpoints

   Algoritmo de estaciones más próximas

2. Algoritmo de calculo de current_air_quality_summary utilizando:

    Datos de las ultimas X, defecto 6, horas
    Excluidos registros sin datos. Si los registro excluidos superan umbral de porcentaje de datos toales, valor parametrizable, se devuelve No disponible.
    POnderadas por pesos de mayor peso la hora más reciente al menor peso de la hora más antigua considerada
    Valor resultante es la media ponderada de los valores presentes.

3. Preparacion de los datos

   Fichero combinado para recoger el historico de todas las estaciones desde 1 enero de 2020
   POsteriormente generacion de los X.npy e Y.npy para cada estación para servir de input a LTSM

4. Creacion del MOdelo - Redes Neuronales LTSM

   Siguientes pasos para el modelo LSTM utilizando tensorflow:
    Cargar los datos preprocesados: Cada archivo .npy contiene las secuencias de entrada (X) y sus correspondientes valores de salida (y) para una estación. Tendrás que cargar estos archivos en tu modelo LSTM para entrenar y evaluar el rendimiento.

    Diseñar el modelo LSTM: Crearás el modelo LSTM utilizando TensorFlow y Keras. El modelo debe ser capaz de predecir la calidad del aire utilizando las secuencias de tiempo generadas.

    Entrenar el modelo: Entrenarás el modelo con los datos de una o más estaciones de calidad del aire. También puedes elegir entrenar un modelo independiente para cada estación, o entrenar un solo modelo general utilizando los datos de todas las estaciones.

    Evaluación: Después de entrenar el modelo, necesitarás evaluar su precisión usando datos de validación o de prueba.

    Explicación del script de creación del modelo de redes neuronales LSTM:
    Carga de los datos: El script carga los datos preprocesados de todas las estaciones desde el directorio ../processed/lstm_data. Cada estación tiene sus propios archivos X_{station}.npy e y_{station}.npy.

    Combina los datos: Una vez cargados, se combinan los datos de todas las estaciones para entrenar un único modelo que puede generalizar a cualquier estación.

    Modelo LSTM: El modelo LSTM tiene dos capas LSTM con 50 unidades, seguidas de capas de Dropout para evitar el sobreajuste y una capa densa final para predecir el valor del ICA.

    Entrenamiento: El modelo se entrena con los datos combinados de todas las estaciones durante un número de épocas definido (en este caso, 10), con una parte de los datos separados para validación.

    Guardado del modelo: El modelo entrenado se guarda en ../model/air_quality_lstm_model.h5 para que puedas cargarlo más tarde en tu API.