import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys

# Registrar en el log
def log_message(message):
    with open('entrenamiento.log', 'a', encoding='utf-8') as log_file:
        log_file.write(message + "\n")
    print(message)

log_message(f"Iniciando el script a las {str(np.datetime64('now'))}")

# Directorio de datos preprocesados
data_dir = '../processed/lstm_data'
log_message(f"Directorio de datos: {data_dir}")

# Cargar el escalador
scaler_file = '../processed/scaler.pkl'
scaler = joblib.load(scaler_file)
log_message(f"Escalador cargado desde {scaler_file}")

# Función para cargar todos los archivos .npy de la carpeta de datos
def load_all_station_data(data_dir):
    all_X = []
    all_y = []
    all_dates = []

    station_count = 0
    for file_name in os.listdir(data_dir):
        if file_name.startswith('X_') and file_name.endswith('.npy'):
            station_id = file_name.split('X_')[1].split('.npy')[0]
            try:
                # Cargar datos
                X = np.load(os.path.join(data_dir, f'X_{station_id}.npy'))
                y = np.load(os.path.join(data_dir, f'y_{station_id}.npy'))
                dates = np.load(os.path.join(data_dir, f'dates_{station_id}.npy'))

                # Verificar si hay valores NaN
                nan_count_X = np.isnan(X).sum()
                nan_count_y = np.isnan(y).sum()
                if nan_count_X > 0 or nan_count_y > 0:
                    # Omitir muestras con NaN
                    valid_indices = ~np.isnan(X).any(axis=(1,2)) & ~np.isnan(y)
                    X = X[valid_indices]
                    y = y[valid_indices]
                    dates = dates[valid_indices]

                # Asegurarse de que las etiquetas están en el rango 0-5
                valid_labels = [0, 1, 2, 3, 4, 5]
                valid_indices = np.isin(y, valid_labels)
                X = X[valid_indices]
                y = y[valid_indices]
                dates = dates[valid_indices]

                log_message(f"Cargando datos para la estación {station_id}")
                log_message(f"Datos cargados para la estación {station_id}: X.shape={X.shape}, y.shape={y.shape}")
                log_message(f"NaNs en X: {nan_count_X}, NaNs en y: {nan_count_y}")
                log_message(f"Después de filtrar etiquetas inválidas: X.shape={X.shape}, y.shape={y.shape}")

                all_X.append(X)
                all_y.append(y)
                all_dates.append(dates)
                station_count += 1
            except FileNotFoundError as e:
                log_message(f"Archivos para la estación {station_id} no encontrados: {e}")

    log_message(f"Datos concatenados: {station_count} estaciones cargadas")
    if all_X and all_y:
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        dates_all = np.concatenate(all_dates, axis=0)
    else:
        X_all = np.array([])
        y_all = np.array([])
        dates_all = np.array([])

    return X_all, y_all, dates_all

# Cargar todos los datos de todas las estaciones
log_message("Cargando todos los datos de las estaciones...")
X_all, y_all, dates_all = load_all_station_data(data_dir)

# Convertir las etiquetas a enteros
y_all = y_all.astype(int)
log_message("Etiquetas convertidas a enteros.")

# Comprobar si aún existen valores NaN
log_message(f"NaN en X_all: {np.isnan(X_all).any()}")
log_message(f"NaN en y_all: {np.isnan(y_all).any()}")

# Comprobar el tamaño de los datos
log_message(f"Tamano de X_all: {X_all.shape}")
log_message(f"Tamano de y_all: {y_all.shape}")
log_message(f"Uso de memoria de X_all: {X_all.nbytes / (1024 ** 3):.2f} GB")
log_message(f"Uso de memoria de y_all: {y_all.nbytes / (1024 ** 3):.2f} GB")

# División de los datos en entrenamiento y prueba respetando la secuencia temporal
if X_all.size > 0 and y_all.size > 0:
    log_message("Dividiendo los datos en entrenamiento y prueba...")
    # Convertir fechas a números para poder ordenar
    dates_all_num = np.array([date.astype('datetime64[s]').astype('int') for date in dates_all])

    # Ordenar los datos por fecha
    sorted_indices = np.argsort(dates_all_num)
    X_all = X_all[sorted_indices]
    y_all = y_all[sorted_indices]
    dates_all = dates_all[sorted_indices]

    # Dividir los datos
    train_size = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]
    log_message(f"Datos divididos: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

    # Verificar que X_train y X_test tengan tres dimensiones
    if len(X_train.shape) < 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_test.shape) < 3:
        X_test = np.expand_dims(X_test, axis=-1)

    # Crear el modelo LSTM para clasificación
    log_message("Creando el modelo LSTM...")
    num_classes = 6  # Clases de 0 a 5

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_message("Modelo compilado.")

    # Convertir las etiquetas a enteros por si acaso
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Calcular los pesos de las clases
    classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights_dict = dict(zip(classes, class_weights))
    log_message(f"Pesos de las clases calculados: {class_weights_dict}")

    # Ajustar manualmente los pesos de las clases 1, 2 y 3
    class_weights_dict[1] *= 10
    class_weights_dict[2] *= 7
    class_weights_dict[3] *= 5

    log_message(f"Pesos de las clases ajustados manualmente: {class_weights_dict}")

    # Crear datasets de TensorFlow para manejar grandes volúmenes de datos
    log_message("Creando datasets de TensorFlow...")
    batch_size = 128  # Aumentado para acelerar el entrenamiento si la memoria lo permite
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    log_message("Datasets creados.")

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('../model/lstm_air_quality_best_model.keras', save_best_only=True, monitor='val_loss')
    log_message("Callbacks configurados.")

    try:
        # Entrenar el modelo
        log_message("Iniciando el entrenamiento del modelo...")
        history = model.fit(
            train_dataset,
            epochs=50,
            validation_data=val_dataset,
            class_weight=class_weights_dict,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        # Guardar el modelo entrenado final
        model_dir = '../model'
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, 'lstm_air_quality_model.keras')
        model.save(model_file)
        log_message(f"Entrenamiento completado y modelo guardado en {model_file}")

        # Evaluación del modelo
        y_pred_probs = model.predict(val_dataset)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_eval = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)

        # Reporte de clasificación
        log_message("\nReporte de clasificación:")
        log_message(classification_report(y_test_eval, y_pred, digits=4))

        # Matriz de confusión
        cm = confusion_matrix(y_test_eval, y_pred)
        log_message("Matriz de confusión:")
        log_message(str(cm))

        # Registrar métricas de entrenamiento
        epochs_trained = len(history.history['loss'])
        log_message(f"El modelo fue entrenado durante {epochs_trained} épocas.")

        final_train_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        log_message(f"Precisión final en entrenamiento: {final_train_accuracy:.4f}")
        log_message(f"Precisión final en validación: {final_val_accuracy:.4f}")
        log_message(f"Pérdida final en entrenamiento: {final_train_loss:.4f}")
        log_message(f"Pérdida final en validación: {final_val_loss:.4f}")

    except Exception as e:
        log_message(f"Error durante el entrenamiento: {str(e)}")
        sys.exit(1)

else:
    log_message("No hay datos suficientes para entrenar el modelo.")
