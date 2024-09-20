# process_air_quality_data.py

import pandas as pd
import os
import chardet

def load_and_combine_csv(folder_path):
    all_files = []
    file_info = []
    file_count = 0  # Contador de archivos procesados
    for subdir, dirs, files in os.walk(folder_path):
        print(f"Buscando en el directorio: {subdir}")
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                station_name = os.path.splitext(os.path.basename(file))[0]  # Nombre del archivo sin la extensión
                file_count += 1
                print(f"\nProcesando archivo ({file_count}): {file_path}")
                print(f"Nombre de la estación: {station_name}")

                try:
                    # Detectar la codificación del archivo
                    with open(file_path, 'rb') as f:
                        result = chardet.detect(f.read(100000))  # Leer solo los primeros 100KB
                    encoding = result['encoding'] if result['encoding'] else 'ISO-8859-1'
                    print(f"Codificación detectada: {encoding}")

                    # Intentar leer el archivo con delimitador ';'
                    try:
                        df = pd.read_csv(
                            file_path,
                            delimiter=';',
                            decimal=',',
                            encoding=encoding,
                            dayfirst=True,
                            low_memory=False
                        )
                    except Exception as e:
                        print(f"Error al leer {file_path} con delimitador ';': {e}")
                        # Intentar leer con delimitador ','
                        df = pd.read_csv(
                            file_path,
                            delimiter=',',
                            decimal='.',
                            encoding=encoding,
                            dayfirst=True,
                            low_memory=False
                        )
                        print(f"Archivo {file_path} leído con delimitador ',' y decimal '.'")

                    # Verificar y ajustar las columnas
                    expected_columns = ['Date', 'Hour  (GMT)']
                    missing_columns = [col for col in expected_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Advertencia: Las siguientes columnas faltan en {file_path}: {missing_columns}")
                        # Intentar renombrar columnas similares
                        for col in missing_columns:
                            possible_matches = [c for c in df.columns if col.lower() in c.lower()]
                            if possible_matches:
                                df.rename(columns={possible_matches[0]: col}, inplace=True)
                                print(f"Columna '{possible_matches[0]}' renombrada a '{col}' en {file_path}")
                            else:
                                print(f"No se encontró una columna similar a '{col}' en {file_path}. Saltando este archivo.")
                                raise Exception(f"Columnas clave faltantes en {file_path}")

                    # Eliminar columnas sin nombre o con 'Unnamed'
                    df.drop(columns=[col for col in df.columns if 'Unnamed' in str(col) or pd.isna(col)], inplace=True)

                    # Reemplazar la hora 24:00 con 00:00
                    df['Hour  (GMT)'] = df['Hour  (GMT)'].replace('24:00', '00:00')

                    # Combinar las columnas de fecha y hora en una columna de datetime
                    df['DateTime'] = pd.to_datetime(
                        df['Date'] + ' ' + df['Hour  (GMT)'],
                        format='%d/%m/%Y %H:%M',
                        dayfirst=True,
                        errors='coerce'  # Para manejar fechas inválidas
                    )

                    # Eliminar las filas con DateTime NaT (errores en la conversión)
                    df.dropna(subset=['DateTime'], inplace=True)

                    # Eliminar las columnas originales de fecha y hora
                    df.drop(columns=['Date', 'Hour  (GMT)'], inplace=True)

                    # Añadir la columna 'StationName'
                    df['StationName'] = station_name

                    # Reordenar las columnas para que 'DateTime' sea la primera y 'StationName' sea la segunda
                    columns = ['DateTime', 'StationName'] + [col for col in df.columns if col not in ['DateTime', 'StationName']]
                    df = df[columns]

                    all_files.append(df)
                    file_info.append({
                        'station_name': station_name,
                        'file_path': file_path,
                        'columns': df.columns.tolist()
                    })
                except Exception as e:
                    print(f"Error al procesar el archivo {file_path}: {e}")

    print(f"\nNúmero total de archivos CSV encontrados: {file_count}")
    print(f"Número total de DataFrames combinados: {len(all_files)}")

    if len(all_files) == 0:
        print("No se encontraron archivos CSV válidos para combinar.")
        return pd.DataFrame(), []

    # Combinar todos los archivos
    combined_df = pd.concat(all_files, ignore_index=True)

    # Ordenar por la columna 'DateTime'
    combined_df = combined_df.sort_values(by='DateTime').reset_index(drop=True)

    return combined_df, file_info

def main():
    # Ruta al directorio donde están almacenados los CSV
    folder_path = '../raw'

    # Llamar a la función para cargar y combinar los CSV
    combined_df, file_info = load_and_combine_csv(folder_path)

    # Continuar con el preprocesamiento si el DataFrame no está vacío
    if not combined_df.empty:
        # Mostrar todas las columnas del DataFrame
        print("\nColumnas del DataFrame combinado:")
        print(combined_df.columns.tolist())

        # Verificar si hay columnas con nombres 'NaN' o 'Unnamed'
        nan_columns = [col for col in combined_df.columns if pd.isna(col) or 'Unnamed' in str(col)]
        if nan_columns:
            print("Columnas con nombres 'NaN' o 'Unnamed' encontradas:", nan_columns)
        else:
            print("No se encontraron columnas con nombres 'NaN' o 'Unnamed'.")

        # Verificar el número total de estaciones
        num_estaciones = combined_df['StationName'].nunique()
        print(f"\nNúmero total de estaciones en el DataFrame: {num_estaciones}")
        print("Lista de estaciones:")
        print(combined_df['StationName'].unique())

        # Mostrar las primeras filas para verificar los datos
        print("\nPrimeras filas del DataFrame combinado:")
        print(combined_df.head())

        # Resumen de valores faltantes
        missing_values = combined_df.isnull().sum()
        print("\nResumen de valores faltantes por columna:\n", missing_values)

        # Mapear las columnas categóricas
        category_columns = ['ICA Estación', 'NO2 - ICA', 'PM10 - ICA', 'PM2,5 - ICA', 'SO2 - ICA', 'O3 - ICA']

        # Mapear las categorías a valores numéricos (normalizando las cadenas)
        category_mapping = {
            "muy bueno / oso ona": 5,
            "bueno / ona": 4,
            "regular / erregularra": 3,
            "malo / txarra": 2,
            "muy malo / oso txarra": 1,
            "sin datos / daturik gabe": 0
        }

        # Normalizar las claves del mapeo
        category_mapping_normalized = {k.lower(): v for k, v in category_mapping.items()}

        # Normalizar y mapear las columnas categóricas
        for col in category_columns:
            if col in combined_df.columns:
                # Convertir a string y normalizar
                combined_df[col] = combined_df[col].astype(str).str.strip().str.lower()
                # Aplicar el mapeo
                combined_df[col] = combined_df[col].map(category_mapping_normalized)

        # Verificar y manejar valores no mapeados
        for col in category_columns:
            if col in combined_df.columns:
                unmapped_values = combined_df[combined_df[col].isnull()][col].unique()
                if len(unmapped_values) > 0:
                    print(f"Valores no mapeados en '{col}': {unmapped_values}")
                    # Asignar 0 (sin datos) a los valores no mapeados
                    # En lugar de usar inplace=True, asignamos el resultado a la columna
                    combined_df[col] = combined_df[col].fillna(0)


        # Manejar los valores faltantes en 'ICA Estación' (variable objetivo)
        missing_target = combined_df['ICA Estación'].isnull().sum()
        print(f"\nValores faltantes en 'ICA Estación': {missing_target}")
        # Eliminar filas con valores faltantes en 'ICA Estación'
        combined_df.dropna(subset=['ICA Estación'], inplace=True)

        # Identificar columnas numéricas para interpolación
        numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in category_columns]

        # Configurar 'DateTime' como índice para la interpolación temporal
        combined_df.set_index('DateTime', inplace=True)

        # Interpolar valores faltantes en columnas numéricas utilizando el tiempo
        combined_df[numeric_columns] = combined_df[numeric_columns].interpolate(method='time')

        # Rellenar valores faltantes restantes en columnas numéricas
        combined_df[numeric_columns] = combined_df[numeric_columns].ffill().bfill()

        # Restablecer el índice para que 'DateTime' sea una columna nuevamente
        combined_df.reset_index(inplace=True)

        # Verificar si aún quedan valores faltantes
        remaining_missing = combined_df.isnull().sum()
        print("\nValores faltantes después de la interpolación:")
        print(remaining_missing)

        # Opcional: Eliminar filas que aún contengan valores faltantes (si las hay)
        # combined_df.dropna(inplace=True)

        # Guardar el DataFrame combinado y preprocesado
        output_file = '../processed/combined_air_quality_data_filled.csv'
        combined_df.to_csv(output_file, index=False)

        print(f"\nDatos combinados y preprocesados guardados en: {output_file}")
    else:
        print("El DataFrame combinado está vacío. No se puede continuar con el preprocesamiento.")

if __name__ == "__main__":
    main()
