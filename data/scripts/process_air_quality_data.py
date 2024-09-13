import pandas as pd
import os

def load_and_combine_csv(folder_path):
    all_files = []
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                station_name = os.path.splitext(os.path.basename(file))[0]  # Nombre del archivo sin la extensión
                
                df = pd.read_csv(file_path, delimiter=';', decimal=',', encoding='ISO-8859-1', dayfirst=True)
                
                # Reemplazar la hora 24:00 con 00:00
                df['Hour  (GMT)'] = df['Hour  (GMT)'].replace('24:00', '00:00')

                # Combinar las columnas de fecha y hora en una columna de datetime
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Hour  (GMT)'], format='%d/%m/%Y %H:%M', dayfirst=True)
                
                # Eliminar las columnas originales de fecha y hora
                df.drop(columns=['Date', 'Hour  (GMT)'], inplace=True)
                
                # Añadir la columna 'StationName'
                df['StationName'] = station_name
                
                # Reordenar las columnas para que 'DateTime' sea la primera y 'StationName' sea la segunda
                columns = ['DateTime', 'StationName'] + [col for col in df.columns if col not in ['DateTime', 'StationName']]
                df = df[columns]
                
                all_files.append(df)
    
    # Combinar todos los archivos
    combined_df = pd.concat(all_files, ignore_index=True)
    
    # Ordenar por la columna 'DateTime'
    combined_df = combined_df.sort_values(by='DateTime').reset_index(drop=True)
    
    return combined_df

# Ruta al directorio donde están almacenados los CSV
folder_path = '../raw'
combined_df = load_and_combine_csv(folder_path)

# Guardar el DataFrame combinado en un archivo CSV
output_file = '../processed/combined_air_quality_data.csv'
combined_df.to_csv(output_file, index=False)

print(f"Data combinada y ordenada guardada en: {output_file}")
