import numpy as np
import os

data_dir = '../processed/lstm_data'
y_all = []

for file_name in os.listdir(data_dir):
    if file_name.startswith('y_') and file_name.endswith('.npy'):
        y = np.load(os.path.join(data_dir, file_name))
        y_all.extend(y)

y_all = np.array(y_all)
unique, counts = np.unique(y_all, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Distribución de clases:", class_distribution)
