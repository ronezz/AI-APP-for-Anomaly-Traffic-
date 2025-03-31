import pandas as pd
import numpy as np
import seaborn as sns    
import matplotlib.pyplot as plt

dt = pd.read_csv('dtcm.csv')
dt_notlabel = dt.drop(['label'], axis=1)
print(dt_notlabel)

dt_new = dt_notlabel.drop(['src_ip', 'dst_ip', 'timestamp'], axis=1)

# Creamos la matriz de correlación
correlation_matrix = dt_new.corr()

high_corr_pairs = (
    correlation_matrix
    .where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) 
    .stack()  
    .reset_index()  
)
high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']

high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'] >= 0.95]

print("Pares de atributos con una correlación >= 95%:")
print(high_corr_pairs)