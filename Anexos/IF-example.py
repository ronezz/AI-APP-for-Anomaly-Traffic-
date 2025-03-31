import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un DataFrame de ejemplo
np.random.seed(42)
data = {
    'tot_fwd_pkts': np.random.randint(10, 100, 1000).tolist() + [1000, 1200, 1500],  # Añadimos outliers
    'totlen_fwd_pkts': np.random.randint(100, 1000, 1000).tolist() + [5000, 8000, 10000],
    'fwd_pkt_len_max': np.random.randint(50, 500, 1000).tolist() + [2000, 2500, 3000],
    'fwd_iat_tot': np.random.randint(1, 100, 1000).tolist() + [500, 700, 900]
}

df = pd.DataFrame(data)

# Visualizamos las primeras filas
print("Datos originales:")
print(df.head())

# Paso 1: Calcular matriz de correlación
print("\nMatriz de correlación:")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualizar la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

# Paso 2: Aplicar Isolation Forest
isolation_forest = IsolationForest(contamination=0.02, random_state=42)

# Entrenamos el modelo
isolation_forest.fit(df)

# Generamos las predicciones (-1: anomalía, 1: normal)
df['anomaly'] = isolation_forest.predict(df)
df['anomaly_score'] = isolation_forest.decision_function(df)

# Filtrar las anomalías
outliers = df[df['anomaly'] == -1]
cleaned_data = df[df['anomaly'] == 1]

print("\nDatos considerados como anomalías:")
print(outliers)

print("\nDatos después de eliminar anomalías:")
print(cleaned_data.head())

# Visualización antes y después de eliminar anomalías
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='tot_fwd_pkts', y='totlen_fwd_pkts', hue='anomaly', palette={1: 'blue', -1: 'red'})
plt.title("Datos Originales con Anomalías")
plt.subplot(1, 2, 2)
sns.scatterplot(data=cleaned_data, x='tot_fwd_pkts', y='totlen_fwd_pkts', color='blue')
plt.title("Datos Limpiados (Sin Anomalías)")
plt.show()
