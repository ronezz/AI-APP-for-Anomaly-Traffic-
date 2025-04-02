import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from Anexos.remove_columns import remove_columns

# Establecimiento de semilla
np_random_seed = 2
np.random.seed(2)

# Carga y filtrado de datasets
normalDt = pd.read_csv("normalDt/normalDt.csv")
columns_to_remove_normal = remove_columns(normalDt, 0.9)
normalDt['label'] = 0
print("Número de row de normal data:", len(normalDt))

anomalyDt = pd.read_csv("anomalyDt/anomalyDt-fus-rec-tcpdump.pcap_Flow.csv")
columns_to_remove_anomaly = remove_columns(anomalyDt, 0.9)
anomalyDt['label'] = 1
print("Número de row de metasploit:", len(anomalyDt))

columns_to_remove = columns_to_remove_normal | columns_to_remove_anomaly
print(columns_to_remove)
print("Número de columnas a eliminar:", len(columns_to_remove))

# Concatenación de datasets
dt = pd.concat([normalDt, anomalyDt], axis=0, ignore_index=True)
dt_new = dt.drop(columns_to_remove, axis=1)
print("Número de filas fusionadas:", len(dt))

# Limpieza de columnas irrelevantes
dt_new = dt_new.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1)

dt_new = dt_new[dt_new['Flow Duration'] >= 0]


dt_new.to_csv(r"dt_pre_matrix_red.csv", encoding='utf-8', index=False)

print("Atributos a entrenar: ", set(dt_new.columns))



variable1 = ['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 
            'Flow IAT Min', 'Flow IAT Max', 'Flow IAT Std','TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
            'Tot Fwd Pkts', 'Tot Bwd Pkts']

variable2 = ['Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Fwd Pkt Len Std', 
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd Pkt Len Max', 
            'Pkt Len Std', 'Pkt Len Mean', 'Pkt Len Max', 'Pkt Size Avg', 
            'Pkt Len Var']

variable3 = ['SYN Flag Cnt', 'FIN Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 
            'Protocol', 'Dst Port', 'Src Port', 'Down/Up Ratio']

variable4 = ['Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Seg Size Avg', 
            'Fwd Seg Size Min', 'Bwd Seg Size Avg', 
            'Fwd IAT Mean', 'Fwd IAT Min', 'Fwd IAT Max', 'Fwd IAT Std', 
            'Fwd Act Data Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Byts']


# Isolation Forest
model = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.01, max_features=8, random_state=1)

# Aplicar Isolation Forest por grupos
for group in [variable1, variable2, variable3, variable4]:
    model.fit(dt_new[group])
    dt_new['anomaly'] = model.predict(dt_new[group])
    dt_new = dt_new[dt_new['anomaly'] == 1]  
    dt_new = dt_new.drop(['anomaly'], axis=1) 


'''
# Matriz de Correlación

# Flow Duration -> Duración total del flujo
# ------ Flow Duration - 0.99 -> Redundante con Bwd IAT Tot, que representa el tiempo total entre paquetes de respuesta
dt_new = dt_new.drop(['Flow Duration'], axis=1)

# Fwd Header Len -> Longitud total de los encabezados de los paquetes de reenvío
# ------ Fwd Header Len - 0.99 -> Redundante con Tot Fwd Pkts, que ya contiene la cantidad total de paquetes reenviados
dt_new = dt_new.drop(['Fwd Header Len'], axis=1)

# Bwd Header Len -> Longitud total de los encabezados de los paquetes de respuesta
# ------ Bwd Header Len - 0.98 -> Redundante con Tot Bwd Pkts, que ya mide el total de paquetes de respuesta
dt_new = dt_new.drop(['Bwd Header Len'], axis=1)

# Fwd Act Data Pkts -> Número de paquetes de datos activos enviados en dirección forward
# ------ Fwd Act Data Pkts - 0.99 -> Redundante con TotLen Fwd Pkts, que representa el tamaño total de los paquetes reenviados
dt_new = dt_new.drop(['Fwd Act Data Pkts'], axis=1)

# Fwd Seg Size Avg -> Tamaño promedio de los segmentos reenviados
# ------ Fwd Seg Size Avg - 1.00 -> Redundante con Fwd Pkt Len Mean, que ya mide el tamaño promedio de los paquetes reenviados
dt_new = dt_new.drop(['Fwd Seg Size Avg'], axis=1)

# Bwd Seg Size Avg -> Tamaño promedio de los segmentos de respuesta
# ------ Bwd Seg Size Avg - 1.00 -> Redundante con Bwd Pkt Len Mean, que mide el tamaño promedio de los paquetes de respuesta
dt_new = dt_new.drop(['Bwd Seg Size Avg'], axis=1)

# Flow Pkts/s -> Número de paquetes por segundo en el flujo
# ------ Flow Pkts/s - 0.99 -> Redundante con Fwd Pkts/s y Bwd Pkts/s, que separan la velocidad de envío y recepción
dt_new = dt_new.drop(['Flow Pkts/s'], axis=1)

# Flow IAT Max -> Mayor tiempo entre paquetes en el flujo
# ------ Flow IAT Max - 0.99 -> Redundante con Fwd IAT Max y Bwd IAT Max, que desglosan los tiempos de inactividad por dirección
dt_new = dt_new.drop(['Flow IAT Max'], axis=1)

# Fwd IAT Tot -> Tiempo total entre paquetes en dirección forward
# ------ Fwd IAT Tot - 0.99 -> Redundante con Fwd IAT Max y Bwd IAT Tot, que ofrecen una mejor distribución del tiempo
dt_new = dt_new.drop(['Fwd IAT Tot'], axis=1)

# Idle Mean -> Tiempo medio de inactividad en la conexión
# ------ Idle Mean - 0.99 -> Redundante con Idle Max e Idle Min, que ofrecen valores más representativos de la inactividad
dt_new = dt_new.drop(['Idle Mean'], axis=1)

# Pkt Size Avg -> Tamaño medio de los paquetes
# ------ Pkt Size Avg - 0.99 -> Redundante con Pkt Len Mean, que ya mide el tamaño promedio de los paquetes de manera estándar
dt_new = dt_new.drop(['Pkt Size Avg'], axis=1)
'''
# Reemplazar inf y -inf por NaN y eliminar filas con NaN
dt_post_rf = dt_new.replace([np.inf, -np.inf], np.nan)

dt_post_rf = dt_post_rf.dropna()  # Eliminar filas con NaN



target = dt_post_rf['label']  
X = dt_post_rf.drop('label', axis=1)

#RFECV
rfecv = RFECV(estimator=RandomForestClassifier(random_state=0), step=1, 
              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3), 
              scoring='accuracy', verbose=4, min_features_to_select=1)

rfecv.fit(X, target)

selected_columns = X.columns[rfecv.support_].tolist()
selected_columns.append('label') 
print("Columnas seleccionadas por RFECV:", selected_columns)

dt_post_matrix = dt_post_rf[selected_columns]
print("Dataset final tras RFECV:")

dt_post_matrix.reset_index(inplace=True, drop=True)

dt_post_matrix.to_csv(r"resultado-reducido.csv", encoding='utf-8', index=False)
print("Se guarda el dataset final")
