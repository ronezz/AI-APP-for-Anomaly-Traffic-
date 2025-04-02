import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import sys

# Establecimiento de semilla
np_random_seed = 2
np.random.seed(2)


# Carga y unión de datasets
normalDt=pd.read_csv("normalDt/normalDt.csv")
normalDt['label'] = 0
print("Número de row de normal data")
print(len(normalDt))

anomalyDt=pd.read_csv("anomalyDt/anomalyDt.csv")
anomalyDt['label'] = 1
print("Número de row de metasploit")
print(len(anomalyDt.index))
print("Datos cargados")

# Concatenación de datasets
dt = pd.concat([normalDt, anomalyDt], axis=0, ignore_index= True)
print("Número de row de fusionados")
print(len(dt))

dt.to_csv(r"dtcm.csv",encoding='utf-8', index= False)

# Variables que no varían

dt_notNan = dt.drop(['bwd_pkt_len_max', 'bwd_pkt_len_min', 'pkt_len_max', 'pkt_len_min', 'fwd_seg_size_min',
             'fwd_urg_flags', 'bwd_urg_flags', 'rst_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt',
             'active_max','active_min', 'active_std', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg',
              'fwd_blk_rate_avg', 'bwd_blk_rate_avg' , 'cwr_flag_count'], axis=1)


# Limpieza de columnas irrelevantes
dt_new = dt_notNan.drop(['src_ip', 'dst_ip', 'timestamp'], axis=1)

# Limpieza de valores negativos en flow_duration
dt_new = dt_new[dt_new['flow_duration'] >= 0]


# Isolation_Forest
variable1 = ['flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 
             'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std']
variable2 = ['bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean',
             'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std']
variable3 = ['fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
             'bwd_iat_min', 'fwd_header_len', 'bwd_header_len']
variable4 = ['fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
             'fin_flag_cnt', 'syn_flag_cnt', 'psh_flag_cnt']


model = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.01, max_features=8, random_state=1)

for group in [variable1, variable2, variable3, variable4]:
    model.fit(dt_new[group])
    dt_new['anomaly'] = model.predict(dt_new[group])
    dt_new = dt_new[dt_new['anomaly'] == 1]
    dt_new = dt_new.drop(['anomaly'], axis=1) 



# Atributos redundantes -> Matriz de Correlación

# Flow Duration -> Duración total del flujo
# ------ Fwd IAT Tot - 0.99 -> Tiempo total entre paquetes enviados en dirección forward
dt_new = dt_new.drop(['fwd_iat_tot'], axis=1)

# Flow Pkts/s -> Número total de paquetes por segundo en el flujo
# ------ Fwd Pkts/s - 0.97 -> Paquetes por segundo enviados en dirección forward
dt_new = dt_new.drop(['fwd_pkts_s'], axis=1)

# Tot Fwd Pkts -> Número total de paquetes enviados en dirección forward
# ------ Subflow Fwd Pkts - 1.00 -> Número medio de paquetes enviados en subflow dirección forward
dt_new = dt_new.drop(['subflow_fwd_pkts'], axis=1)

# Tot Bwd Pkts -> Número total de paquetes enviados en dirección backward
# ------ Subflow Bwd Pkts - 1.00 -> Número medio de paquetes enviados en subflow dirección backward
dt_new = dt_new.drop(['subflow_bwd_pkts'], axis=1)

# TotLen Fwd Pkts -> Tamaño total de los paquetes en dirección forward
# ------ Subflow Fwd Byts - 1.00 -> Tamaño medio de bytes en subflow dirección forward
dt_new = dt_new.drop(['subflow_fwd_byts'], axis=1)

# TotLen Bwd Pkts -> Tamaño total de los paquetes en dirección backward
# ------ Subflow Bwd Byts - 1.00 -> Tamaño medio de bytes en subflow dirección backward
dt_new = dt_new.drop(['subflow_bwd_byts'], axis=1)

# Fwd Pkt Len Mean -> Tamaño medio de paquete en dirección forward
# ------ Fwd Seg Size Avg - 1.00 -> Tamaño medio observado en dirección forward
dt_new = dt_new.drop(['fwd_seg_size_avg'], axis=1)

# Bwd Pkt Len Mean -> Tamaño medio de paquete en dirección backward
# ------ Bwd Seg Size Avg - 1.00 -> Tamaño medio observado en dirección backward
dt_new = dt_new.drop(['bwd_seg_size_avg'], axis=1)

# Pkt Len Mean -> Tamaño medio del paquete
# ------ Pkt Size Avg - 1.00 -> Tamaño promedio del paquete
dt_new = dt_new.drop(['pkt_size_avg'], axis=1)

# Ack Flag Cnt -> Número de paquetes con flag ACK
# ------ Tot Fwd Pkts - 0.98 -> Número total de paquetes enviados en dirección forward
dt_new = dt_new.drop(['ack_flag_cnt'], axis=1)

# Idle Max -> Tiempo máximo que el flujo estuvo inactivo
# ------ Idle Mean - 0.98 -> Tiempo promedio que el flujo estuvo inactivo
dt_new = dt_new.drop(['idle_mean'], axis=1)

# Active Max -> Tiempo máximo de actividad en el flujo
# ------ Active Mean - 0.98 -> Tiempo promedio de actividad en el flujo
dt_new = dt_new.drop(['active_mean'], axis=1)




target = dt_new['label']  
X = dt_new.drop('label', axis=1)

# Crear el modelo y RFECV
rfecv = RFECV(estimator=RandomForestClassifier(random_state=0), step=1, 
              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3), 
              scoring='accuracy', verbose=4, min_features_to_select=1)

rfecv.fit(X, target)

selected_columns = X.columns[rfecv.support_].tolist()
selected_columns.append('label') 
print(selected_columns)
dt_new = dt_new[selected_columns]

dt_new.reset_index(inplace=True,drop=True)
#FIN-RFECV

print("Número de row final")
print(len(dt_new.index))

dt_new.to_csv(r"resultado.csv",encoding='utf-8', index= False)
print("Se guarda el dataset")