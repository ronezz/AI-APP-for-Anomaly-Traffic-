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

# Limpieza de atributos que no varian
dt_not_nan = dt.drop(['protocol','fwd_pkt_len_max', 'fwd_pkt_len_min', 'bwd_pkt_len_max', 'bwd_pkt_len_min',
                      'pkt_len_max', 'pkt_len_min', 'fwd_seg_size_min', 'fwd_urg_flags', 'bwd_urg_flags', 
                      'rst_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'active_max', 'active_mean', 
                      'active_std', 'cwr_flag_count'], axis=1)

# Limpieza de columnas irrelevantes
dt_new = dt_not_nan.drop(['src_ip', 'dst_ip', 'timestamp'], axis=1)

# Limpieza de valores negativos en flow_duration
dt_new = dt_new[dt_new['flow_duration'] >= 0]

dt_new.to_csv("dtcm.csv", encoding='utf-8', index=False)

# Agrupación de variables relacionadas
variable1 = ['flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts',
              'fwd_pkt_len_mean', 'fwd_pkt_len_std']
variable2 = ['bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean',
             'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std']
variable3 = ['fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
             'bwd_iat_min', 'fwd_header_len', 'bwd_header_len']
variable4 = ['fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
             'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt']

# Isolation Forest
model = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.01, max_features=8, random_state=1)

dt_new_delete_anomalies = dt_new

for variable in [variable1, variable2, variable3, variable4]:
    model.fit(dt_new_delete_anomalies[variable])
    dt_new_delete_anomalies['anomaly'] = model.predict(dt_new_delete_anomalies[variable])
    dt_new_delete_anomalies = dt_new_delete_anomalies[dt_new_delete_anomalies['anomaly'] == 1]
    dt_new_delete_anomalies= dt_new_delete_anomalies.drop(['anomaly'], axis=1) 


correlation_matrix = dt_new_delete_anomalies.corr()

threshold = 0.9
high_corr_pairs=set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            high_corr_pairs.add((var1, var2))

variables_to_remove = set()
for var1, var2 in high_corr_pairs:
    variables_to_remove.add(var2)
print (variables_to_remove)

# Eliminar variables altamente correlacionadas
dt_new = dt_new.drop(['bwd_iat_tot', 'subflow_fwd_pkts', 'subflow_fwd_byts', 
                      'subflow_bwd_byts', 'bwd_header_len', 'subflow_bwd_pkts', 
                      'fwd_pkt_len_std', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 
                      'pkt_len_std', 'pkt_size_avg', 'flow_pkts_s', 
                      'flow_iat_std', 'flow_iat_max', 'idle_mean', 'idle_max'], axis=1)


target = dt_new['label']  # Asegúrate de que la columna de etiquetas se llame "label"
X = dt_new.drop('label', axis=1)

# RFECV
rfecv = RFECV(estimator=RandomForestClassifier(random_state=0), step=1, 
              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3), 
              scoring='accuracy', verbose=4, min_features_to_select=1)

rfecv.fit(X, target)


selected_columns = X.columns[rfecv.support_].tolist()
selected_columns.append('label')
dt_new = dt_new[selected_columns]

dt_new.reset_index(drop=True, inplace=True)
dt_new.to_csv("dataset_final.csv", encoding='utf-8', index=False)
print("Dataset final guardado con éxito.")
