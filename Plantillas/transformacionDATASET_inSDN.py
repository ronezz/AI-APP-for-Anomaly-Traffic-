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
from sklearn.tree import DecisionTreeClassifier
import sys

#Establecimiento de la semilla
np_random_seed = 2
np.random.seed(2)

# Obtención de los datasets por separado
dataset_ovs=pd.read_csv(r"C:\Users\oscar\Desktop\TFM\visual\Tests\TFG\Dataset\InSDN_DatasetCSV\OVS.csv")
print("Número de row de ovs")
print(len(dataset_ovs.index))
dataset_metasploit=pd.read_csv(r"C:\Users\oscar\Desktop\TFM\visual\Tests\TFG\Dataset\InSDN_DatasetCSV\metasploitable-2.csv")
print("Número de row de metasploit")
print(len(dataset_metasploit.index))
dataset_normalData=pd.read_csv(r"C:\Users\oscar\Desktop\TFM\visual\Tests\TFG\Dataset\InSDN_DatasetCSV\Normal_data.csv")
print("Número de row de normal data")
print(len(dataset_normalData.index))
print("Datos cargados")

# Fusion de los 3 datasets en 1 solo
dataset = [dataset_metasploit, dataset_normalData, dataset_ovs]
dataset_total = pd.concat(dataset, ignore_index=True)
print("Número de row de fusionados")
print(len(dataset_total.index))

## Cambiamos la feature "Label" para darle un valor numerico. 
## 1 para el tráfico de ataque
## 0 para el tráfico normal
cleanup_nums = {"Label": {"Normal":0,"Probe":1,"DDoS":1,"DoS":1,"BFA":1,"Web-Attack":1,"BOTNET":1,"U2R":1,"DDoS ":1}}
dataset_total_filtrado = dataset_total.replace(cleanup_nums)

## Eliminacion de las features cuyo valor no variaba de una fila a otra y del identificador (valores de NaN en la matriz de correlacion) -> Revisado
dt_not_NaN= dataset_total_filtrado.drop(['Flow ID','Fwd PSH Flags','Fwd URG Flags','CWE Flag Count','ECE Flag Cnt','Fwd Byts/b Avg','Fwd Pkts/b Avg',
                                         'Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg','Init Fwd Win Byts','Fwd Seg Size Min'], axis=1)

#Creacion de variables en funcion de la subred de destino
dt_new = dt_not_NaN
dt_new['Dst 192.168.8']=np.where(dt_new["Dst IP"].str.startswith('192.168.8.'),1,0)     #ONOS y Mininet
dt_new['Dst 192.168.3.']=np.where(dt_new["Dst IP"].str.startswith('192.168.3.'),1,0)    #Metasploitable y Mininet
dt_new['Dst 200.175.2.']=np.where(dt_new["Dst IP"].str.startswith('200.175.2.'),1,0)    #Kali y Mininet
dt_new['Dst 192.168.20.']=np.where(dt_new["Dst IP"].str.startswith('192.168.20.'),1,0)  #Máquinas internas mininet
dt_new['Dst 172.17.0.']=np.where(dt_new["Dst IP"].str.startswith('172.17.0.'),1,0)      #DVWA

#Creacion de variables en funcion de la subred de fuente
dt_new['Src 192.168.3.']=np.where(dt_new["Src IP"].str.startswith('192.168.3.'),1,0)    #Metasploitable y Mininet
dt_new['Src 200.175.2.']=np.where(dt_new["Src IP"].str.startswith('200.175.2.'),1,0)    #Kali y Mininet
dt_new['Src 192.168.20.']=np.where(dt_new["Src IP"].str.startswith('192.168.20.'),1,0)  #Máquinas internas mininet
dt_new['Src 172.17.0.']=np.where(dt_new["Src IP"].str.startswith('172.17.0.'),1,0)      #DVWA

#Limpeza de variables que no se van a usar
dt_new= dt_new.drop(['Src IP','Dst IP','Timestamp'], axis=1)

#Limpieza de valores negativos en el campo de Flow Duration
negative_index = dt_new.index[dt_new['Flow Duration'] < 0]
dt_new = dt_new.drop(negative_index)

#Isolation_Forest
#Casos a eliminar
#1 Variable 1 -> Valores de 'Flow Duration' demasiado altos
#2 Variable 2 -> Valores de 'Flow IAT Std', 'Fwd IAT Std', 'Flow IAT Max' y 'Fwd IAT Tot' demasiado altos
#3 Variable 3 -> Valores de 'Bwd IAT Tot' muy altos
#4 Variable 4 -> Valores de 'Pkt Len Max' muy altos

dt_new_delete_outliers = dt_new

variable1=['Flow Duration', 'Tot Fwd Pkts','Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max','Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std','Bwd Pkt Len Max', 'Bwd Pkt Len Min']
variable2=['Bwd Pkt Len Mean','Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean','Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot','Fwd IAT Mean', 'Fwd IAT Std']
variable3=['Fwd IAT Max', 'Fwd IAT Min','Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max','Bwd IAT Min', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Header Len','Bwd Header Len']
variable4=['Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min','Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var','FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt']
variable5=['ACK Flag Cnt', 'URG Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg','Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts','Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts','Init Bwd Win Byts']
variable6=['Fwd Act Data Pkts', 'Active Mean', 'Active Std','Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max','Idle Min']

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=0.0001,max_features=8, random_state=1)

model.fit(dt_new_delete_outliers[variable1])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable1])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

model.fit(dt_new_delete_outliers[variable2])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable2])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

model.fit(dt_new_delete_outliers[variable3])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable3])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

model.fit(dt_new_delete_outliers[variable4])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable4])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

dt_new= dt_new.drop(['anomaly'], axis=1)

#FIN-Isolation_Forest

#Variables con correlación muy elevada
#Flow Duration -> Duración del flow en Microsegundos
#------Bwd IAT Tot - 0,98 -> Total del t. entre 2 paquetes enviados en el flow dir backward
#Se elimina esta por que tiene menos correlación con la label 0.01
dt_new= dt_new.drop(['Bwd IAT Tot'], axis=1)


#Tot Fwd Pkts -> Número de paquetes en la dirección forward
#------Subflow Fwd Pkts - 1.00 -> Número medio de paquetes en sub flow en d. forward
#Indiferente mismo valor
dt_new= dt_new.drop(['Subflow Fwd Pkts'], axis=1)


#TotLen Fwd Pkts -> Tamaño total de los paquetes en la dirección forward
#------Subflow Fwd Byts - 0.99 -> Número medio de Bytes en sub flow en d. forward
#Indiferente mismo valor
dt_new= dt_new.drop(['Subflow Fwd Byts'], axis=1)


#TotLen Bwd Pkts -> Tamaño total de los paquetes en la dirección backward
#------ Subflow Bwd Byts - 0.99  -> Número medio de Bytes en sub flow en d. backward
#Indiferente mismo valor
dt_new= dt_new.drop(['Subflow Bwd Byts'], axis=1)


#Tot Bwd Pkts -> Número de paquetes en la dirección backward
#------ Bwd Header Len - 0.99 -> Número de bytes usados para cabeceras en la dir backward
#------Subflow Bwd Pkts - 1.00 -> Número medio de paquetes en sub flow en d. backward
dt_new= dt_new.drop(['Bwd Header Len'], axis=1)
dt_new= dt_new.drop(['Subflow Bwd Pkts'], axis=1)


#Fwd Pkt Len Mean -> Tamaño medio de paquete en dir forward
#------ Fwd Pkt Len Std - 0.954 -> Desviación estándar del tamaño de paquete en dir forward
#------ Fwd Seg Size Avg - 1.00 -> Tamaño medio observado en la dirección de forward
dt_new= dt_new.drop(['Fwd Pkt Len Std'], axis=1)
dt_new= dt_new.drop(['Fwd Seg Size Avg'], axis=1)


# Bwd Pkt Len Max -> Tamaño máximo de paquete en dir backward
#------ Pkt Len Max  - 0.97 -> Máximo del tamaño de un paquete
dt_new= dt_new.drop(['Bwd Pkt Len Max'], axis=1)


# Bwd Pkt Len Mean -> Tamaño medio de paquete en dir backward
#------Bwd Seg Size Avg - 1.0000 -> Número medio de Bts bulk rate observado en la d. de backward
dt_new= dt_new.drop(['Bwd Seg Size Avg'], axis=1)


# Flow Pkts/s -> Flow paquetes por segundo
#------ Bwd Pkts/s - 0.99 -> Número de paquetes backward por segundo 
dt_new= dt_new.drop(['Flow Pkts/s'], axis=1)


#Flow IAT Std -> Desviación estándar del tiempo entre 2 paquetes enviados en el flow
#------ Flow IAT Max - 0.98 -> Tiempo máximo entre 2 paquetes enviados en el flow
#------ Bwd IAT Std - 0.98 -> Tiempo medio entre 2 paquetes enviados en el flow dir backward
#------ Bwd IAT Max - 0.97 -> Tiempo máximo entre 2 paquetes enviados en el flow dir backward
#------ Idle Mean - 0.98 -> Tiempo media que el flujo estuvo inactivo antes estar activo
#------ Idle Max - 0.98 -> Tiempo máximo que el flujo estuvo inactivo antes estar activo
#------ Idle Min - 0.98 ->Tiempo  mínimo que el flujo estuvo inactivo antes estar activo

dt_new= dt_new.drop(['Flow IAT Std'], axis=1)
dt_new= dt_new.drop(['Flow IAT Max'], axis=1)
dt_new= dt_new.drop(['Bwd IAT Max'], axis=1)
dt_new= dt_new.drop(['Idle Mean'], axis=1)
dt_new= dt_new.drop(['Idle Max'], axis=1)
dt_new= dt_new.drop(['Idle Min'], axis=1)




# Bwd PSH Flags -> Número de veces que el flag PSH usado en paquetes dir backward
#------ PSH Flag Cnt - 1 -> Número de paquetes con el flag PSH 
dt_new= dt_new.drop(['Bwd PSH Flags'], axis=1)


# Bwd URG Flags -> Número de veces que el flag URG usado en paquetes dir backward
#------ URG Flag Cnt - 1 -> Número de paquetes con el flag URG 
dt_new= dt_new.drop(['URG Flag Cnt'], axis=1)


# Pkt Len Mean -> Media del tamaño de un paquete
#------ Pkt Len Std - 0. 96 -> Desviación estándar del tamaño de un paquete
#------ Pkt Size Avg - 0.99 -> Tamaño medio del paquete
dt_new= dt_new.drop(['Pkt Len Mean'], axis=1)
dt_new= dt_new.drop(['Pkt Len Std'], axis=1)


#RFECV
target = dt_new['Label']
X = dt_new.drop('Label', axis=1)
rfc = DecisionTreeClassifier(random_state=0)

#Ejecución de RFECV
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=3), scoring='accuracy',verbose=4, min_features_to_select=1)
rfecv.fit(X, target)

#Obtener el número de características
print('Optimal number of features: {}'.format(rfecv.n_features_))
print("Características a no seleccionar")
print(np.where(rfecv.support_ == False)[0])
print("Características a seleccionar")
print(np.where(rfecv.support_ == True)[0])
X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
selected_columns = X.columns.tolist()
selected_columns.append('Label')
dt_new = dt_new[selected_columns]
#FIN-RFECV

#Se hace el reset de los indices para poder trabajar con ellos
dt_new.reset_index(inplace=True,drop=True)
#FIN-RFECV

print("Número de row final")
print(len(dt_new.index))

dt_new.to_csv(r"C:\Users\oscar\Desktop\TFM\visual\Tests\TFG\Dataset\InSDN_DatasetCSV\dataset_final_final.csv",encoding='utf-8', index= False)
print("Se guarda el dataset")


