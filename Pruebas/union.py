import pandas as pd

# Carga y unión de datasets
normalDt=pd.read_csv("normalDt/normalDt.csv")
normalDt['label'] = 0
print("Número de row de normal data")
print(len(normalDt))

anomalyDt=pd.read_csv("anomalyDt/anomalyDt.csv")
anomalyDt['label'] = 1
print("Número de row de metasploit")
print(len(anomalyDt.index))

dataset_total = pd.concat((normalDt, anomalyDt), ignore_index=True)
dataset_total.to_csv("dataset_completo.csv", encoding='utf-8', index=False)
