import pandas as pd

def remove_high_zero_columns(df, threshold=0.9):
    zero_percentage = (df == 0).mean()  # Calcula el porcentaje de ceros por columna
    columns_to_keep = zero_percentage[zero_percentage < threshold].index  # Filtra las columnas a conservar
    return df[columns_to_keep]

# Cargar dataset (ejemplo)
df = pd.read_csv("normalDt/normalDt.csv")  
df2 = pd.read_csv("anomalyDt/anomalyDt.csv") 
# Aplicar la función con un umbral del 90%
df_filtered = remove_high_zero_columns(df, threshold=0.9)
df2_filtered = remove_high_zero_columns(df2, threshold=0.9)

# Guardar el dataset filtrado
#df_filtered.to_csv("dataset_filtrado.csv", index=False)

print("Columnas eliminadas del dataset normal:", set(df.columns) - set(df_filtered.columns))
print("Columnas eliminadas del dataset anómalo:", set(df2.columns) - set(df2_filtered.columns))

