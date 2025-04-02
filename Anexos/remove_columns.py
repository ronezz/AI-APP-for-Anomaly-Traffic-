def remove_columns(df, threshold=0.9):
    
    # Eliminar columnas de valores nulos
    zero_percentage = (df == 0).mean()
    df_not_nan= df[zero_percentage[zero_percentage < threshold].index]
    
    # Eliminar columnas de valores únicos
    df_filtered = df_not_nan.loc[:, df.nunique() > 1]

    print("Columnas eliminadas: ", set(df.columns) - set(df_filtered.columns))
    
    return set(df.columns) - set(df_filtered.columns)