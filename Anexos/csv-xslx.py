import pandas as pd

def csv_to_xlsx(csv_file, xlsx_file):
    try:
        # Leer el archivo CSV
        data = pd.read_csv(csv_file)
        
        # Guardar como archivo XLSX
        data.to_excel(xlsx_file, index=False)
        print(f"Archivo convertido con éxito: {xlsx_file}")
    except Exception as e:
        print(f"Error al convertir el archivo: {e}")

# Uso de la función
csv_file = "normalDt/normalDt-fus-tcpdump.pcap_Flow.csv"  # Nombre del archivo CSV de entrada
xlsx_file = "dataset_normal.xlsx"  # Nombre del archivo XLSX de salida
csv_to_xlsx(csv_file, xlsx_file)
