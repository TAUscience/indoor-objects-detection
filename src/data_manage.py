import pandas as pd

def copiar_etiquetas(ruta_dataset_a, ruta_dataset_b):
    # Leer datasets
    df_a = pd.read_csv(ruta_dataset_a)
    df_b = pd.read_csv(ruta_dataset_b)
    
    # Copiar la columna cluster_label de A a B
    df_b['cluster_label'] = df_a['cluster_label']
    
    # Guardar el dataset B actualizado
    df_b.to_csv("data/etiquetado_coord.csv", index=False)
    
    print(f'Se ha copiado la columna cluster_label de {ruta_dataset_a} a {ruta_dataset_b}.')

def extraer_por_etiqueta(ruta_dataset_b, etiqueta_interes):
    # Leer dataset B
    df_b = pd.read_csv(ruta_dataset_b)
    
    # Extraer las instancias con la etiqueta de interés
    df_interes = df_b[df_b['cluster_label'] == etiqueta_interes]
    
    return df_interes

def buscar_objetos_clase(clase):
    ruta_dataset_a = 'data/etiquetado.csv'
    ruta_dataset_b = 'data/coordenadas.csv'
    copiar_etiquetas(ruta_dataset_a, ruta_dataset_b)
    
    ruta_dataset_busqueda="data/etiquetado_coord.csv"
    # Cambiar por la etiqueta específica que desees extraer
    df_instancias_interes = extraer_por_etiqueta(ruta_dataset_busqueda, clase)
    print(df_instancias_interes)
    return df_instancias_interes

"""
EJEMPLO DE USO

df=buscar_objetos_clase(0)
# Acceder a valores específicos por posición
valor_especifico = df.iloc[0, 1]  # Accede al valor en la primera fila, segunda columna
print(f'Valor específico: {valor_especifico}')
"""