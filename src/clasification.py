import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def clasificar_etiquetar(ruta_ds,k,ruta_guardado):

    """Carga y preparación de datos"""
    # Leer el archivo CSV llamado 'dataset.csv'
    data = pd.read_csv(ruta_ds)
    print(type(data))

    # Todas las columnas son características
    X = data.values

    # Imputar valores faltantes y escalar los datos
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_imputed_scaled = scaler.fit_transform(imputer.fit_transform(X))

    # Inicializar KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_imputed_scaled)

    # Calcular el coeficiente de silueta
    silhouette_avg = silhouette_score(X_imputed_scaled, kmeans.labels_)
    print(f'Coeficiente de silueta promedio: {silhouette_avg}')

    # Agregar las etiquetas al DataFrame original
    data['cluster_label'] = kmeans.labels_
    data.to_csv(ruta_guardado,index=False)

    # Mostrar un mensaje de confirmación
    print('Conjunto con etiquetas generado exitosamente.')

    # Visualizar los clusters (opcional)
    plt.scatter(X_imputed_scaled[:, 0], X_imputed_scaled[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.show()

    return data


"""Ejemplo de uso"""
x=clasificar_etiquetar("../data/datasetPrueba.csv",3,"../data/datasetPruebaEtiquetado.csv")
print(x)

