import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

def clasificar_etiquetar(ruta_ds, k, ruta_guardado):
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

    # Reducir las dimensiones a 3 utilizando PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_imputed_scaled)

    # Inicializar KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_imputed_scaled)

    # Calcular el coeficiente de silueta
    silhouette_avg = silhouette_score(X_imputed_scaled, kmeans.labels_)
    print(f'Coeficiente de silueta promedio: {silhouette_avg}')

    # Agregar las etiquetas al DataFrame original
    data['cluster_label'] = kmeans.labels_
    data.to_csv(ruta_guardado, index=False)

    # Mostrar un mensaje de confirmación
    print('Conjunto con etiquetas generado exitosamente.')

    # Visualizar los clusters en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans.labels_, s=50, cmap='viridis')
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], c='red', s=200, alpha=0.75)

    # Leyenda y etiquetas
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

    return data


"""Ejemplo de uso"""
#x=clasificar_etiquetar("../data/datasetPrueba.csv",3,"../data/datasetPruebaEtiquetado.csv")
#print(x)

