from csvs import csv_coordinates, csv_characteristics
from clasification import clasificar_etiquetar

carpeta_imagenes = 'data/test/images'
num_imagenes = 80
archivo_salida_coord = 'data/coordenadas.csv'
archivo_salida_charact = 'data/caracteristicas.csv'

csv_coordinates(carpeta_imagenes, num_imagenes, archivo_salida_coord)
csv_characteristics(archivo_salida_coord, archivo_salida_charact)

x=clasificar_etiquetar("data/caracteristicas.csv",15,"data/etiquetado.csv")
#print(x)