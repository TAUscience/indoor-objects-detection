import os
import csv
from segmentation import segmentar


def csv_coordenadas(carpeta_imagenes, num_imagenes, archivo_salida):
    # Obtener la lista de archivos en la carpeta de imágenes
    archivos_imagenes = os.listdir(carpeta_imagenes)
    # Filtrar solo los archivos de imagen
    archivos_imagenes = [f for f in archivos_imagenes if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limitar el número de imágenes a procesar
    archivos_imagenes = archivos_imagenes[:num_imagenes]
    
    # Abrir el archivo CSV para escritura
    with open(archivo_salida, mode='w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(['a','b','w','h','nombre_imagen'])
        # Procesar cada imagen
        for nombre_imagen in archivos_imagenes:
            # Obtener la ruta completa de la imagen
            imagen_path = os.path.join(carpeta_imagenes, nombre_imagen)
            # Obtener las coordenadas
            _,coordenadas,_ = segmentar(imagen_path)
            nombre_sin_extension = os.path.splitext(nombre_imagen)[0]

            for coordenada in coordenadas:
                writer.writerow([coordenada[0], coordenada[1], coordenada[2], coordenada[3], nombre_sin_extension])


# Ejemplo de uso
carpeta_imagenes = '../data/test/images'
num_imagenes = 5  # Número de imágenes a procesar
archivo_salida = '../data/coordenadas.csv'  # Nombre del archivo CSV de salida

csv_coordenadas(carpeta_imagenes, num_imagenes, archivo_salida)
