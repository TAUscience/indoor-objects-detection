import os
import csv
from segmentation import segmentar
from give_features import caract_fragmento

def csv_coordinates(carpeta_imagenes, num_imagenes, archivo_salida):
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


def csv_characteristics(archivo_entrada, archivo_salida):
    # Leer el archivo CSV de coordenadas
    with open(archivo_entrada, mode='r', newline='') as archivo_csv_entrada:
        reader = csv.reader(archivo_csv_entrada)
        headers = next(reader)  # Leer los encabezados
        
        # Abrir el archivo CSV para escritura
        with open(archivo_salida, mode='w', newline='') as archivo_csv_salida:
            writer = csv.writer(archivo_csv_salida)
            # Escribir la cabecera del archivo CSV de salida
            lista=[]
            for i in range(41):
                lista.append(f"caract{i}")
            lista.append("nombre_imagen")

            writer.writerow(lista)
            
            # Procesar cada fila del CSV de entrada
            for row in reader:
                a, b, w, h, nombre_imagen = row
                coordenadas = [float(a), float(b), float(w), float(h)]
                # Obtener las características usando la función proporcionada

                imagen_scr = f'data/test/images/{nombre_imagen}.png'

                vector_caracteristicas = caract_fragmento(imagen_scr,coordenadas)
                
                # Convertir el vector de características a una lista de cadenas
                vector_caracteristicas_str = [str(x) for x in vector_caracteristicas]
                nombre_sin_extension = os.path.splitext(nombre_imagen)[0]
                # Escribir las características y el nombre de la imagen en el CSV de salida
                writer.writerow(vector_caracteristicas_str)




# Ejemplo de uso
"""carpeta_imagenes = '../data/test/images'
num_imagenes = 2  # Número de imágenes a procesar
archivo_coordenadas = '../data/coordenadas.csv'  # Nombre del archivo CSV de salida de coordenadas
archivo_caracteristicas = '../data/caracteristicas.csv'  # Nombre del archivo CSV de salida de características

# Generar el archivo CSV de coordenadas
csv_coordinates(carpeta_imagenes, num_imagenes, archivo_coordenadas)

# Generar el archivo CSV de características
csv_characteristics(archivo_coordenadas, archivo_caracteristicas)
"""