import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocesar(src):
    img = Image.open(src)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Incrementar el contraste usando ecualización del histograma
    equ = cv2.equalizeHist(blurred)

    return img_np, gray, blurred, equ, img

def normalizar_coordenadas(bbox_coords, img):
    x_min, y_min, x_max, y_max = bbox_coords
    normalized_x_min = x_min / img.width
    normalized_y_min = y_min / img.height
    normalized_x_max = x_max / img.width
    normalized_y_max = y_max / img.height

    return [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]

def denormalizar_coordenadas(normalized_coords, img):

    center_x = normalized_coords[0] * img.width
    center_y = normalized_coords[1] * img.height
    half_width = normalized_coords[2] * img.width / 2
    half_height = normalized_coords[3] * img.height / 2

    x_min = int(center_x - half_width)
    y_min = int(center_y - half_height)
    x_max = int(center_x + half_width)
    y_max = int(center_y + half_height)
    return [x_min, y_min, x_max, y_max]

def segmentar(src):
    # Obtener resultados preprocesados
    _, _, blurred, equ, img = preprocesar(src)
    # Cargar modelo YOLOv8s
    model = YOLO('yolov8s.pt')
    # Realizar inferencia con el modelo YOLO
    results = model(src)
    # Lista de cajas bounding boxes
    bounding_boxes = []
    # Filtrar detecciones por tamaño de bounding box
    min_size_threshold = 50
    # Dibujar bounding boxes de YOLO en la imagen
    draw = ImageDraw.Draw(img)
    for result in results:
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width >= min_size_threshold and box_height >= min_size_threshold:
                draw.rectangle([x_min, y_min, x_max, y_max], outline='yellow', width=3)
                normalized_bbox = normalizar_coordenadas([x_min, y_min, x_max, y_max], img)
                bounding_boxes.append(normalized_bbox)
    
    #______Canny
    edges = cv2.Canny(equ, 50, 100)
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_size_threshold and h >= min_size_threshold:
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
            normalized_bbox = normalizar_coordenadas([x, y, x + w, y + h], img)
            bounding_boxes.append(normalized_bbox)
    
    return img, np.array(bounding_boxes), np.array(img)


def agrupar_colores(imagen):

    filas, columnas, _ = imagen.shape

    for i in range(1,filas-1):
        for j in range(1,columnas-1):
            rangoR = imagen[i-1][j][0] - imagen[i][j][0]
            rangoG = imagen[i-1][j][1] - imagen[i][j][1]
            rangoB = imagen[i-1][j][2] - imagen[i][j][2]



# Ejemplo de uso
img, bounding_boxes, _ = segmentar('../data/test/images/1003.png')

# Dibujar recuadro 0.330290 0.562228 0.139548 0.234109
normalized_coords = []



for normalized_coord in normalized_coords:
    denormalized_coord = denormalizar_coordenadas(normalized_coord, img)
    # Dibujar el recuadro adicional
    draw = ImageDraw.Draw(img)
    draw.rectangle(denormalized_coord, outline='pink', width=3)

# Mostrar la imagen resultante (opcional)
plt.imshow(img)
plt.show()

print(bounding_boxes)
