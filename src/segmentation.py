import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocesar(src):
    img = Image.open(src)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return img_np, gray, blurred, img

def normalizar_coordenadas(bbox_coords, img):
    x_min, y_min, x_max, y_max = bbox_coords
    normalized_x_min = x_min / img.width
    normalized_y_min = y_min / img.height
    normalized_x_max = x_max / img.width
    normalized_y_max = y_max / img.height

    return [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]

def segmentar(src):
    # Obtener resultados preprocesados
    _, _, blurred, img = preprocesar(src)
    # Cargar modelo YOLOv5s
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    #_______YOLO
    results = model(src)
    # Lista de cajas bounding boxes
    bounding_boxes = []
    # Obtener DataFrame
    df = results.pandas().xyxy[0]
    # Filtrar detecciones por tamaño de bounding box
    min_size_threshold = 50
    # Dibujar bounding boxes de YOLO en la imagen
    draw = ImageDraw.Draw(img)
    for index, row in df.iterrows():
        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width >= min_size_threshold and box_height >= min_size_threshold:
            draw.rectangle([x_min, y_min, x_max, y_max], outline='yellow', width=3)
            normalized_bbox = normalizar_coordenadas([x_min, y_min, x_max, y_max], img)
            bounding_boxes.append(normalized_bbox)
    
    #______Canny
    edges = cv2.Canny(blurred, 100, 200)
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_size_threshold and h >= min_size_threshold:
            draw.rectangle([x, y, x + w, y + h], outline='yellow', width=2)
            normalized_bbox = normalizar_coordenadas([x, y, x + w, y + h], img)
            bounding_boxes.append(normalized_bbox)
    
    return img, np.array(bounding_boxes), np.array(img)

# Ejemplo de uso
'''

img, bounding_boxes, _ = segmentar('../data/test/images/1280.png')

# Guardar la imagen resultante
output_image_path = 'output_image.jpg'
img.save(output_image_path)

# Mostrar la imagen resultante (opcional)
plt.imshow(img)
plt.show()

print(bounding_boxes)


'''