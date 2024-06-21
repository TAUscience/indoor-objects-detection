import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocesar(src):

    img = Image.open(src)
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Aplicar suavizado Gaussiano
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return img_np,gray,blurred,img


def segmentar(src):
    #Obtener resultados preprocesados
    _, _, blurred, img = preprocesar(src)
    # Cargar modelo YOLOv5s
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    #YOLO
    results = model(src)
    #Lista de cajas bounding boxes
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
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            bounding_boxes.append([x_min, y_min, x_max, y_max])

    # Canny
    edges = cv2.Canny(blurred, 100, 200)
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_size_threshold and h >= min_size_threshold:
            draw.rectangle([x, y, x + w, y + h], outline='blue', width=2)
            bounding_boxes.append([x, y, x + w, y + h])

    return img, np.array(bounding_boxes), np.array(img)



#____________Ejemplo de uso
'''


img, bounding_boxes, _  = segmentar('../data/test/images/888.png')

output_image_path = 'output_image.jpg'
img.save(output_image_path)

print(bounding_boxes)

'''