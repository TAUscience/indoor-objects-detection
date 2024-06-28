import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.feature_selection import SelectKBest, chi2

def select_features_with_chi2(features, k=10):
    selector = SelectKBest(score_func=chi2, k=k)
    selected_features = selector.fit_transform(features.reshape(1, -1))
    return selected_features.flatten()

def caract_lbp(fragmento, P=8, R=1.0, method='uniform', tamano=30):
    # Calcular LBP
    lbp = local_binary_pattern(fragmento, P, R, method=method)
    
    # Aplanar LBP en un arreglo 1D
    lbp_features = lbp.ravel()

    # Asegurar que el vector de características tenga exactamente el tamaño deseado
    if len(lbp_features) < tamano:
        lbp_features = np.pad(lbp_features, (0, tamano - len(lbp_features)), mode='constant')
    elif len(lbp_features) > tamano:
        lbp_features = lbp_features[:tamano]
    
    return lbp_features

def caract_hu_moments(fragmento, tamano=7):
    # Calcular momentos de Hu
    moments = cv2.moments(fragmento)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Asegurar que el vector de características tenga exactamente el tamaño deseado
    if len(hu_moments) < tamano:
        hu_moments = np.pad(hu_moments, (0, tamano - len(hu_moments)), mode='constant')
    elif len(hu_moments) > tamano:
        hu_moments = hu_moments[:tamano]
    
    return hu_moments

def caract_contours(fragmento, tamano=4):
    # Encontrar contornos en la imagen binaria del fragmento
    contours, _ = cv2.findContours(fragmento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extraer características de los contornos (por ejemplo, área, perímetro)
    if len(contours) > 0:
        contour = contours[0]  # Tomar el primer contorno encontrado
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        moments = cv2.moments(contour)
        centroid_x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else 0
        centroid_y = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else 0
        
        # Ejemplo de vector de características
        contour_features = [area, perimeter, centroid_x, centroid_y]
    else:
        contour_features = [0, 0, 0, 0]  # En caso de no encontrar contornos
    
    # Asegurar que el vector de características tenga exactamente el tamaño deseado
    if len(contour_features) < tamano:
        contour_features = np.pad(contour_features, (0, tamano - len(contour_features)), mode='constant')
    elif len(contour_features) > tamano:
        contour_features = contour_features[:tamano]
    
    return np.array(contour_features)

def caract_fragmento(ruta_img, normalized_coords):
    # Leer la imagen en escala de grises
    image = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file '{ruta_img}' not found.")
    
    # Obtener las dimensiones de la imagen
    img_height, img_width = image.shape

    # Calcular las coordenadas del fragmento
    center_x = normalized_coords[0] * img_width
    center_y = normalized_coords[1] * img_height
    half_width = normalized_coords[2] * img_width / 2
    half_height = normalized_coords[3] * img_height / 2

    x_min = int(center_x - half_width)
    y_min = int(center_y - half_height)
    x_max = int(center_x + half_width)
    y_max = int(center_y + half_height)

    # Asegurarse de que las coordenadas están dentro de los límites de la imagen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    # Extraer el fragmento de la imagen
    fragmento = image[y_min:y_max, x_min:x_max]
    
    lbp_features = caract_lbp(fragmento)
    hu_moments = caract_hu_moments(fragmento)
    contour_features = caract_contours(fragmento)

    # Concatenar todos los vectores de características en uno solo
    vector = np.concatenate((lbp_features, hu_moments, contour_features))

    return vector

# Ejemplo de uso
"""
ruta_img = "data/b1.jpg"
normalized_coords = [0.125,0.75,0.25,0.25]
vector = caract_fragmento(ruta_img, normalized_coords)

print(vector)
print(len(vector))  # Esto debería imprimir la longitud del vector concatenado
"""