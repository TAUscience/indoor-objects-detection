import random
from data_manage import buscar_objetos_clase
from PIL import Image, ImageDraw
from segmentation import denormalizar_coordenadas
import matplotlib.pyplot as plt
import numpy as np

def colores_etiquetas(k):

    colores = []

    for i in range(k):
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        colores.append([R,G,B])

    return colores

df=buscar_objetos_clase(2)
# Acceder a valores específicos por posición
valor_especifico = df.iloc[0, 1]  # Accede al valor en la primera fila, segunda columna
print(f'Valor específico: {valor_especifico}')

objeto = 0

xmin = df.iloc[objeto,0]
ymin = df.iloc[objeto,1]
xmax = df.iloc[objeto,2]
ymax = df.iloc[objeto,3]

nombre = f'data/test/images/{df.iloc[objeto,4]}.png'

img = Image.open(nombre)
draw = ImageDraw.Draw(img)
coordenadas = denormalizar_coordenadas([xmin,ymin,xmax,ymax],img)

print(coordenadas)
print(xmin,ymin,xmax,ymax)
print(nombre)

draw.rectangle(coordenadas, outline='yellow', width=3)

# Convertir la imagen a un formato compatible con matplotlib
img_np = np.array(img)

# Mostrar la imagen usando matplotlib
plt.imshow(img_np)
plt.axis('off') 
plt.show()