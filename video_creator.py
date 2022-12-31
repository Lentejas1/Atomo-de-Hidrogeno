import os
import cv2

path = r"C:/Users/minih/Documents/GitHub/Atomo-de-Hidrogeno/frames/free"
archivos = sorted(os.listdir(path))
img_array = []

for x in range(0, len(archivos)):
    nomArchivo = archivos[x]
    dirArchivo = path + "/" + str(nomArchivo)
    img = cv2.imread(dirArchivo)
    img_array.append(img)

height, width = img.shape[:2]
video = cv2.VideoWriter('Vídeo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 200, (width, height))

for i in range(0, len(archivos)):
    video.write(img_array[i])

video.release()
