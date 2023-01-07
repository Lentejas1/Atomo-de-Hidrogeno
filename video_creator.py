import cv2
import numpy as np
import os
from os.path import isfile, join


def convertToVideo(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key=lambda x: int((x.split(".")[0]).split(" ")[1]))  # REORDENA FRAMES
    for i in range(len(files)):
        filename = pathIn+files[i]
        print(filename)
        img=cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    print("TASK COMPLETED")

#EJECUTAMOS  FUNCIÓN.
directory = "C:/Users/minih/Documents/GitHub/Atomo-de-Hidrogeno/frames/atomo"#RUTA A LA COLECCIÓN DE FRAMES ej:'C:/Users/Antonio/Documents/Mis programas/frames'
pathIn = directory + '/'
pathOut = pathIn + 'animacion.avi'
fps = 20
time = 200/20
convertToVideo(pathIn, pathOut, fps)