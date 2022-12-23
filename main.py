import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi
from initial_psi import *
from plots import *
from functions import *

plt.style.use("science")
########### CONDICIONES INICIALES ###########
k_0 = 0  # Número de onda inicial (p/hbar)
sigma_0 = 0.5  # Desviación estándar inicial
x_0, y_0 = 0, 0  # Coordenadas iniciales
L = 1000  # Pasos espaciales
T = 5  # Pasos temporales
l = 10  # Borde del mallado (va de -l a l)
dx = 2 * l / L  # DeltaX

psi = gaussian_package(x_0, y_0, k_0, L, dx, sigma_0)

########### PLOTS ###########
heatmap(psi, l)
# plot3d(psi, l) # Consume muchos recursos
# animate(s=2)
