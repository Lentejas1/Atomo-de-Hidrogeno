import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi
from numpy.linalg import inv
from initial_psi import *
from plots import *
from functions import *

plt.style.use("science")
########### CONDICIONES INICIALES ###########
k_x, k_y = 0 * pi, 0 * pi  # Número de onda inicial (p/hbar)
sigma_0 = 1  # Desviación estándar inicial
x_0, y_0 = 1, 0  # Coordenadas iniciales
nL = 100  # Pasos espaciales
nT = 899  # Pasos temporales
l = 8  # Borde del mallado (va de -l a l)
dx = (2 * l + 1) / nL  # DeltaX
dt = 0.25 * dx ** 2
r = 1j * dt / (2 * dx ** 2)


def V(n, m):
    potencial = 0
    x = (n - nL // 2) * dx  # x normalizada
    y = (m - nL // 2) * dx
    potencial += coloumb(x, y)
    # slit(potencial, 4, x, y)
    return potencial


def coloumb(x, y):
    try:
        return - 1 / (x ** 2 + y ** 2)
    except ZeroDivisionError:
        return - 10E6


def slit(potencial, slit_x, x, y):
    if slit_x <= x <= slit_x + 0.5:
        if abs(y) > 0.1:
            potencial += np.inf


def alpha(k):
    n = 1 + k // (nL - 2)
    m = 1 + k % (nL - 2)
    return 1 + 4 * r + 1j * dt * V(n, m) / 2


def beta(k):
    n = 1 + k // (nL - 2)
    m = 1 + k % (nL - 2)
    return 1 - 4 * r - 1j * dt * V(n, m) / 2


def A_mat(L=nL):
    A = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    for k in range(1, (L - 2) ** 2 - 1):
        A[k][k] = alpha(k)
        A[k][k - 1] = -r
        A[k][k + 1] = -r
        if k + (L - 2) < (L - 2) ** 2:
            A[k][k + (L - 2)] = -r
        if k - (L - 2) >= 0:
            A[k][k - (L - 2)] = -r
    A[0][0] = alpha(0)
    A[-1][-1] = alpha(L - 2)
    A[0][1] = A[-1][-2] = -r
    return A


def B_mat(L=nL):
    B = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    for k in range(1, (L - 2) ** 2 - 1):
        B[k][k] = beta(k)
        B[k][k - 1] = r
        B[k][k + 1] = r
        if k + (L - 2) < (L - 2) ** 2:
            B[k][k + (L - 2)] = r
        if k - (L - 2) >= 0:
            B[k][k - (L - 2)] = r
    B[0][0] = beta(0)
    B[-1][-1] = beta(L - 2)
    B[0][1] = B[-1][-2] = r
    return B


def resolve():
    array_TS = np.dot(A_inv, np.dot(B, array_T))
    probs = np.zeros((nL, nL))
    for k in range((nL - 2) ** 2):
        probs[1 + k // (nL - 2)][1 + k % (nL - 2)] = np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2)
    return probs, array_TS


psis_t = gaussian_package(x_0, y_0, k_x, k_y, nL - 2, dx, sigma_0)
array_T = psis_t.flatten("C")
heatmap(prob(psis_t), l).savefig(f"frames/atomo/psi_0.jpg")
A_inv = np.linalg.inv(A_mat(nL))
B = B_mat(nL)
print(f"0: mean = {np.mean(prob(psis_t))} std = {np.std(prob(psis_t))}")
for ts in range(nT):
    probs, array_TS = resolve()
    heatmap(probs, l).savefig(f"frames/atomo/psi_{ts + 1}.jpg")
    print(f"{ts + 1}")
    array_T = array_TS

"""def open_boundary_conditions(array_T, k):
    n = 1 + k // (nL - 2)
    m = 1 + k % (nL - 2)
    if n == 1:
        array_T[k] -= algo
    if n == (L-2):
        algo
        
    igual con m """