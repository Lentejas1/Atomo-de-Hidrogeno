import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi
from numpy.linalg import inv
from initial_psi import *
from plots import *
from functions import *

plt.style.use("science")
########### CONDICIONES INICIALES ###########
k_0 = 2 * pi  # Número de onda inicial (p/hbar)
sigma_0 = 1  # Desviación estándar inicial
x_0, y_0 = 0, 0  # Coordenadas iniciales
nL = 50  # Pasos espaciales
nT = 100  # Pasos temporales
l = 10  # Borde del mallado (va de -l a l)
dx = (2 * l + 1) / nL  # DeltaX
dt = 0.25 * dx ** 2
r = 1j * dt / (2 * dx ** 2)


def V(n, m):
    potencial = 0
    x = (n - nL // 2) * dx  # x normalizada
    y = (m - nL // 2) * dx
    #coloumb(potencial, x, y)
    slit(potencial, 4, x, y)
    return potencial


def coloumb(potencial, x, y):
    try:
        potencial += - 1 / (x ** 2 + y ** 2)
    except ZeroDivisionError:
        pass


def slit(potencial, X, x, y):
    if X <= x <= X + 0.2:
        if not -0.5 < y < 0.5:
            potencial += 10E6



def alpha(k):
    n = 1 + k // (nL - 2)
    m = 1 + k % (nL - 2)
    return 1 + 4 * r + 1j * dt * V(n, m) / 2


def beta(k):
    n = 1 + k // (nL - 2)
    m = 1 + k % (nL - 2)
    return 1 - 4 * r - 1j * dt * V(n, m) / 2


def A_mat(L=nL):
    D = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    E = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    for k in range(1, (L - 2) ** 2 - 1):
        D[k][k] = alpha(k)
        E[k][k - 1] = -r
        E[k][k + 1] = -r
        if k + (L - 2) < (L - 2) ** 2:
            E[k][k + (L - 2)] = -r
        if k - (L - 2) >= 0:
            E[k][k - (L - 2)] = -r
    D[0][0] = alpha(0)
    D[-1][-1] = alpha(L - 2)
    E[0][1] = E[-1][-2] = -r
    return D + E


def B_mat(L=nL):
    D = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    E = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    for k in range(1, (L - 2) ** 2 - 1):
        D[k][k] = beta(k)
        E[k][k - 1] = r
        E[k][k + 1] = r
        if k + (L - 2) < (L - 2) ** 2:
            E[k][k + (L - 2)] = r
        if k - (L - 2) >= 0:
            E[k][k - (L - 2)] = r
    D[0][0] = beta(0)
    D[-1][-1] = beta(L - 2)
    E[0][1] = E[-1][-2] = r
    return D + E


def resolve():
    A_inv = np.linalg.inv(A_mat(nL))
    B = B_mat(nL)
    array_TS = np.dot(A_inv, np.dot(B, array_T))
    probs = np.zeros((nL, nL))
    for k in range((nL - 2) ** 2):
        probs[1 + k // (nL - 2)][1 + k % (nL - 2)] = np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2)
    return probs, array_TS


psis_t = gaussian_package(x_0, y_0, k_0, nL - 2, dx, sigma_0)
array_T = psis_t.flatten("C")
heatmap(prob(psis_t), l).savefig(f"frames/slit/psi_0.jpg")
print(f"0: mean = {np.mean(prob(psis_t))} std = {np.std(prob(psis_t))}")
for ts in range(nT):
    probs, array_TS = resolve()
    heatmap(probs, l).savefig(f"frames/slit/psi_{ts + 1}.jpg")
    print(f"{ts + 1}")
    array_T = array_TS
