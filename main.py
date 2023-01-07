import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi
from numpy.linalg import inv
from initial_psi import *
from plots import *
from functions import *

plt.style.use("science")
########### CONDICIONES INICIALES ###########
k_x, k_y = 0 * pi, +15 * pi  # Número de onda inicial (p/hbar)
sigma_0 = 1  # Desviación estándar inicial
x_0, y_0 = 0, 0  # Coordenadas iniciales
nL = 100  # Pasos espaciales NO CAMBIAR (si se ponen más, va hacia atrás idk why)
nT = 300  # Pasos temporales
l = 10  # Borde del mallado (va de -l a l)
dx = (2 * l + 1) / nL  # DeltaX
dt = 0.49 * dx ** 2
r = 1j * dt / (2 * dx ** 2)


def V(n, m):
    potencial = 0
    x = (n - nL // 2) * dx  # x normalizada
    y = (m - nL // 2) * dx
    # potencial += coloumb(x, y)
    potencial += double_slit(4, x, y, 1)
    return potencial


def coloumb(x, y):
    try:
        return - 1 / (x ** 2 + y ** 2)
    except ZeroDivisionError:
        return - 10E6


def slit(slit_y, x, y):
    if slit_y <= x <= slit_y + 0.5:  # No sé por qué va al revés
        if abs(y) > 0.5:
            return 10E6
        else:
            return 0
    else:
        return 0


def double_slit(slit_y, x, y, d):
    if slit_y <= x <= slit_y + 0.5:  # No sé por qué va al revés
        if -d/2 - 0.25 <= y <= -d / 2 + 0.25 or d / 2 - 0.25 <= y <= d / 2 + 0.25:
            return 0
        else:
            return 10E6
    else:
        return 0


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


def open_boundary_conditions(psi_T, L):
    """for k in range(psi_T.shape[0]):
        n = 1 + k // (nL - 2)
        m = 1 + k % (nL - 2)
        if n == 1 or n == ((L - 2) ** 2 - 1):
            psi_T[k] += r * psi_T[k]
        if m == 1 or m == ((L - 2) ** 2 - 1):
            psi_T[k] += r * psi_T[k]"""

    return psi_T


def resolve():
    array_TS = np.dot(A_inv, open_boundary_conditions(np.dot(B, array_T), nL))
    probs = np.zeros((nL, nL))
    for k in range((nL - 2) ** 2):
        probs[1 + k // (nL - 2)][1 + k % (nL - 2)] = np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2)
    return probs, array_TS


psis_t = gaussian_package(x_0, y_0, k_x, k_y, nL - 2, dx, sigma_0)
array_T = psis_t.flatten("C")
heatmap(prob(psis_t), l).savefig(f"frames/free/psi_0.jpg")
A_inv = np.linalg.inv(A_mat(nL))
B = B_mat(nL)
print(f"0: mean = {np.mean(prob(psis_t))} std = {np.std(prob(psis_t))}")
for ts in range(nT):
    probs, array_TS = resolve()
    heatmap(probs, l).savefig(f"frames/free/psi_{ts + 1}.jpg")
    print(f"{ts + 1}/{nT}")
    array_T = array_TS
