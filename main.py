import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi
from numpy.linalg import inv
from initial_psi import *
from plots import *
from functions import *

plt.style.use("science")
########### CONDICIONES INICIALES ###########
k_0 = 5  # Número de onda inicial (p/hbar)
sigma_0 = 0.5  # Desviación estándar inicial
x_0, y_0 = 0, 0  # Coordenadas iniciales
L = 4  # Pasos espaciales
T = 200  # Pasos temporales
l = 10  # Borde del mallado (va de -l a l)
dx = (2 * l + 1) / L  # DeltaX
dt = 0.49 * dx ** 2
r = 1j * dt / (2 * dx ** 2)


def V(n, m):
    return 0


def alpha(i):
    n = i // L
    m = i % L
    return 1 + 4 * r + 1j * dt * V(n, m)


def beta(i):
    n = i // L
    m = i % L
    return 1 - 4 * r - 1j * dt * V(n, m)




def A_mat():
    a = L - 2
    mat = np.zeros((a*a, a*a), complex)
    mat[0][0] = alpha(0)
    mat[-1][-1] = alpha(a - 1)
    mat[0][1] = -r
    mat[-1][-2] = -r
    for i in range(1, a - 1):
        mat[i][i] = alpha(i)
        mat[i][i - 1] = -r
        mat[i][i + 1] = -r
        for k in range(L, a // L - 1):
            mat[i + k * L][i] = -r
            mat[i - k * L][i] = -r
    return mat


def B_mat():
    a = L - 2
    mat = np.zeros((a*a, a*a), complex)
    mat[0][0] = beta(0)
    mat[-1][-1] = beta(a - 1)
    mat[0][1] = r
    mat[-1][-2] = r
    for i in range(1, a - 1):
        mat[i][i] = beta(i)
        mat[i][i - 1] = r
        mat[i][i + 1] = r
        for k in range(L, a // L - 1):
            mat[i + k * L][i] = r
            mat[i - k * L][i] = r
    return mat


def resolve():
    print(array_T.shape)
    A = A_mat()
    B = B_mat()
    array_TS = np.dot(inv(A), np.dot(B, array_T))
    probs = np.zeros((L, L))
    for i in range(L):
        probs[i // L][i % L] = np.sqrt(np.real(array_TS[i]) ** 2 + np.imag(array_TS[i]) ** 2)
    return probs, array_TS

print(A_mat())
psis_t = gaussian_package(x_0, y_0, k_0, L, dx, sigma_0)
print(psis_t.shape)
psis_ts = psis_t
array_T = np.array([psis_t[i][j] for i in range(1, L - 1) for j in range(1, L - 1)], complex).transpose()
"""heatmap(prob(psis_ts), l).savefig(f"frames/psi_0.jpg")
for ts in range(T):
    array_T = resolve()[1]
    heatmap(resolve()[0], l).savefig(f"frames/double_slit/psi_{ts+1}.jpg")
    print(ts+1)"""


