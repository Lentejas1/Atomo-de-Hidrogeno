import numpy as np
from numpy.linalg import inv

L = 100  # Pasos espaciales
T = 200  # Pasos temporales
l = 10  # Borde del mallado (va de -l a l)
dx = (2 * l + 1) / L  # DeltaX
dt = 0.49 * dx ** 2
r = -1j * dt / (2 * dx ** 2)

array_T = np.array([0 for _ in range(1, L - 1) for _ in range(1, L - 1)], complex).transpose()

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


a = np.shape(array_T)[0]


def A_mat():
    mat = np.zeros((np.shape(array_T)[0], np.shape(array_T)[0]), complex)
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
    mat = np.zeros((a, a), complex)
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
    A = A_mat()
    B = B_mat()
    array_TS = np.dot(inv(A), np.dot(B, array_T))
    probs = np.zeros((L, L))
    for i in range(L):
        probs[i // L][i % L] = np.sqrt(np.real(array_TS[i])**2+np.imag(array_TS[i])**2)
    return probs


resolve()
