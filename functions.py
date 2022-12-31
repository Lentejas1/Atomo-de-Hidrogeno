import numpy as np


def prob(psi):
    """
    :param psi: Wave function.
    :return: Probability. I.e., |psi|^2.
    """
    L = np.shape(psi)[0]
    p = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            p[i][j] = np.sqrt(np.real(psi[i][j]) ** 2 + np.imag(psi[i][j]) ** 2)
    return p


def double_slit(psi, L):
    psi[750:760, 0:(L // 2 - 20)] = psi[750:760, (L // 2 - 10):(L // 2 + 10)] = psi[750:760, (L // 2 + 20):L] = 0
    return psi


def y_wall(y, ys):
    if y < ys < y + 50:
        return 100 + 0j
    else:
        return 0
