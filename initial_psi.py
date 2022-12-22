import numpy as np
from numpy import exp, sin, cos, pi
def gaussian_package(x_0, y_0, p_0, L, dx, sigma):
    psi_inicial = np.zeros((2 * L + 1, 2 * L + 1), complex)  # x, y, t
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            psi_inicial[i][j] = complex(exp(1 / (2 * sigma ** 2) * ((i * dx - x_0) ** 2 + (j * dx - y_0) ** 2)),
                                        0) * exp(-(complex(0, p_0 * (i * dx - x_0))))
    return psi_inicial


