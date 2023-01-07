import numpy as np
from numpy import zeros, exp, pi
from scipy.stats import multivariate_normal


def gaussian_package(x_0, y_0, k_x, k_y, L, dx, sigma):
    """
    :param x_0: Initial x coordinate.
    :param y_0: Initial y coordinate.
    :param k_0: Initial wave number (i.d.: p/hbar).
    :param L: Number of spatial steps per dimenion: L x L canvas.
    :param dx: Space step.
    :param sigma: Initial standard deviation of the gaussian package. Note that it's the same for both spatial
    dimensions.
    :return: A gaussian package centered in x_0, y_0 with a momentum kick of k_0 * hbar.
    """
    psi_inicial = zeros((L, L), complex)  # x, y, t
    for xs in range(L):  # xs e ys es el step espacial
        for ys in range(L):
            x = (xs - L // 2) * dx  # x normalizada
            y = (ys - L // 2) * dx  # y normalizada
            psi_inicial[xs][ys] = 1 / (2 * pi * sigma ** 2) ** (1 / 4) * exp(
                -((x - x_0) ** 2 + (y - y_0) ** 2) / (2 * sigma ** 2))  # psi_0
            psi_inicial[xs][ys] *= momentum_kick(k_x, k_y, x, y)  # Momentum kick"""
    psi_inicial[0:][0] = psi_inicial[0:][L - 1] = psi_inicial[0][0:] = psi_inicial[L - 1][0:] = 0
    return np.rot90(psi_inicial)


def momentum_kick(k_x, k_y, x, y):
    return exp(-1j * k_x * x) * exp(-1j * k_y * y)
