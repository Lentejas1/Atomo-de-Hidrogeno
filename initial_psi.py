import numpy as np
from numpy import zeros, exp, pi, sin, rot90, sqrt, real, imag
from scipy.special import gamma
from functions import prob

def modos_normales(X, Y, k_x, k_y, low_lim, up_lim, L, dx):
    """
    :param k_x: x mode
    :param k_y: y mode
    :param low_lim: Lower spatial limit
    :param up_lim: Upper spatial limit
    :param l: Length
    :param L: Spatial steps
    :param dx: dx
    :return:
    """
    l = up_lim - low_lim
    x, y = np.linspace(low_lim, up_lim + dx, L, dtype=complex), np.linspace(low_lim, up_lim + dx, L, dtype=complex)
    X, Y = np.meshgrid(x, y)
    psi_inicial = 1 / l * sin(k_x / (2 * l) * X) * sin(k_y / (2 * l) * Y) * 1 / 4
    psi_inicial[0:][0] = psi_inicial[0:][-1] = psi_inicial[0][0:] = psi_inicial[-1][0:] = 0
    psi_inicial = np.flipud(psi_inicial)
    return psi_inicial


def gaussian_package(X, Y, x_0, y_0, k_x, k_y, low_lim, up_lim, L, dx, sigma):
    """
    :param x_0: Initial x coordinate.
    :param y_0: Initial y coordinate.
    :param k_x: Initial wave number (i.d.: p/hbar).
    :param k_x: Initial wave number (i.d.: p/hbar).
    :param L: Number of spatial steps per dimenion: L x L canvas.
    :param dx: Space step.
    :param sigma: Initial standard deviation of the gaussian package. Note that it's the same for both spatial
    dimensions.
    :return: A gaussian package centered in x_0, y_0 with a momentum kick of k_0 * hbar.
    """
    x, y = np.linspace(low_lim, up_lim + dx, L, dtype=complex), np.linspace(low_lim, up_lim + dx, L, dtype=complex)
    X, Y = np.meshgrid(x, y)
    psi_inicial = exp(-((X - x_0) ** 2 + (Y - y_0) ** 2) / (4 * sigma ** 2)) * \
                  momentum_kick(k_x, k_y, X, Y)
    suma = 0
    for i in range(125):
        for j in range(125):
            suma += sqrt(real(psi_inicial[i, j]) ** 2 + imag(psi_inicial[i, j]) ** 2)
    psi_inicial = psi_inicial / (suma * dx ** 2)
    psi_inicial[0:][0] = psi_inicial[0:][- 1] = psi_inicial[0][0:] = psi_inicial[- 1][0:] = 0
    #psi_inicial = np.flip(psi_inicial, axis=0)
    #psi_inicial = np.flip(psi_inicial, axis=1)
    return psi_inicial


def onda_plana(X, Y, A, x_0, y_0, k_x, k_y, low_lim, up_lim, L, dx):
    x, y = np.linspace(low_lim, up_lim + dx, L, dtype=complex), np.linspace(low_lim, up_lim + dx, L, dtype=complex)
    X, Y = np.meshgrid(x, y)
    psi_inicial = A * exp(1j * (k_x * X + k_y * Y))
    return psi_inicial


def momentum_kick(k_x, k_y, X, Y):
    return exp(1j * k_x * X + 1j * k_y * Y)


def hydrogen_bounded_state(X, Y, n):
    if n == 1:
        psi = 1 / sqrt(4 * pi) * exp(-sqrt(X ** 2 + Y ** 2) / 2) * sqrt((2 / n) ** 3 * 1 / (2 * n * gamma(n)))
    else:
        psi = 1 / sqrt(4 * pi) * exp(-sqrt(X ** 2 + Y ** 2) / 2) * sqrt(
            (2 / n) ** 3 * gamma(n - 1) / (2 * n * gamma(n)))
    return psi
