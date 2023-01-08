from numpy import shape, zeros, sqrt, real, imag


def prob(psi):
    """
    :param psi: Wave function.
    :return: Probability. I.e., |psi|^2.
    """
    L = shape(psi)[0]
    p = zeros((L, L))
    for i in range(L):
        for j in range(L):
            p[i][j] = sqrt(real(psi[i][j]) ** 2 + imag(psi[i][j]) ** 2)
    return p


def coloumb(x, y):
    try:
        return - 1 / ((x - 4) ** 2 + y ** 2)
    except ZeroDivisionError:
        return - 10E6


def slit(slit_y, x, y):
    if slit_y <= y <= slit_y + 0.5:
        if abs(x) > 0.5:
            return 10E6
        else:
            return 0
    else:
        return 0


def double_slit(slit_y, x, y, d, dx):
    if slit_y <= y <= slit_y + 0.5:  # No sé por qué va al revés
        if -d / 2 - dx <= x <= -d / 2 + dx or d / 2 - dx <= x <= d / 2 + dx:
            return 0
        else:
            return 10E6
    else:
        return 0
