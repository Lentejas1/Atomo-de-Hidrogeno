from numpy import shape, zeros, sqrt, real, imag


def prob(nL, psi):
    """
    :param psi: Wave function.
    :return: Probability. I.e., |psi|^2.
    """
    p = zeros((nL, nL))
    for k in range(nL**2):
        p[k % nL, k // nL] = sqrt(real(psi[k]) ** 2 + imag(psi[k]) ** 2)

    return p


def coloumb(x, y):
    try:
        return - 1 / (x ** 2 + y ** 2)
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
    if slit_y <= y <= slit_y + dx*4:  # No sé por qué va al revés
        if -d / 2 - dx <= x <= -d / 2 + dx or d / 2 - dx <= x <= d / 2 + dx:
            return 0
        else:
            return 50
    else:
        return 0


def tunnelling(slit_y, y, height, dx):
    if slit_y <= y <= slit_y + dx*4:
        return height
    else:
        return 0
