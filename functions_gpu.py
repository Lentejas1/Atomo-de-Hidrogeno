import torch


def prob(nL, psi):
    """
    :param psi: Wave function.
    :return: Probability. I.e., |psi|^2.
    """
    p = torch.zeros((nL, nL), device=psi.device)
    for k in range(nL ** 2):
        p[k % nL, k // nL] = torch.sqrt(torch.real(psi[k]) ** 2 + torch.imag(psi[k]) ** 2)
    return p


def coloumb(x, y):
    with torch.no_grad():
        return - 1 / (x ** 2 + y ** 2 + 1e-6)  # Evitar división por cero usando un pequeño valor de seguridad


def slit(slit_y, x, y):
    if slit_y <= y <= slit_y + 0.5:
        if torch.abs(x) > 0.5:
            return 10E6
        else:
            return 0
    else:
        return 0


def double_slit(slit_y, x, y, d, dx):
    if slit_y <= x <= slit_y + dx:
        if -d / 2 - dx <= y <= -d / 2 + dx or d / 2 - dx <= y <= d / 2 + dx:
            return 0
        else:
            return 10E6
    else:
        return 0


def tunnelling(slit_y, y, height, dx):
    if slit_y <= y <= slit_y + dx * 3:
        return height
    else:
        return 0
