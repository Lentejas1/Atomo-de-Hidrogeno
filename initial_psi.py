from numpy import zeros, exp, pi


def gaussian_package(x_0, y_0, k_0, L, dx, sigma):
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
    for xs in range(L):
        for ys in range(L):
            x = (xs - L // 2)*dx
            y = (ys - L // 2)*dx
            psi_inicial[xs][ys] = 1/(4*pi*sigma**2)**(1/4) * exp(-((x-x_0)**2+(y-y_0)**2)/(8*sigma**2))
            psi_inicial[xs][ys] *= exp(-1j*p_0*x_0)
    return psi_inicial
