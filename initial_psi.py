from numpy import zeros, exp, pi


def gaussian_package(x_0, y_0, p_0, L, dx, sigma):
    psi_inicial = zeros((2 * L + 1, 2 * L + 1), complex)  # x, y, t
    for xs in range(2*L+1):
        for ys in range(2*L+1):
            x = (xs - L)*dx
            y = (ys - L)*dx
            psi_inicial[xs][ys] = 1/(4*pi*sigma**2)**(1/4) * exp(-((x-x_0)**2+(y-y_0)**2)/(8*sigma**2))
            psi_inicial[xs][ys] *= exp(-1j*p_0*x_0)

    return psi_inicial
