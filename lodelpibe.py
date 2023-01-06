# -*- coding: utf-8 -*-
"""
@author: Arturo Mena López
Script to simulate the passage of a Gaussian packet wave function through a
double slit with hard-walls (infinite potential barrier; the wave function
cancels inside the walls).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def psi0(x, y, x0, y0, sigma=0.5, k=15 * np.pi):
    """
    Proposed wave function for the initial time t=0.
    Initial position: (x0, y0)
    Default parameters:
        - sigma = 0.5 -> Gaussian dispersion.
        - k = 15*np.pi -> Proportional to the momentum.

    Note: if Dy=0.1 use np.exp(-1j*k*(x-x0)), if Dy=0.05 use
          np.exp(1j*k*(x-x0)) so that the particle will move
          to the right.
    """

    return np.exp(-1 / 2 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2) * np.exp(1j * k * (x - x0))


def doubleSlit_interaction(psi, j0, j1, i0, i1, i2, i3):
    """
    Function responsible of the interaction of the psi wave function with the
    double slit in the case of rigid walls.

    The indices j0, j1, i0, i1, i2, i3 define the extent of the double slit.
    slit.

    Input parameters:

        psi -> Numpy array with the values of the wave function at each point
               in 2D space.

        Indices that parameterize the double slit in the space of
        points:

            Horizontal axis.
                j0 -> Left edge.
                j1 -> Right edge.

            Vertical axis.
                i0 -> Lower edge of the lower slit.
                i1 -> Upper edge of the lower slit.
                i2 -> Lower edge of upper slit.
                i3 -> Upper edge of upper slit.
    Returns the array with the wave function values at each point in 2D space
    updated with the interaction with the double slit of rigid walls.
    """

    psi = np.asarray(psi)  # Ensures that psi is a numpy array.

    # We cancel the wave function inside the walls of the double slit.
    psi[0:i3, j0:j1] = 0
    psi[i2:i1, j0:j1] = 0
    psi[i0:, j0:j1] = 0

    return psi


# =============================================================================
# Parameters
# =============================================================================

L = 20  # Well of width L. Shafts from 0 to +L.
Dy = 0.21  # Spatial step size.
Dt = Dy ** 2 / 4  # Temporal step size.
Nx = int(L / Dy) + 1  # Number of points on the x axis.
Ny = int(L / Dy) + 1  # Number of points on the y axis.
Nt = 500  # Number of time steps.
rx = -Dt / (2j * Dy ** 2)  # Constant to simplify expressions.
ry = -Dt / (2j * Dy ** 2)  # Constant to simplify expressions.

# Initial position of the center of the Gaussian wave function.
x0 = L / 2
y0 = L / 2


v = np.zeros((Ny, Ny), complex)  # Potential.

Ni = (Nx - 2) * (Ny - 2)  # Number of unknown factors v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2

# =============================================================================
# First step: Construct the matrices of the system of equations.
# =============================================================================

# Matrices for the Crank-Nicolson calculus. The problem A·x[n+1] = b = M·x[n] will be solved at each time step.
A = np.zeros((Ni, Ni), complex)

# We fill the A and M matrices.
for k in range(Ni):

    # k = (i-1)*(Ny-2) + (j-1)
    i = 1 + k // (Ny - 2)
    j = 1 + k % (Ny - 2)

    # Main central diagonal.
    A[k, k] = 1 + 2 * rx + 2 * ry + 1j * Dt / 2 * v[i, j]

    if i != 1:  # Lower lone diagonal.
        A[k, (i - 2) * (Ny - 2) + j - 1] = -ry

    if i != Nx - 2:  # Upper lone diagonal.
        A[k, i * (Ny - 2) + j - 1] = -ry

    if j != 1:  # Lower main diagonal.
        A[k, k - 1] = -rx

    if j != Ny - 2:  # Upper main diagonal.
        A[k, k + 1] = -rx

with open("A_pibe.csv", "w") as f:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            f.write(f"{A[i][j]},")
        f.write("\n")
# =============================================================================
# Second step: Solve the A·x[n+1] = M·x[n] system for each time step.
# =============================================================================


from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

