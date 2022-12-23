import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sin, cos, pi
from initial_psi import *
from animations import *
from functions import *
from mpl_toolkits import mplot3d
plt.style.use("science")

k_0 = 0
sigma = 1
x_0, y_0 = 0, 0
L = 100
dx = 0.5
T = 5
l = 10

psi = gaussian_package(x_0, y_0, k_0, L, dx, sigma)
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection='3d')
L = np.shape(psi)[0]
X = np.linspace(-l, l, L)
Y = np.linspace(-l, l, L)
X, Y = np.meshgrid(X, Y)
Z = ax.contour3D(X, Y, psi, 1000, cmap=plt.cm.YlGnBu_r)
cbar = fig.colorbar(Z, shrink=0.5, aspect=5)
cbar.set_label("$\lvert\Psi\\rvert^2$")
ax.set_zlabel("$\lvert\Psi\\rvert^2$")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")

plt.show()

#animate(s=2)



with open("inicial_state.txt", "w") as f:
    for i in range(np.shape(psi)[0]):
        for j in range(np.shape(psi)[0]):
            f.write(f"{str(psi[i][j])}\t")
        f.write("\n")
