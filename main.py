import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy import exp, sin, cos, pi

import initial_psi
m = 9.1093837015 / 10 ** 31
e = 1.602176634 / pow(10, 19)
epsilon_0 = 625000 / (22468879468420441 * np.pi)
alpha = m * pow(e, 2) / (4 * np.pi * epsilon_0)
p_0 = m * pow(e, 2) / (4 * np.pi * epsilon_0 * alpha)
dx = .01
sigma = 0.5
x_0, y_0 = 0, 0
L = 100
T = 5

psi = initial_psi.gaussian_package(0, 0, 0, L, dx,sigma)
print(psi)



with open("inicial_state.txt", "w") as f:
    for i in range(-L, L + 1):
        print(i)
        for j in range(-L + 1, L + 1):
            f.write(f"{str(psi[i][j])}\t")
        f.write("\n")

X = np.linspace(-5, 5, 2 * L + 1)
Y = np.linspace(-5, 5, 2 * L + 1)
X, Y = np.meshgrid(X, Y)


def prob(psi):
    p = np.zeros((2 * L + 1, 2 * L + 1))
    for i in range(-L, L + 1):
        for j in range(-L + 1, L + 1):
            p[i][j] = np.vdot(psi[i][j], psi[i][j])
    return p


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, prob(psi), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
