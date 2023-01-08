import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functions import prob
from numpy import shape, meshgrid, linspace

def heatmap(psi, neg_lim, pos_lim):
    """
    :param psi: Wave function.
    :param l: Limits of our canvas. Range = (-l,l)
    :return: Heatmap with probability density.
    """
    plt.close()
    fig = plt.figure(figsize=(16, 9))
    L = shape(psi)[0]
    X, Y = meshgrid(linspace(neg_lim, pos_lim, L), linspace(neg_lim, pos_lim, L))
    Z = prob(psi)
    plt.pcolormesh(X, Y, Z)  # , vmin=0, vmax=0.65
    plt.ylabel("$y$")
    plt.xlabel("$x$")
    plt.axis('scaled')
    cbar = plt.colorbar()
    cbar.set_label("$\lvert\Psi\\rvert^2$")
    return fig
    #plt.show()


def plot3d(psi, l):
    """
    :param psi: Wave function.
    :param l: Limits of our canvas. Range = (-l,l)
    :return: 3D plot with probability density.
    """
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')
    L = shape(psi)[0]
    X, Y = meshgrid(linspace(-l, l, L), linspace(-l, l, L))
    Z = ax.contour3D(X, Y, prob(psi), 1000, cmap=plt.cm.YlGnBu_r)
    #cbar = fig.colorbar(Z, shrink=0.5, aspect=5)
    #cbar.set_label("$\lvert\Psi\\rvert^2$")
    ax.set_zlabel("$\lvert\Psi\\rvert^2$")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    plt.show()


"""def create_frame(step, ax):
    ax.cla()
    psi = gaussian_package(step, step, 0, 100, 0.1, 0.5)
    L = np.shape(psi)[0]
    X, Y = np.meshgrid(np.linspace(-10, 10, L), np.linspace(-10, 10, L))
    plt.pcolormesh(X, Y, prob(psi))

    plt.ylabel("$y$")
    plt.xlabel("$x$")"""


def animate(s):
    fig = plt.figure()
    ax = fig.gca()
    anim = FuncAnimation(fig, create_frame, frames=60 * s, fargs=(ax,))
    anim.save("Animacion.mp4", fps=60, dpi=300)
    plt.show()
