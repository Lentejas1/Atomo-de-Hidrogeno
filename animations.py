import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from initial_psi import *
from functions import *


def create_frame(step, ax):
    ax.cla()
    psi = gaussian_package(step, step, 0, 100, 0.1, 0.5)
    L = np.shape(psi)[0]
    X = np.linspace(-5, 5, L)
    Y = np.linspace(-5, 5, L)
    X, Y = np.meshgrid(X, Y)
    plt.pcolormesh(X, Y, prob(psi))

    plt.ylabel("$y$")
    plt.xlabel("$x$")


def animate(s):
    fig = plt.figure()
    ax = fig.gca()
    anim = FuncAnimation(fig, create_frame, frames=60 * s, fargs=(ax,))
    anim.save("Animacion.mp4", fps=60, dpi=300)
    plt.show()
