import numpy as np
import time
from numpy.linalg import inv
from initial_psi import *
from plots import *
from functions import *

plt.style.use("science")

##################################
# CONDICIONES INICIALES TÉCNICAS #
##################################

nL = 125  # Pasos espaciales NO CAMBIAR de 100 (si se ponen más, va hacia atrás idk why)
nT = 200  # Pasos temporales
l = 10  # Borde del mallado (va de -l a l o de 0 a 2l según centrado, True/False respectivamente)
dx = (2 * l) / (nL - 1)  # DeltaX
ratio = 0.49
dt = ratio * dx ** 2
r = 1j * dt / (2 * dx ** 2)
obc = False
centrado = True

#####################################
# PARÁMETROS DE LAS FUNCIÓN DE ONDA #
#####################################

# PULSO
k_x, k_y = 15 * pi, 15 * pi  # Número de onda inicial (p/hbar)   E=(k_x^2+k_y^2)/2
sigma_0 = 0.5  # Desviación estándar inicial
x_0, y_0 = 4, 4  # Coordenadas iniciales

# MODOS NORMALES
n_x, n_y = 6 * pi, 6 * pi  # Modos si es caja infinita y sus estados

caso = "free"
psi_0 = gaussian_package(x_0, y_0, k_x, k_y, nL - 2, dx, sigma_0)
#psi_0 = modos_normales(n_x, n_y, l, nL - 2, dx)


#############
# POTENCIAL #
#############

def V(n, m):
    potencial = 0
    x = (n - nL // 2) * dx  # x normalizada
    y = (m - nL // 2) * dx
    # potencial += coloumb(x, y)
    # potencial += double_slit(4, x, y, 2, dx)
    return potencial


def alpha(k):
    m = 1 + k // (nL - 2)
    n = 1 + k % (nL - 2)
    return 1 + 4 * r + 1j * dt * V(n, m) / 2


def beta(k):
    m = 1 + k // (nL - 2)
    n = 1 + k % (nL - 2)
    return 1 - 4 * r - 1j * dt * V(n, m) / 2


def A_mat(L=nL):
    A = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    for k in range(1, (L - 2) ** 2 - 1):
        A[k][k] = alpha(k)
        A[k][k - 1] = -r
        A[k][k + 1] = -r
        if k + (L - 2) < (L - 2) ** 2:
            A[k][k + (L - 2)] = -r
        if k - (L - 2) >= 0:
            A[k][k - (L - 2)] = -r
    A[0][0] = alpha(0)
    A[-1][-1] = alpha(L - 2)
    A[0][1] = A[-1][-2] = -r
    return A


def B_mat(L=nL):
    B = np.zeros(((L - 2) ** 2, (L - 2) ** 2), complex)
    for k in range(1, (L - 2) ** 2 - 1):
        B[k][k] = beta(k)
        B[k][k - 1] = r
        B[k][k + 1] = r
        if k + (L - 2) < (L - 2) ** 2:
            B[k][k + (L - 2)] = r
        if k - (L - 2) >= 0:
            B[k][k - (L - 2)] = r
    B[0][0] = beta(0)
    B[-1][-1] = beta(L - 2)
    B[0][1] = B[-1][-2] = r
    return B


def open_boundary_conditions(psi_array, len_steps, obc_switch):
    """
    :param psi_array: Array containing wave function
    :param len_steps: Length steps.
    :param obc_switch: (boolean) Uses open boundary conditions or not.
    """
    if obc_switch:
        for k in range(psi_array.shape[0]):
            n = 1 + k // (len_steps - 2)
            m = 1 + k % (len_steps - 2)
            if n == 1 or n == ((len_steps - 2) ** 2 - 1):
                psi_array[k] += r * psi_array[k]
            if m == 1 or m == ((len_steps - 2) ** 2 - 1):
                psi_array[k] += r * psi_array[k]

    return psi_array


def resolve():
    """
    :return: [0] Matrix with probabilities. [1] Next step wave
    """
    array_TS = np.dot(A_inv, open_boundary_conditions(np.dot(B, current_psi), nL, obc))
    probs_mat = np.zeros((nL, nL))
    for k in range((nL - 2) ** 2):
        probs_mat[1 + k // (nL - 2)][1 + k % (nL - 2)] = np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2)
    return probs_mat, array_TS


if centrado:
    lower_lim, upper_lim = -l, l
else:
    lower_lim, upper_lim = 0, 2 * l

with open(f"frames/{caso}/data.txt", "w") as f:
    f.write(
        f"{caso}\t{time.ctime()}\nk_x={k_x / pi}pi\tk_y={k_y / pi}pi\ndx={dx}\t dt={dt}\t ratio={ratio}\n"
        f"timesteps={nT}\nspatial steps={nL}\t ({lower_lim},{upper_lim})\nOpen boundary conditions:{obc}")

current_psi = psi_0.flatten("C")
heatmap(prob(psi_0), lower_lim, upper_lim).savefig(f"frames/{caso}/psi_0.jpg")
A_inv = inv(A_mat(nL))
B = B_mat(nL)
print(f"Initialization: OK")

for ts in range(nT):
    probs, next_psi = resolve()
    heatmap(probs, lower_lim, upper_lim).savefig(f"frames/{caso}/psi_{ts + 1}.jpg")
    print(f"{ts + 1}/{nT}")
    current_psi = next_psi

"""def update(ts):
    probs, array_TS = resolve()
    Z = probs
    plt.pcolormesh(X, Y, Z)  # , vmin=0, vmax=0.65
    array_T = array_TS
    print(f"{ts + 1}/{nT}")



fig = plt.figure(figsize=(16, 9))
X, Y = np.meshgrid(np.linspace(-l, l, nL), np.linspace(-l, l, nL))
plt.ylabel("$y$")
plt.xlabel("$x$")
plt.axis('scaled')
fig = plt.figure(figsize=(16, 9))


def probs_inicial(array_TS):
    probs = np.zeros((nL, nL))
    for k in range((nL - 2) ** 2):
        probs[1 + k // (nL - 2)][1 + k % (nL - 2)] = np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2)
    return probs


plt.pcolormesh(X, Y, probs_inicial(array_T))  # , vmin=0, vmax=0.65
plt.ylabel("$y$")
plt.xlabel("$x$")
plt.axis('scaled')

cbar = plt.colorbar()
cbar.set_label("$\lvert\Psi\\rvert^2$")
anim = FuncAnimation(fig, update, frames=nT)
anim.save('double_slit.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
"""
