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

nL = 100  # Pasos espaciales NO CAMBIAR de 100 (si se ponen más para un pulso, va hacia atrás idk why)
nT = 200  # Pasos temporales
l = 5  # Borde del mallado (va de -l a l o de 0 a 2l según centrado, True/False respectivamente)
dx = (2 * l) / (nL - 1)  # DeltaX
ratio = 0.49
dt = ratio * dx ** 2
r = 1j * dt / (2 * dx ** 2)
obc = True
centrado = True #-> Falso si son modos

if centrado:
    lower_lim, upper_lim = -l, l
else:
    lower_lim, upper_lim = 0, 2 * l

x = np.linspace(lower_lim, upper_lim + dx, nL-2, dtype=complex)
y = np.linspace(lower_lim, upper_lim + dx, nL-2, dtype=complex)
X, Y = np.meshgrid(x, y)
#####################################
# PARÁMETROS DE LAS FUNCIÓN DE ONDA #
#####################################

# PULSO
k_x, k_y = -20 * pi, 20 * pi  # Número de onda inicial (p/hbar)   E=(k_x^2+k_y^2)/2
sigma_0 = 1  # Desviación estándar inicial
x_0, y_0 = 2, 3  # Coordenadas iniciales

# MODOS NORMALES
n_x, n_y = 6 * pi, 6 * pi  # Modos si es caja infinita y sus estados

caso = "free"
psi_0 = gaussian_package(X, Y, x_0, y_0, k_x, k_y, lower_lim, upper_lim, nL - 2, dx, sigma_0)
#psi_0 = modos_normales(n_x, n_y, lower_lim, upper_lim, l, nL - 2, dx)
#psi_0 = onda_plana(1, x_0, y_0, k_x, k_y, lower_lim, upper_lim, nL - 2, dx)
#psi_0 = hydrogen_bounded_state(X, Y, x_0, y_0, lower_lim, upper_lim, nL - 2, dx)
current_psi = psi_0.flatten("C")


#############
# POTENCIAL #
#############

def V(n, m):
    potencial = 0
    x = (n - nL // 2) * dx  # x normalizada
    y = (m - nL // 2) * dx
    #potencial += coloumb(x, y)
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


def open_boundary_conditions(psi_array, nL, obc_switch):
    """
    :param psi_array: Array containing wave function
    :param nL: Length steps.
    :param obc_switch: (boolean) Uses open boundary conditions or not.
    """
    for k in range(len(psi_array)):
        n = 1 + k // sqrt(psi_array.shape[0])
        m = 1 + k % sqrt(psi_array.shape[0])
        if n == 1:
            if obc_switch:
                psi_array[k] = psi_array[k + 1]
            else:
                psi_array[k] = 0
        elif m == 1 and n != (nL-2):
            if obc_switch:
                psi_array[k] = psi_array[int(k + (nL - 2))]
            else:
                psi_array[k] = 0
        elif n == (nL-2):
            if obc_switch:
                psi_array[k] = psi_array[k - 1]
            else:
                psi_array[k] = 0
        elif m == sqrt(psi_array.shape[0]):
            if obc_switch:
                psi_array[k] = psi_array[int(k - (nL - 2))]
            else:
                psi_array[k] = 0

    return psi_array


def resolve(current_psi=current_psi):
    """
    :return: [0] Matrix with probabilities. [1] Next step wave
    """
    current_psi = open_boundary_conditions(current_psi, nL, obc)
    array_TS = np.dot(A_inv, (np.dot(B, current_psi)))
    probs_mat = np.zeros((nL, nL))
    for k in range((nL - 2) ** 2):
        probs_mat[1 + k // (nL - 2)][1 + k % (nL - 2)] = np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2)
    return probs_mat, array_TS




with open(f"frames/{caso}/data.txt", "w") as f:
    f.write(
        f"{caso}\t{time.ctime()}\nk_x={k_x / pi}pi\tk_y={k_y / pi}pi\ndx={dx}\t dt={dt}\t ratio={ratio}\n"
        f"timesteps={nT}\nspatial steps={nL}\t ({lower_lim},{upper_lim})\nOpen boundary conditions:{obc}")

print(psi_0.shape)
heatmap(prob(psi_0), lower_lim, upper_lim).savefig(f"frames/{caso}/psi_0.jpg")
A_inv = np.linalg.inv(A_mat(nL))
B = B_mat(nL)
print(f"Initialization: OK")

for ts in range(nT):
    probs, next_psi = resolve(current_psi)
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