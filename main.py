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

nL = 125  # Pasos espaciales NO CAMBIAR de 100 (si se ponen más para un pulso, va hacia atrás idk why)
ghost = 0
nT = 400  # Pasos temporales
l = 4  # Borde del mallado (va de -l a l o de 0 a 2l según centrado, True/False respectivamente)
dx = (2 * l) / (nL - 1)  # DeltaX
ratio = 0.25
dt = ratio * dx ** 2
r = 1j * dt / (2 * dx ** 2)
obc = False
centrado = True  # -> Falso si son modos

if centrado:
    lower_lim, upper_lim = -l, l
else:
    lower_lim, upper_lim = 0, 2 * l

x = np.linspace(lower_lim, upper_lim, nL, dtype=float)
y = np.linspace(lower_lim, upper_lim, nL, dtype=float)
X, Y = np.meshgrid(x, y)
#####################################
# PARÁMETROS DE LAS FUNCIÓN DE ONDA #
#####################################

# PULSO
k_x, k_y = 40 * pi, 0 * pi  # Número de onda inicial (p/hbar)   E=(k_x^2+k_y^2)/2
sigma_0 = 0.5  # Desviación estándar inicial
x_0, y_0 = -2, 0  # Coordenadas iniciales

# MODOS NORMALES
n_x, n_y = 6 * pi, 6 * pi  # Modos si es caja infinita y sus estados

caso = "tunnelling"
psi_0 = gaussian_package(X, Y, x_0, y_0, k_x, k_y, lower_lim, upper_lim, nL, dx, sigma_0)
# psi_0 = modos_normales(n_x, n_y, lower_lim, upper_lim, l, nL - 2, dx)
# psi_0 = onda_plana(1, x_0, y_0, k_x, k_y, lower_lim, upper_lim, nL - 2, dx)
# psi_0 = hydrogen_bounded_state(X, Y, 1)
current_psi = psi_0.flatten("C")


#############
# POTENCIAL #
#############

def V_maker():
    xs = np.linspace(lower_lim, upper_lim, nL, dtype=float)
    ys = np.linspace(lower_lim, upper_lim, nL, dtype=float)
    potencial = np.zeros((nL, nL))
    for i in range(nL):
        for j in range(nL):
            # potencial[i, j] += coloumb(xs[i], ys[i])
            #potencial[i, j] += double_slit(1, xs[i], ys[j], dx * 10, dx)
            potencial[i, j] += tunnelling(2, ys[i], 100, dx)

    return potencial


V = V_maker()

with open("V.csv", "w") as f:
    for i in range(nL):
        for j in range(nL):
            f.write(f"{V[i, j]};")
        f.write("\n")


def alpha(k):
    m = k // nL
    n = k % nL
    return 1 + 4 * r + 1j * dt * V[n, m] / 2


def beta(k):
    m = k // nL
    n = k % nL
    return 1 - 4 * r - 1j * dt * V[n, m] / 2


def A_mat(L=nL):
    A = np.zeros((nL ** 2, nL ** 2), complex)
    for k in range(1, nL ** 2 - 1):
        A[k][k] = alpha(k)
        A[k][k - 1] = -r
        A[k][k + 1] = -r
        if k + nL < nL ** 2:
            A[k][k + nL] = -r
        if k - nL >= 0:
            A[k][k - nL] = -r
    A[0][0] = alpha(0)
    A[-1][-1] = alpha(nL)
    A[0][1] = A[-1][-2] = -r
    return A


def B_mat(L=nL):
    B = np.zeros((nL ** 2, nL ** 2), complex)
    for k in range(1, nL ** 2 - 1):
        B[k][k] = beta(k)
        B[k][k - 1] = r
        B[k][k + 1] = r
        if k + nL < nL ** 2:
            B[k][k + nL] = r
        if k - (L - 2) >= 0:
            B[k][k - nL] = r
    B[0][0] = beta(0)
    B[-1][-1] = beta(nL)
    B[0][1] = B[-1][-2] = r
    return B


def open_boundary_conditions(psi_array, nL, obc):
    """
    :param psi_array: Array containing wave function
    :param nL: Length steps.
    :param obc_switch: (boolean) Uses open boundary conditions or not.
    """
    mat_psis = psi_array.reshape(nL, nL)
    if obc:
        for i in range(nL):
            for j in range(nL):
                if i < ghost:
                    mat_psis[i, j] = mat_psis[ghost, j]
                elif i > nL - ghost:
                    mat_psis[i, j] = mat_psis[-ghost, j]
                if j < ghost:
                    mat_psis[i, j] = mat_psis[i, ghost]
                elif j > nL - ghost:
                    mat_psis[i, j] = mat_psis[i, -ghost]

    else:
        mat_psis[0, :] = mat_psis[-1, :] = mat_psis[:, 0] = mat_psis[:, -1] = 0
    psi_array = psi_array.flatten("C")
    return psi_array


def resolve(current_psi=current_psi):
    """
    :return: [0] Matrix with probabilities. [1] Next step wave
    """
    array_TS = np.dot(A_inv, (np.dot(B, current_psi)))
    probs_mat = np.zeros((nL, nL), float)
    for k in range(nL ** 2):
        probs_mat[k % nL][k // nL] = float(np.sqrt(np.real(array_TS[k]) ** 2 + np.imag(array_TS[k]) ** 2))
    return probs_mat, open_boundary_conditions(array_TS, nL, obc)


with open(f"frames/{caso}/data.txt", "w") as f:
    f.write(
        f"{caso}\t{time.ctime()}\nk_x={k_x / pi}pi\tk_y={k_y / pi}pi\ndx={dx}\t dt={dt}\t ratio={ratio}\n"
        f"timesteps={nT}\nspatial steps={nL}\t ({lower_lim},{upper_lim})\nOpen boundary conditions:{obc}")

# heatmap(nL, X, Y, prob(nL, current_psi), lower_lim, upper_lim).savefig(f"frames/{caso}/psi_0.jpg")
A_inv = np.linalg.inv(A_mat(nL))
B = B_mat(nL)
print(f"Initialization: OK")

p_0 = sum(sum(prob(nL, current_psi))) * dx ** 2
print(p_0)
error = [0]

for ts in range(nT):
    probs, next_psi = resolve(current_psi)
    heatmap(X, Y, probs, V).savefig(f"frames/{caso}/psi_{ts + 1}.jpg")
    print(f"{ts + 1}/{nT}")
    error.append((sum(sum(probs)) * dx ** 2 - p_0) / p_0)
    current_psi = next_psi


print(sum(psi_0.flatten("C") - current_psi) / p_0)

"""fig = plt.figure(figsize=(16, 9))


def update(ts):
    probs, current_psi = resolve()
    Z = probs
    plt.pcolormesh(X, Y, Z)  # , vmin=0, vmax=0.65
    print(f"{ts + 1}/{nT}")


fig = plt.figure(figsize=(16, 9))
X, Y = np.meshgrid(np.linspace(lower_lim, upper_lim, nL), np.linspace(lower_lim, upper_lim, nL))
plt.ylabel("$y$ ($a_0$)")
plt.xlabel("$x$ ($a_0$)")
plt.axis('scaled')


def probs_inicial(current_psi):
    probs = np.zeros((nL, nL))
    for k in range(nL ** 2):
        probs[k % nL][k // nL] = np.sqrt(np.real(current_psi[k]) ** 2 + np.imag(current_psi[k]) ** 2)
    return probs


plt.pcolormesh(X, Y, probs_inicial(current_psi), vmin=0, vmax=0.33)
plt.ylabel("$y$ ($a_0$)")
plt.xlabel("$x$ ($a_0$)")
plt.axis('scaled')
cbar = plt.colorbar()
cbar.set_label("$\lvert\Psi\\rvert^2$")
anim = FuncAnimation(fig, update, frames=nT)
anim.save('double_slit.gif', fps=24) #, extra_args=['-vcodec', 'libx264']"""