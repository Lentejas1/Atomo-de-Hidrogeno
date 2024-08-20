import torch
import time
from initial_psi_gpu import *
from plots import *
from functions_gpu import *
import scienceplots
plt.style.use("science")

##################################
# CONDICIONES INICIALES TÉCNICAS #
##################################

nL = 125  # Pasos espaciales
ghost = 0
nT = 160  # Pasos temporales
l = 4  # Borde del mallado (va de -l a l o de 0 a 2l según centrado, True/False respectivamente)
dx = (2 * l) / (nL - 1)  # DeltaX
ratio = 0.25
dt = ratio * dx ** 2
r = 1j * dt / (2 * dx ** 2)
obc = False  # True no va
centrado = True  # -> Falso si son modos

# Set device to ROCm GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if centrado:
    lower_lim, upper_lim = -l, l
else:
    lower_lim, upper_lim = 0, 2 * l

x = torch.linspace(lower_lim, upper_lim, nL, dtype=torch.float, device=device)
y = torch.linspace(lower_lim, upper_lim, nL, dtype=torch.float, device=device)
X, Y = torch.meshgrid(x, y)

#####################################
# PARÁMETROS DE LAS FUNCIÓN DE ONDA #
#####################################

# PULSO
k_x, k_y = 40 * pi, 0 * pi  # Número de onda inicial (p/hbar)
sigma_0 = 0.5    # Desviación estándar inicial
x_0, y_0 = -2, 0  # Coordenadas iniciales

# MODOS NORMALES
n_x, n_y = 6 * pi, 6 * pi  # Modos si es caja infinita y sus estados

caso = "tunnelling"
psi_0 = gaussian_package(X, Y, x_0, y_0, k_x, k_y, lower_lim, upper_lim, nL, dx, sigma_0)
current_psi = psi_0.flatten().to(device)

#############
# POTENCIAL #
#############

def V_maker(height):
    xs = torch.linspace(lower_lim, upper_lim, nL, device=device)
    ys = torch.linspace(lower_lim, upper_lim, nL, device=device)
    potencial = torch.zeros((nL, nL), device=device)
    for i in range(nL):
        for j in range(nL):
            potencial[i, j] += tunnelling(0, ys[i], height, dx)
    return potencial

V = V_maker(750)

# Guardar el potencial en un archivo CSV
with open("V.csv", "w") as f:
    for i in range(nL):
        for j in range(nL):
            f.write(f"{V[i, j].item()};")
        f.write("\n")

def alpha(k):
    m = k // nL
    n = k % nL
    return 1 + 4 * r + 1j * dt * V[n][m] / 2

def beta(k):
    m = k // nL
    n = k % nL
    return 1 - 4 * r - 1j * dt * V[n][m] / 2

def A_mat(L=nL):
    A = torch.zeros((nL ** 2, nL ** 2), dtype=torch.complex128, device=device)
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
    B = torch.zeros((nL ** 2, nL ** 2), dtype=torch.complex128, device=device)
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
    mat_psis = psi_array.view(nL, nL)
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
    return mat_psis.flatten()

def resolve(current_psi=current_psi):
    array_TS = torch.matmul(A_inv, torch.matmul(B, current_psi))
    probs_mat = torch.zeros((nL, nL), dtype=torch.float, device=device)
    for k in range(nL ** 2):
        probs_mat[k % nL][k // nL] = float(torch.sqrt(torch.real(array_TS[k]) ** 2 + torch.imag(array_TS[k]) ** 2))
    return probs_mat, open_boundary_conditions(array_TS, nL, obc)

# Guardar la configuración inicial
with open(f"frames/{caso}/data.txt", "w") as f:
    f.write(
        f"{caso}\t{time.ctime()}\nk_x={k_x / pi}pi\tk_y={k_y / pi}pi\ndx={dx}\t dt={dt}\t ratio={ratio}\n"
        f"timesteps={nT}\nspatial steps={nL}\t ({lower_lim},{upper_lim})\nOpen boundary conditions:{obc}")

# Inversión de matriz usando torch
A_inv = torch.linalg.inv(A_mat(nL))
B = B_mat(nL)
print(f"Initialization: OK")

p_0 = sum(sum(prob(nL, current_psi))).item() * dx ** 2
print(p_0)
error = [0]

# Loop de simulación
for ts in range(nT):
    probs, next_psi = resolve(current_psi)
    heatmap(X.cpu(), Y.cpu(), probs.cpu(), dx, ts).savefig(f"frames/{caso}/psi_{ts + 1}.jpg", dpi=300)
    print(f"{ts + 1}/{nT}")
    error.append((sum(sum(probs)).item() * dx ** 2 - p_0) / p_0)
    current_psi = next_psi

plt.figure(figsize=(8, 2))
plt.plot(np.arange(0, nT + 1, 1), error, color="red")
plt.axhline(0, ls="--", color="black", alpha=0.5)
plt.xlabel("$n$")
plt.ylabel(r"$E\sim\dfrac{p - p_0}{p_0}$")
plt.savefig(f"frames/{caso}/error_final.jpg", dpi=300)
