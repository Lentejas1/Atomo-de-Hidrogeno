import numpy as np

def prob(psi):
    L = np.shape(psi)[0]
    p = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            p[i][j] = np.sqrt(np.real(psi[i][j])**2+np.imag(psi[i][j])**2)
    return p
