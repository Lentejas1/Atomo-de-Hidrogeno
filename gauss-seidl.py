### Open Boundary conditions
for xs in range(L):
    psis_ts[xs][0] = (r * (psis_ts[xs][0] + psis_ts[xs - 1][0] + psis_ts[xs][0] + psis_ts[xs][0] +
                           psis_t[xs][0] + psis_t[xs - 1][0] + psis_t[xs][0] + psis_t[xs][0] -
                           4 * psis_t[xs][0] + p * psis_t[xs][0]) - psis_t[xs][0]) / (1 + 4 * r - p * r)
    psis_ts[xs][L - 1] = (r * (
            psis_ts[xs][L - 1] + psis_ts[xs - 1][L - 1] + psis_ts[xs][L - 1] + psis_ts[xs][L - 1] +
            psis_t[xs][L - 1] + psis_t[xs - 1][L - 1] + psis_t[xs][L - 1] + psis_t[xs][L - 1] -
            4 * psis_t[xs][L - 1] + p * psis_t[xs][L - 1]) - psis_t[xs][L - 1]) / (1 + 4 * r - p * r)
for ys in range(L):
    psis_ts[0][ys] = (r * (
            psis_ts[0][ys] + psis_ts[0][ys] + psis_ts[0][ys] + psis_ts[0][ys - 1] +
            psis_t[0][ys] + psis_t[0][ys] + psis_t[0][ys] + psis_t[0][ys - 1] -
            4 * psis_t[0][ys] + p * psis_t[0][ys]) - psis_t[0][ys]) / (1 + 4 * r - p * r)
    psis_ts[L - 1][ys] = (r * (
            psis_ts[L - 1][ys] + psis_ts[L - 1][ys] + psis_ts[L - 1][ys] + psis_ts[L - 1][ys - 1] +
            psis_t[L - 1][ys] + psis_t[L - 1][ys] + psis_t[L - 1][ys] + psis_t[L - 1][ys - 1] -
            4 * psis_t[L - 1][ys] + p * psis_t[L - 1][ys]) - psis_t[L - 1][ys]) / (1 + 4 * r - p * r)
# GAUSS-SEIDEL MOD
for i in range(gs_iterations):
    for xs in range(1, L - 1):
        for ys in range(1, L - 1):
            n = xs - L // 2
            m = ys - L // 2
            # p = dx / (np.sqrt(n ** 2 + m ** 2))
            psis_ts[xs][ys] = (r * (
                    psis_ts[xs + 1][ys] + psis_ts[xs - 1][ys] + psis_ts[xs][ys + 1] + psis_ts[xs][ys - 1] +
                    psis_t[xs + 1][ys] + psis_t[xs - 1][ys] + psis_t[xs][ys + 1] + psis_t[xs][ys - 1] -
                    4 * psis_t[xs][ys] + p * psis_t[xs][ys]) - psis_t[xs][ys]) / (1 + 4 * r - p * r)
heatmap(prob(psis_ts), l).savefig(f"frames/free/psi_fkx5_{ts + 1}.jpg")
print(ts + 1)
psis_t = psis_ts
