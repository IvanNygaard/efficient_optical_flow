import numpy as np
from utils import norm, F
from multigrid import V_cycle, smoothing


def pcg(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    s1=1,
    s2=1,
    max_level=4,
    tol=1.0e-8,
    maxitr=4000,
    h: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    # Initialize
    Fu, Fv = F(u0, v0, Ix, Iy, lam, h)
    ru = np.copy(rhs_u - Fu)
    rv = np.copy(rhs_v - Fv)

    # Residualts (for experiments/plotting)
    relative_residuals = []

    #print("ru: ", ru)
    #print("rv: ", rv)
    u = np.copy(u0)
    v = np.copy(v0)

    # Calculate the norm
    r0 = norm(ru, rv)
    r = r0  # To be updated in the iterations
    #print("r0: ", r0)

    assert ru.shape == rv.shape

    # M*z0 = r0
    # zu, zv = cg(
    #     np.zeros_like(ru), np.zeros_like(rv), Ix, Iy, lam, ru, rv, 1e-8, 3, h
    # )
    zu, zv = V_cycle(
        np.zeros_like(ru),
        np.zeros_like(rv),
        Ix,
        Iy,
        lam,
        ru,
        rv,
        s1,
        s2,
        level=0,
        max_level=max_level,
    )
    pu = np.copy(zu)
    pv = np.copy(zv)

    # Define it here so the iterations work
    rk1_zk1 = np.sum(ru * zu) + np.sum(rv * zv)

    it = 0
    while it < maxitr:
        it += 1

        # Calculate alpha
        # r.T @ z
        rk_zk = rk1_zk1
        Fpu, Fpv = F(pu, pv, Ix, Iy, lam, h)
        pAp = np.sum(Fpu * pu) + np.sum(Fpv * pv)

        alpha = rk_zk / pAp

        u = np.copy(u + alpha * pu)
        v = np.copy(v + alpha * pv)

        ru = np.copy(ru - alpha * Fpu)
        rv = np.copy(rv - alpha * Fpv)

        # Break condition
        r = norm(ru, rv)
        #print("Residual: ", r)
        #print("Rel Residual: ", r / r0)

        if r / r0 < tol:
            it += 1
            break

        relative_residuals.append(r / r0)

        # Solve Mz=r
        zu, zv = V_cycle(
            np.zeros_like(zu), np.zeros_like(zv), Ix, Iy, lam, ru, rv, s1, s2, level=0, max_level=max_level
        )
        # zu, zv = cg(zu, zv, Ix, Iy, lam, ru, rv, 1e-8, 3, h)

        assert ru.shape == zu.shape
        rk1_zk1 = np.sum(ru * zu) + np.sum(rv * zv)
        beta = rk1_zk1 / rk_zk

        pu = np.copy(zu + beta * pu)
        pv = np.copy(zv + beta * pv)

    return u, v, relative_residuals
