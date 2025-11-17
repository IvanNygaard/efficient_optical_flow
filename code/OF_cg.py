# Imports
import scipy.sparse as sp
import numpy as np
from utils import norm, F


# CG-function.
def OF_cg(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    reg: float,
    rhsu: np.ndarray,
    rhsv: np.ndarray,
    tol=1.0e-8,
    maxit=4000,
    level: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    The CG method for the optical flow problem.

    Args:
    ---
    u0 - initial guess for u
    v0 - initial guess for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter lambda
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    tol - relative residual tolerance
    maxit - maximum number of iterations

    Returns:
    ---
     tuple[np.ndarray, np.ndarray, list[float]]
        Numerical solution for u, v, and list of residuals
    """
    # Dimensions, step-size, constants for diagonal (k1) and off-diagonal (k2) elements and cross derivatives
    n, m = Ix.shape
    # n = n - 2
    # m = m - 2
    h = float(2**level)
    k1 = (4 * reg) / (h * h)
    k2 = -reg / (h * h)
    res_ratios = []

    # Note: Implicitly imposing Dirichlet B.C. by only acting on interior nodes (n-2, m-2) and settung u0 = v0 = 0
    # Ix = Ix[1:-1, 1:-1]
    # Iy = Iy[1:-1, 1:-1]
    Ixy = Ix * Iy
    # rhsu = rhsu[1:-1, 1:-1]
    # rhsv = rhsv[1:-1, 1:-1]

    # Initialize x and b with row-wise numbering
    u = u0
    v = v0
    x = np.hstack((u.ravel(order="C"), v.ravel(order="C")))
    b = np.hstack((rhsu.ravel(order="C"), rhsv.ravel(order="C")))

    # 1D Laplacian
    Lx = sp.diags([k2, k1, k2], [-1, 0, 1], shape=(m, m))
    Ly = sp.diags([k2, 0, k2], [-1, 0, 1], shape=(n, n))

    # 2D Laplacian from Kroneckersum, https://stackoverflow.com/questions/34895970/buildin-a-sparse-2d-laplacian-matrix-using-scipy-modules
    ex = sp.eye(n)
    ey = sp.eye(m)
    L = sp.kron(ex, Lx) + sp.kron(Ly, ey)

    A_11 = sp.diags(Ix.ravel() ** 2) + L
    A_22 = sp.diags(Iy.ravel() ** 2) + L
    A_12 = sp.diags(Ixy.ravel())
    A_21 = A_12.copy()

    A = sp.block_array([[A_11, A_12], [A_21, A_22]]).tocsr()

    # plt.spy(A_11, markersize=0.8, color = "black")
    # plt.show()
    # plt.spy(A, markersize=0.8, color = "black")
    # plt.show()

    # CG-method (p. 190 Saad)
    iter = 0
    r0 = b - A @ x
    r_old = r0.copy()
    p = r_old.copy()


    while True:
        alpha = (r_old.T @ r_old) / ((A @ p).T @ p)
        x += alpha * p
        r_new = r_old - alpha * (A @ p)
        beta = (r_new.T @ r_new) / (r_old.T @ r_old)
        p = r_new + beta * p
        r_old = r_new
        iter += 1
        res_ratios.append(np.linalg.norm(r_old) / np.linalg.norm(r0))

        if (np.linalg.norm(r_old) / np.linalg.norm(r0) < tol) or (iter >= maxit):
            break

    # x, info = sp.linalg.cg(A, b)

    # Unpack solution
    u = x[: (n * m)].reshape((n, m))
    v = x[(n * m) :].reshape((n, m))
    # print(np.max(Ix**2), 4*reg/(h*h))

    return u, v, res_ratios


def cg(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    tol=1.0e-8,
    maxitr=4000,
    h: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The CG method for the optical flow problem.

    Args:
    ---
    u0 : np.ndarray
        Initial guess for u
    v0 : np.ndarray
        Initial guess for v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame
    lam : float
        Regularisation parameter lambda
    rhs_u : np.ndarray
        Right-hand side in the equation for u
    rhs_v : np.ndarray
        Right-hand side in the equation for v
    tol : np.ndarray, default=1e-8
        Relative residual tolerance
    maxitr : np.ndarray, default=4000
        Maximum number of iterations
    h : float, default=1
        Steplength

    Returns:
    ---
     tuple[np.ndarray, np.ndarray, list[float]]
        Numerical solution for u, v, and list of residuals
    """
    # Residuals (For experiments)
    relative_residuals_arr = np.zeros(maxitr)

    # Initialize
    Fu, Fv = F(u0, v0, Ix, Iy, lam, h)

    # r
    ru = np.copy(rhs_u - Fu)
    rv = np.copy(rhs_v - Fv)
    # x
    u = np.copy(u0)
    v = np.copy(v0)
    # p
    pu = np.copy(ru)
    pv = np.copy(rv)

    # Calculate the norm of r
    rr0 = norm(ru, rv) ** 2
    rk1_rk1 = rr0

    assert ru.shape == rv.shape

    it = 0
    while it < maxitr:
        # Calculate alpha
        rk_rk = rk1_rk1
        Fpu, Fpv = F(pu, pv, Ix, Iy, lam, h)
        pAp = np.sum(Fpu * pu) + np.sum(Fpv * pv)

        alpha = rk_rk / pAp

        u = np.copy(u + alpha * pu)
        v = np.copy(v + alpha * pv)

        ru = np.copy(ru - alpha * Fpu)
        rv = np.copy(rv - alpha * Fpv)

        # Break condition
        rk1_rk1 = norm(ru, rv) ** 2
        # 'tol' raised to power of 2 as we are dealing with norm squared

        # Store the residuals
        relative_residuals_arr[it] = np.sqrt(rk1_rk1) / np.sqrt(rr0)

        if rk1_rk1 / rr0 < tol**2:
            # So we index the residuals correctly afterwards
            it += 1
            break
        # TESTING
        # Fu, Fv = F(u, v, Ix, Iy, lam, h)
        # print("Est. Residual (Norm): ", np.sqrt(rk1_rk1))
        # print("Residual (Norm): ", norm(rhs_u - Fu, rhs_v - Fv))

        beta = rk1_rk1 / rk_rk

        pu = np.copy(ru + beta * pu)
        pv = np.copy(rv + beta * pv)

        # Increase iteration counter
        it += 1

    #print("Itr: ", it)
    return u, v, relative_residuals_arr[:it]
