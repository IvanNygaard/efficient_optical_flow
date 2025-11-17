import numpy as np
from OF_cg import cg, OF_cg
from utils import F


def V_cycle(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    s1: int,
    s2: int,
    level: int,
    max_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    V-cycle for the optical flow problem

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
        Penalty term
    rhs_u : np.ndarray
        RHS in eq for u
    rhs_v : np.ndarray
        RHS in eq for v
    s1 : int
        Number of pre-smoothings
    s2 : int
        Number of post-smoothings
    level : int
        Current level
    max_level : int
        Total number of levels

    Returns:
    ---
    tuple[np.ndarray, np.ndarray, list[float]]
        Numerical solution for u, v
    """
    # Stepsize
    h = float(2**level)

    u0, v0 = np.copy(u0), np.copy(v0)
    u, v = smoothing(u0, v0, Ix, Iy, lam, rhs_u, rhs_v, s1, h)
    ru_h, rv_h = residual(u, v, Ix, Iy, lam, rhs_u, rhs_v, h)
    ru_2h, rv_2h, Ix2h, Iy2h = restriction(ru_h, rv_h, Ix, Iy)
    if level == max_level - 1:
        # Ignore the residuals here
        eu_2h, ev_2h, _ = cg(
            np.zeros_like(ru_2h),
            np.zeros_like(rv_2h),
            Ix2h,
            Iy2h,
            lam,
            ru_2h,
            rv_2h,
            1e-8,
            1000,
            h,  # Why h and not 2h?
        )
    else:
        eu_2h, ev_2h = V_cycle(
            np.zeros_like(ru_2h),
            np.zeros_like(rv_2h),
            Ix2h,
            Iy2h,
            lam,
            ru_2h,
            rv_2h,
            s1,
            s2,
            level + 1,
            max_level,
        )

    eu_h, ev_h = prolongation(eu_2h, ev_2h)
    u = u + eu_h
    v = v + ev_h
    u, v = smoothing(u, v, Ix, Iy, lam, rhs_u, rhs_v, s2, h)

    return u, v


def smoothing(
    u: np.ndarray,
    v: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhsu: np.ndarray,
    rhsv: np.ndarray,
    iterations: int,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smoothing using Red-Black Gauss-Seidel

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
        Penalty term
    rhsu : np.ndarray
        RHS in eq for u
    rhsv : np.ndarray
        RHS in eq for v
    s1 : int
        Number of pre-smoothings
    h : int
        Stepsize corresponding to the level


    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Smoothed u, v
    """
    assert u.shape == v.shape

    # Pad u,v with zeros around (Dirichlet BC)
    u_pad = np.pad(u, 1)
    v_pad = np.pad(v, 1)

    n, m = v_pad.shape

    for _ in range(iterations):
        # These two are independent of each other
        black_update(u_pad, v_pad, Ix, Iy, lam, rhsu, h)
        red_update(v_pad, u_pad, Iy, Ix, lam, rhsv, h)

        # These last two depends on the first two
        black_update(v_pad, u_pad, Iy, Ix, lam, rhsv, h)
        red_update(u_pad, v_pad, Ix, Iy, lam, rhsu, h)

        # For symmetric GS

        red_update(u_pad, v_pad, Ix, Iy, lam, rhsu, h)
        black_update(v_pad, u_pad, Iy, Ix, lam, rhsv, h)

        red_update(v_pad, u_pad, Iy, Ix, lam, rhsv, h)
        black_update(u_pad, v_pad, Ix, Iy, lam, rhsu, h)

    return u_pad[1 : n - 1, 1 : m - 1], v_pad[1 : n - 1, 1 : m - 1]


def black_update(
    w: np.ndarray,
    p: np.ndarray,
    Iw: np.ndarray,
    Ip: np.ndarray,
    lam: float,
    rhs: np.ndarray,
    h: float,
) -> None:
    """
    Black update using Red-Black Gauss-Seidel.
    Note: This function mutates w

    Args:
    ---
    w : np.ndarray
        The padded array to update
    p : np.ndarray
        The padded other array
    Iw : np.ndarray
        If w=u -> Iw=Ix
        If w=v -> Iw=Iy
    Ip : np.ndarray
        If w=u -> Ip=Iy
        If w=v -> Ip=Ix
    lam : float
        Penalty term
    rhs : np.ndarray
        RHS in eq
    h : int
        Stepsize corresponding to the level

    Returns:
    ---
    None
    """
    n, m = w.shape
    k, d = Iw.shape

    # Lower left update
    w[2 : n - 1 : 2, 1 : m - 1 : 2] = (
        rhs[1:k:2, 0:d:2]
        - Ip[1:k:2, 0:d:2] * Iw[1:k:2, 0:d:2] * p[2 : n - 1 : 2, 1 : m - 1 : 2]
        + lam
        / h**2
        * (
            # Left
            w[2 : n - 1 : 2, 0 : m - 2 : 2]
            # Right
            + w[2 : n - 1 : 2, 2:m:2]
            # Up
            + w[1 : n - 2 : 2, 1 : m - 1 : 2]
            # Down
            + w[3:n:2, 1 : m - 1 : 2]
        )
    ) / (Iw[1:k:2, 0:d:2] ** 2 + 4 * lam / h**2)

    # Upper right update
    w[1 : n - 1 : 2, 2 : m - 1 : 2] = (
        rhs[0:k:2, 1:d:2]
        - Ip[0:k:2, 1:d:2] * Iw[0:k:2, 1:d:2] * p[1 : n - 1 : 2, 2 : m - 1 : 2]
        + lam
        / h**2
        * (
            # Left
            w[1 : n - 1 : 2, 1 : m - 2 : 2]
            # Right
            + w[1 : n - 1 : 2, 3:m:2]
            # Up
            + w[0 : n - 2 : 2, 2 : m - 1 : 2]
            # Down
            + w[2:n:2, 2 : m - 1 : 2]
        )
    ) / (Iw[0:k:2, 1:d:2] ** 2 + 4 * lam / h**2)


def red_update(
    w: np.ndarray,
    p: np.ndarray,
    Iw: np.ndarray,
    Ip: np.ndarray,
    lam: float,
    rhs: np.ndarray,
    h: float,
) -> None:
    """
    Red update using Red-Black Gauss-Seidel
    Note: This function mutates w

    Args:
    ---
    w : np.ndarray
        The padded array to update
    p : np.ndarray
        The padded other array
    Iw : np.ndarray
        If w=u -> Iw=Ix
        If w=v -> Iw=Iy
    Ip : np.ndarray
        If w=u -> Ip=Iy
        If w=v -> Ip=Ix
    lam : float
        Penalty term
    rhs : np.ndarray
        RHS in eq
    h : int
        Stepsize corresponding to the level

    Returns:
    ---
    None
    """
    n, m = w.shape
    k, d = Iw.shape

    # Corner update
    w[1 : n - 1 : 2, 1 : m - 1 : 2] = (
        rhs[0:k:2, 0:d:2]
        - Ip[0:k:2, 0:d:2] * Iw[0:k:2, 0:d:2] * p[1 : n - 1 : 2, 1 : m - 1 : 2]
        + lam
        / h**2
        * (
            # Left
            w[1 : n - 1 : 2, 0 : m - 2 : 2]
            # Right
            + w[1 : n - 1 : 2, 2:m:2]
            # Up
            + w[0 : n - 2 : 2, 1 : m - 1 : 2]
            # Down
            + w[2:n:2, 1 : m - 1 : 2]
        )
    ) / (Iw[0:k:2, 0:d:2] ** 2 + 4 * lam / h**2)

    # Interior corner update
    w[2 : n - 1 : 2, 2 : m - 1 : 2] = (
        rhs[1:k:2, 1:d:2]
        - Ip[1:k:2, 1:d:2] * Iw[1:k:2, 1:d:2] * p[2 : n - 1 : 2, 2 : m - 1 : 2]
        + lam
        / h**2
        * (
            # Left
            w[2 : n - 1 : 2, 1 : m - 2 : 2]
            # Right
            + w[2 : n - 1 : 2, 3:m:2]
            # Up
            + w[1 : n - 2 : 2, 2 : m - 1 : 2]
            # Down
            + w[3:n:2, 2 : m - 1 : 2]
        )
    ) / (Iw[1:k:2, 1:d:2] ** 2 + 4 * lam / h**2)


def residual(
    u: np.ndarray,
    v: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the residual of the system

    Args:
    ---
    u : np.ndarray
        u
    v : np.ndarray
        v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame
    lam : float
        Penalty term
    rhs_u : np.ndarray
        RHS of eq for u
    rhs_v : np.ndarray
        RHS of eq for v
    h: int
        Stepsize corresponding to the level

    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Residual of u, v
    """
    fu, fv = F(u, v, Ix, Iy, lam, h)
    du = rhs_u - fu
    dv = rhs_v - fv
    return du, dv


def restriction(
    ru_h: np.ndarray,
    rv_h: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict by power of 2

    Args:
    ---
    ru_h : np.ndarray
        Residual for u
    rv_h : np.ndarray
        Residual for v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame

    Returns:
    ---
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Restricted u, v, Ix, Iy
    """
    assert ru_h.shape == rv_h.shape == Ix.shape == Iy.shape

    n, m = ru_h.shape

    ru_2h = (
        ru_h[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + ru_h[1:n:2, 0 : m - 1 : 2]  # lower left
        + ru_h[0 : n - 1 : 2, 1:m:2]  # upper right
        + ru_h[1:n:2, 1:m:2]  # lower right
    ) / 4
    rv_2h = (
        rv_h[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + rv_h[1:n:2, 0 : m - 1 : 2]  # lower left
        + rv_h[0 : n - 1 : 2, 1:m:2]  # upper right
        + rv_h[1:n:2, 1:m:2]  # lower right
    ) / 4

    Ix2h = (
        Ix[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + Ix[1:n:2, 0 : m - 1 : 2]  # lower left
        + Ix[0 : n - 1 : 2, 1:m:2]  # upper right
        + Ix[1:n:2, 1:m:2]  # lower right
    ) / 4

    Iy2h = (
        Iy[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + Iy[1:n:2, 0 : m - 1 : 2]  # lower left
        + Iy[0 : n - 1 : 2, 1:m:2]  # upper right
        + Iy[1:n:2, 1:m:2]  # lower right
    ) / 4

    return ru_2h, rv_2h, Ix2h, Iy2h


def prolongation(eu_2h: np.ndarray, ev_2h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prolongation of error of coarse grid onto a fine grid

    Args:
    ---
    eu_2h : np.ndarray
        Residual for u (At coarse grid)
    ev_2h : np.ndarray
        Residual for v (At coarse grid)

    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Prolongated eu_2h, ev_2h to e_h
    """
    assert eu_2h.shape == ev_2h.shape

    eu_h = interpolate(eu_2h)
    ev_h = interpolate(ev_2h)

    return eu_h, ev_h


def interpolate(e_2h: np.ndarray) -> np.ndarray:
    """
    Interpolation of error of coarse grid onto a fine grid

    Args:
    ---
    e_2h : np.ndarray
        Residual to interpolate

    Returns:
    ---
    np.ndarray
        Interpolated e_2h to e_h
    """
    n, m = e_2h.shape

    e_h = np.zeros((2 * n, 2 * m))
    k, d = e_h.shape

    # Upper left
    e_h[0 : k - 1 : 2, 0 : d - 1 : 2] = e_2h
    # Lower left
    e_h[1:k:2, 0 : d - 1 : 2] = e_2h
    # Upper right
    e_h[0 : k - 1 : 2, 1:d:2] = e_2h
    # Lower right
    e_h[1:k:2, 1:d:2] = e_2h

    return e_h


def interpolate_fancy(e_2h: np.ndarray) -> np.ndarray:
    """
    Interpolation of error of coarse grid onto a fine grid

    Args:
    ---
    e_2h : np.ndarray
        Residual to interpolate

    Returns:
    ---
    np.ndarray
        Interpolated e_2h to e_h
    """
    n, m = e_2h.shape

    e_h = np.zeros((2 * n, 2 * m))
    k, d = e_h.shape

    # ---
    # Inner points
    # ---

    # Upper left
    e_h[1 : k - 1 : 2, 1 : d - 1 : 2] = (
        9 * e_2h[0 : n - 1, 0 : m - 1]  # Upper left
        + 3 * e_2h[0 : n - 1, 1:m]  # Upper right
        + 3 * e_2h[1:n, 0 : m - 1]  # Lower left
        + e_2h[1:n, 1:m]  # Lower right
    ) / 16

    # Upper right
    e_h[1 : k - 1 : 2, 2 : d - 1 : 2] = (
        3 * e_2h[0 : n - 1, 0 : m - 1]  # Upper left
        + 9 * e_2h[0 : n - 1, 1:m]  # Upper right
        + 1 * e_2h[1:n, 0 : m - 1]  # Lower left
        + 3 * e_2h[1:n, 1:m]  # Lower right
    ) / 16

    # Lower left
    e_h[2 : k - 1 : 2, 1 : d - 1 : 2] = (
        3 * e_2h[0 : n - 1, 0 : m - 1]  # Upper left
        + 1 * e_2h[0 : n - 1, 1:m]  # Upper right
        + 9 * e_2h[1:n, 0 : m - 1]  # Lower left
        + 3 * e_2h[1:n, 1:m]  # Lower right
    ) / 16

    # Lower right
    e_h[2 : k - 1 : 2, 2 : d - 1 : 2] = (
        1 * e_2h[0 : n - 1, 0 : m - 1]  # Upper left
        + 3 * e_2h[0 : n - 1, 1:m]  # Upper right
        + 3 * e_2h[1:n, 0 : m - 1]  # Lower left
        + 9 * e_2h[1:n, 1:m]  # Lower right
    ) / 16

    # ---
    # {Edges} \ {Vertices}
    # ---
    # Left edge
    e_h[1 : k - 2 : 2, 0] = (3 * e_2h[0 : n - 1, 0] + e_2h[1:n, 0]) / 4  # Upper
    e_h[2 : k - 1 : 2, 0] = (e_2h[0 : n - 1, 0] + 3 * e_2h[1:n, 0]) / 4  # Lower

    # Right edge
    e_h[1 : k - 2 : 2, -1] = (3 * e_2h[0 : n - 1, -1] + e_2h[1:n, -1]) / 4  # Upper
    e_h[2 : k - 1 : 2, -1] = (e_2h[0 : n - 1, -1] + 3 * e_2h[1:n, -1]) / 4  # Lower

    # Upper edge
    e_h[0, 1 : d - 2 : 2] = (3 * e_2h[0, 0 : m - 1] + e_2h[0, 1:m]) / 4  # Left
    e_h[0, 2 : d - 1 : 2] = (e_2h[0, 0 : m - 1] + 3 * e_2h[0, 1:m]) / 4  # Right

    # Lower edge
    e_h[-1, 1 : d - 2 : 2] = (3 * e_2h[-1, 0 : m - 1] + e_2h[-1, 1:m]) / 4  # Left
    e_h[-1, 2 : d - 1 : 2] = (e_2h[-1, 0 : m - 1] + 3 * e_2h[-1, 1:m]) / 4  # Right

    # ---
    # Vertices
    # ---
    e_h[0, 0] = e_2h[0, 0]  # Upper left
    e_h[0, -1] = e_2h[0, -1]  # Upper right
    e_h[-1, 0] = e_2h[-1, 0]  # Lower left
    e_h[-1, -1] = e_2h[-1, -1]  # Lower right

    return e_h


if __name__ == "__main__":
    N = 4
    M = 5
    u = np.linspace(1, N * M, N * M).reshape(N, M)
    u = np.pad(u, 1)
    v = u.copy()
    Ix, Iy = np.zeros((N, M)), np.zeros((N, M))
    rhsu, rhsv = np.zeros((N, M)), np.zeros((N, M))
    # smoothing(u, v, Ix, Iy, 1, rhsu, rhsv, 1, 1)
    print(u)
    black_update(u, v, Ix, Iy, 1, rhsu, 1)
    print(u)
