# Imports
from preprocessing import preprocessing_image
from OF_cg import cg
from helper_functions import mycomputeColor
import numpy as np
import matplotlib.pyplot as plt
from multigrid import V_cycle, residual
from time import perf_counter
from PCG import pcg
from utils import F, norm


def test_MG():
    """
    Main function for calling procedues to numerically solve the optical flow problem.

    Returns:
    ---
    None
    """

    # Unpack data from preprocessing (spatial derivatives, temporal derivatives, etc.)
    Ix, Iy, It, rhsu, rhsv, h = preprocessing_image(1, 1, 1, 1)
    print("Ix: ", Ix.shape)
    print("Iy: ", Iy.shape)
    print("rhsu: ", rhsu.shape)
    print("rhsv: ", rhsv.shape)

    n, m = Ix.shape
    lam = 5
    max_level = 5

    # rng = np.random.default_rng(0)
    # Ix = rng.random((n, m))
    # Iy = rng.random((n, m))
    # It = rng.random((n, m))
    # rhsu = rng.random((n, m))
    # rhsv = rng.random((n, m))
    # Ix = np.zeros_like(Ix)
    # Iy = np.zeros_like(Iy)
    # It = np.zeros_like(It)
    # rhsu = np.zeros((n, m))
    # rhsv = np.zeros((n, m))

    # Call OF_cg to numerically solve for u anv using the CG-method.
    x, y = np.zeros((n, m)), np.zeros((n, m))

    # result = mycomputeColor(u, v)
    # plt.imshow(result)
    # plt.show()

    # start = perf_counter()
    # u_cg, v_cg = cg(x, y, Ix, Iy, lam, rhsu, rhsv, tol=1e-2)
    # end = perf_counter()
    # print("Time: ", end - start)

    fu, fv = F(x, y, Ix, Iy, lam, h)
    ru = rhsu - fu
    rv = rhsv - fv
    print("Initial Residual V-cycle: ", norm(ru, rv))

    # u, v = V_cycle(
    #     x, y, Ix, Iy, lam, rhsu, rhsv, s1=2, s2=3, level=-1, max_level=max_level-1
    # )
    # u_mult, v_mult = V_cycle(x, y, Ix, Iy, lam, rhsu, rhsv, s1=3, s2=3, level=0, max_level=4)
    # u, v = pcg(x, y, Ix, Iy, 1, rhsu, rhsv)
    #

    # fu, fv = F(u_cg, v_cg, Ix, Iy, lam, h)
    # ru = rhsu - fu
    # rv = rhsv - fv
    # print("Residual CG: ", norm(ru, rv))

    # fu, fv = F(u, v, Ix, Iy, lam, h)
    # ru = rhsu - fu
    # rv = rhsv - fv
    # print("Residual V-cycle: ", norm(ru, rv))

    # fu, fv = F(u_mult, v_mult, Ix, Iy, lam, h)
    # ru = rhsu - fu
    # rv = rhsv - fv
    # print("Residual V-cycle: ", norm(ru, rv))

    # Iterating V_cycle
    u_itr, v_itr = np.zeros((n, m)), np.zeros((n, m))

    fu, fv = F(u_itr, v_itr, Ix, Iy, lam, h)
    ru = rhsu - fu
    rv = rhsv - fv
    print("Initial Residual V-cycle: ", norm(ru, rv))

    for _ in range(50):
        u_itr, v_itr = V_cycle(
            np.copy(u_itr),
            np.copy(v_itr),
            Ix,
            Iy,
            lam,
            rhsu,
            rhsv,
            s1=3,
            s2=3,
            level=0,
            max_level=max_level,
        )
        ru, rv = residual(u_itr, v_itr, Ix, Iy, lam, rhsu, rhsv, h)

        print("Residual V-cycle (Itr): ", norm(ru, rv))

    print(u_itr)
    fu, fv = F(u_itr, v_itr, Ix, Iy, lam, h)
    ru = rhsu - fu
    rv = rhsv - fv
    print("Residual V-cycle (Itr): ", norm(ru, rv))

    result = mycomputeColor(u_itr, v_itr)
    plt.imshow(result)
    plt.show()

    # result = mycomputeColor(u_mult, v_mult)
    # plt.imshow(result)
    # plt.show()

    # result = mycomputeColor(u_cg, v_cg)
    # plt.imshow(result)
    # plt.show()

    # Plotting
    # result = mycomputeColor(u, v)
    # plt.imshow(result)
    # plt.show()


test_MG()
