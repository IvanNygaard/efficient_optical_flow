# Imports
from preprocessing import preprocessing_test_images
from OF_cg import OF_cg, cg
from helper_functions import mycomputeColor
import numpy as np
import matplotlib.pyplot as plt
from multigrid import V_cycle
from time import perf_counter
from run_numerical_experiments import run_numerical_experiments


def main():
    """
    Main function for calling procedues to numerically solve the optical flow problem.

    Returns:
    ---
    None
    """

    # Unpack data from preprocessing (spatial derivatives, temporal derivatives, etc.)
    #Ix, Iy, It, rhsu, rhsv, h = preprocessing(1, 1, 1, 1, 1)
    #print("Ix: ", Ix.shape)
    #print("Iy: ", Iy.shape)
    #print("rhsu: ", rhsu.shape)
    #print("rhsv: ", rhsv.shape)


    #n, m = Ix.shape

    # Call OF_cg to numerically solve for u anv using the CG-method.
    #x, y = np.zeros((n, m)), np.zeros((n, m))


    # start = perf_counter()
    # u, v = OF_cg(x, y, Ix, Iy, 1, rhsu, rhsv)
    # end = perf_counter()
    # print("Time: ", end - start)
    #
    # result = mycomputeColor(u, v)
    # plt.imshow(result)
    # plt.show()

    #start = perf_counter()
    #cg_res = cg(x, y, Ix, Iy, 1, rhsu, rhsv)
    #u_cg, v_cg = cg_res[0], cg_res[1]
    #end = perf_counter()
    #print("Time: ", end - start)


    #vcyc_res = V_cycle(x, y, Ix, Iy, 1, rhsu, rhsv, s1=1, s2=1, level=1, max_level=2)
    #u, v = vcyc_res[0], vcyc_res[1]

    # Plotting
    #result = mycomputeColor(u, v)
    #plt.imshow(result)
    #plt.show()

    
    # Experiment
    run_numerical_experiments(1,1,1,1)



main()
