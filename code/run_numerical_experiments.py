import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing_test_images, preprocessing_image
from OF_cg import OF_cg, cg
from time import perf_counter
from multigrid import V_cycle, residual
from helper_functions import mycomputeColor
from utils import F, norm
from PCG import pcg


def run_numerical_experiments(sigma, 
                              dx, 
                              dy, 
                              dt):
    """
    Function used to create convergence plots from numerical experiments, with some nice styling,
    styling retrieved from https://github.com/AndreyChurkin/BeautifulFigures/blob/main/Python/scripts/beautiful_figure_example.py
    """
    # k values dictate simage size
    k_vals = [6, 7, 8, 9] 

    # TEST I : sparse CG vs on-the-grid-CG for synthetic images of two Gaussian circeling each other for image sizes 2^k x 2^k, k = 6,7,8,9 
    res_ratio_vals = []

    # testing of sparse implementation of cg
    for i in range(len(k_vals)):
        Ix, Iy, It, rhsu, rhsv, h = preprocessing_test_images(dx, dy, dt, k_vals[i]) 

        n, m = Ix.shape
        x, y = np.zeros((n, m)), np.zeros((n, m))

        start = perf_counter()
        u, v, res_ratio = OF_cg(x, 
                                y, 
                                Ix, 
                                Iy, 
                                4 ** (k_vals[i]-4), 
                                rhsu, 
                                rhsv)
        stop = perf_counter()
        print(f'Time for sparse conjugate gradient = {stop - start}, k = {k_vals[i]}')


        res_ratio_vals.append(res_ratio)

        result = mycomputeColor(u, v)
        plt.imshow(result)
        plt.show()


    plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
    }) 


    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")  
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")

    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3) 
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3) 
  
    ax.set_yscale("log")
    
    plt.axhline(
    y=1e-8,         
    color="black",
    linestyle=":",
    linewidth=1.0
    )


    plt.title("Sparse conjugate gradient applied to synthetic test case")
    ax.legend()
    plt.savefig("synthetic_sparse_cg_no_multi_all_k.pdf", dpi=450, bbox_inches='tight')
    plt.show()
 


    # testing of working on the grid implementation of cg
    for i in range(len(k_vals)):

            Ix, Iy, It, rhsu, rhsv, h = preprocessing_test_images(dx, dy, dt, k_vals[i])    
            n, m = Ix.shape
            x, y = np.zeros((n, m)), np.zeros((n, m))


            start = perf_counter()
            cg_res = cg(x, 
                        y, 
                        Ix, 
                        Iy, 
                        4 ** (k_vals[i]-4), 
                        rhsu, 
                        rhsv)
            stop = perf_counter()
            print(f'On-the-grid conjugate gradient applied to test case = {stop - start}, k = {k_vals[i]}')
  
            u, v, res_ratio = cg_res[0], cg_res[1], cg_res[2]
 
            res_ratio_vals.append(res_ratio)
            
            result = mycomputeColor(u, v)
            plt.imshow(result)
            plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3)
 
    ax.set_yscale("log")

    plt.axhline(
    y=1e-8,
    color="black",
    linestyle=":",
    linewidth=1.0
    )


    plt.title("On-the-grid conjugate gradient applied to synthetic test case")
    ax.legend() 
    plt.savefig("synthetic_grid_cg_no_multi_all_k.pdf", dpi=450, bbox_inches='tight')
    plt.show()



    # TEST II : on-the-grid-cg with multigrid for synthetic images of two Gaussian circeling each other for image sizes 2^k x 2^k, k = 6,7,8,9
    # on-the-grid CG + multigrid
    res_ratio_vals = []

    for i in range(len(k_vals)):
        Ix, Iy, It, rhsu, rhsv, h = preprocessing_test_images(dx, dy, dt, k_vals[i])

        n, m = Ix.shape    
        x, y = np.zeros((n, m)), np.zeros((n, m))

        start = perf_counter()
        u,v = V_cycle(x, y, Ix, Iy, 4 ** (k_vals[i]-4), rhsu, rhsv, s1=15, s2=15, level=0, max_level=5) 
        stop = perf_counter()
        print(fr'Time multigird CG = {stop - start}, k = {k_vals[i]}, $\nu$ = 15, max_level = 5')



        # Comute first residual
        fu, fv = F(x,y, Ix, Iy, 4 ** (k_vals[i]-4), h)
        ru = rhsu - fu
        rv = rhsv - fv
        r0 = norm(ru, rv)

        residuals = []

        for _ in range(50):
            x, y = V_cycle(
                            np.copy(x),
            		        np.copy(y),
            		        Ix,
            		        Iy,
            		        4 ** (k_vals[i]-4),
            		        rhsu,
            		        rhsv,
            		        s1=15,
            		        s2=15,
            		        level=0,
            		        max_level=5,
        		            )

            ru, rv = residual(x, y, Ix, Iy, 4 ** (k_vals[i]-4), rhsu, rhsv, h)

            #print("Residual V-cycle (Itr): ", norm(ru, rv))
            residuals.append(norm(ru,rv)/r0)
        res_ratio_vals.append(residuals)

     
        result = mycomputeColor(u, v)
        plt.imshow(result)
        plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))


    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3)
  
    ax.set_yscale("log")

    plt.title(r"Multigrid conjugate gradient applied to synthetic test case, $\nu$ = 5, max-level = 5")
    ax.legend()
    plt.savefig("cg_MULTIGRID_synthetic_images.pdf", dpi=450, bbox_inches='tight')
    plt.show()



    # TEST III: on-the-grid-cg with multigrid for provided test frames of man closing a car door. 
    # on-the-grid CG + multigrid
    #sigma_values   = [1.0, 2.0, 5.0]           V-cycle diverges for smalle lambda than 5.0 for sigma = 1.0, these values
    #reg_values     = [1e-3, 1.0, 1e7]          
    res_ratio_vals = []

    sigma_values = [1.0, 2.5, 5.0]
    reg_values   = [1e-3, 1.0, 1e7]

    for i in range(len(sigma_values)):
        for j in range(len(reg_values)):
            Ix, Iy, It, rhsu, rhsv, h = preprocessing_image(sigma_values[i], dx, dy, dt)

            n, m = Ix.shape    
            x, y = np.zeros((n, m)), np.zeros((n, m))

            start = perf_counter()
            u,v = V_cycle(x, y, Ix, Iy, reg_values[j], rhsu, rhsv, s1=15, s2=15, level=0, max_level=5) 
            stop = perf_counter()
            print(f'Time for on-the-grid CG man closing car door (with multigrid) = {stop - start}, sigma = {sigma_values[i]}, lambda = {reg_values[j]}, nu = 15, max_level = 5')



            # Comute first residual
            fu, fv = F(x,y, Ix, Iy, reg_values[j], h)
            ru = rhsu - fu
            rv = rhsv - fv
            r0 = norm(ru, rv)

            residuals = []

            for _ in range(50):
                x, y = V_cycle(
                                  np.copy(x),
            		              np.copy(y),
            		              Ix,
            		              Iy,
            		              reg_values[j],
            		              rhsu,
            		              rhsv,
            		              s1=15,
            		              s2=15,
            		              level=0,
            		              max_level=5,
        		                  )
                ru, rv = residual(x, y, Ix, Iy, reg_values[j], rhsu, rhsv, h)

                #print("Residual V-cycle (Itr): ", norm(ru, rv))
                residuals.append(norm(ru,rv)/r0)
            res_ratio_vals.append(residuals)

     
            result = mycomputeColor(u, v)
            plt.imshow(result)
            plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[0]}", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[1]}", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[2]}", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[0]}", color = "cyan", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[4])), res_ratio_vals[4], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[1]}", color = "crimson", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[5])), res_ratio_vals[5], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[2]}", color = "goldenrod", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[6])), res_ratio_vals[6], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[0]}", color = "indigo", linewidth = 1.5, zorder = 3) 
    ax.plot(np.arange(0, len(res_ratio_vals[7])), res_ratio_vals[7], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[1]}", color = "darkorange", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[8])), res_ratio_vals[8], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[2]}", color = "slategray", linewidth = 1.5, zorder = 3)

    ax.set_yscale("log")


    plt.title(r"Multigrid conjugate gradient applied to test frames of man closing car door, $\nu$ = 15, max-level = 5")
    ax.legend()
    plt.savefig("cardoor_multigriddifferentsigmaslambdas.pdf", dpi=450, bbox_inches='tight')
    plt.show()



    # TEST IV: Same as III, but with larger number of iterations for the smoother. 
    res_ratio_vals = []

    sigma_values = [1.0, 2.5, 5.0]
    reg_values   = [1e-3, 1.0, 1e7]


    for i in range(len(sigma_values)):
        for j in range(len(reg_values)):
            Ix, Iy, It, rhsu, rhsv, h = preprocessing_image(sigma_values[i], dx, dy, dt)

            n, m = Ix.shape    
            x, y = np.zeros((n, m)), np.zeros((n, m))

            start = perf_counter()
            u,v = V_cycle(x, y, Ix, Iy, reg_values[j], rhsu, rhsv, s1=50, s2=50, level=0, max_level=5) 
            stop = perf_counter()
            print(f'Time for on-the-grid CG man closing car door (with multigrid) = {stop - start}, sigma = {sigma_values[i]}, lambda = {reg_values[j]}')
            print(f'Time multigird CG = {stop - start}, k = {k_vals[i]}, smooth_iter = 50, max_level = 5')


            # Comute first residual
            fu, fv = F(x,y, Ix, Iy, reg_values[j], h)
            ru = rhsu - fu
            rv = rhsv - fv
            r0 = norm(ru, rv)

            residuals = []

            for _ in range(50):
                x, y = V_cycle(
                                  np.copy(x),
            		              np.copy(y),
            		              Ix,
            		              Iy,
            		              reg_values[j],
            		              rhsu,
            		              rhsv,
            		              s1=50,
            		              s2=50,
            		              level=0,
            		              max_level=5,
        		                  )
                ru, rv = residual(x, y, Ix, Iy, reg_values[j], rhsu, rhsv, h)

                #print("Residual V-cycle (Itr): ", norm(ru, rv))
                residuals.append(norm(ru,rv)/r0)
            res_ratio_vals.append(residuals)

     
            result = mycomputeColor(u, v)
            plt.imshow(result)
            plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[0]}", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[1]}", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[2]}", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[0]}", color = "cyan", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[4])), res_ratio_vals[4], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[1]}", color = "crimson", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[5])), res_ratio_vals[5], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[2]}", color = "goldenrod", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[6])), res_ratio_vals[6], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[0]}", color = "indigo", linewidth = 1.5, zorder = 3) 
    ax.plot(np.arange(0, len(res_ratio_vals[7])), res_ratio_vals[7], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[1]}", color = "darkorange", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[8])), res_ratio_vals[8], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[2]}", color = "slategray", linewidth = 1.5, zorder = 3)

    ax.set_yscale("log")

    plt.title(r"Multigrid conjugate gradient applied to test frames of man closing car door, $\nu$ = 50, max-level = 5")
    ax.legend()
    plt.savefig("cardoor_multigriddifferentsigmaslambdas_BIGSMOOTHING.pdf", dpi=450, bbox_inches='tight')
    plt.show()


    # TEST V: Same as III, but with lower max_level
    res_ratio_vals = []

    sigma_values = [1.0, 2.5, 5.0]
    reg_values   = [1e-3, 1.0, 1e7]

    for i in range(len(sigma_values)):
        for j in range(len(reg_values)):
            Ix, Iy, It, rhsu, rhsv, h = preprocessing_image(sigma_values[i], dx, dy, dt)

            n, m = Ix.shape    
            x, y = np.zeros((n, m)), np.zeros((n, m))

            start = perf_counter()
            u,v = V_cycle(x, y, Ix, Iy, reg_values[j], rhsu, rhsv, s1=15, s2=15, level=0, max_level=2) 
            stop = perf_counter()
            print(f'Time for on-the-grid CG man closing car door (with multigrid) = {stop - start}, sigma = {sigma_values[i]}, lambda = {reg_values[j]}')
            print(f'Time multigird CG = {stop - start}, k = {k_vals[i]}, s1, s2 = 15, 15, max_level = 2')


            # Comute first residual
            fu, fv = F(x,y, Ix, Iy, reg_values[j], h)
            ru = rhsu - fu
            rv = rhsv - fv
            r0 = norm(ru, rv)

            residuals = []

            for _ in range(50):
                x, y = V_cycle(
                                  np.copy(x),
            		              np.copy(y),
            		              Ix,
            		              Iy,
            		              reg_values[j],
            		              rhsu,
            		              rhsv,
            		              s1=15,
            		              s2=15,
            		              level=0,
            		              max_level=5,
        		                  )
                ru, rv = residual(x, y, Ix, Iy, reg_values[j], rhsu, rhsv, h)

                #print("Residual V-cycle (Itr): ", norm(ru, rv))
                residuals.append(norm(ru,rv)/r0)
            res_ratio_vals.append(residuals)

     
            result = mycomputeColor(u, v)
            plt.imshow(result)
            plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[0]}", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[1]}", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = fr"$\sigma$ = {sigma_values[0]}, $\lambda$ = {reg_values[2]}", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[0]}", color = "cyan", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[4])), res_ratio_vals[4], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[1]}", color = "crimson", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[5])), res_ratio_vals[5], label = fr"$\sigma$ = {sigma_values[1]}, $\lambda$ = {reg_values[2]}", color = "goldenrod", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[6])), res_ratio_vals[6], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[0]}", color = "indigo", linewidth = 1.5, zorder = 3) 
    ax.plot(np.arange(0, len(res_ratio_vals[7])), res_ratio_vals[7], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[1]}", color = "darkorange", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[8])), res_ratio_vals[8], label = fr"$\sigma$ = {sigma_values[2]}, $\lambda$ = {reg_values[2]}", color = "slategray", linewidth = 1.5, zorder = 3)

    ax.set_yscale("log")

    plt.title(r"Multigrid conjugate gradient applied to test frames of man closing car door, $\nu = 15$, max-level = 2")
    ax.legend()
    plt.savefig("cardoor_multigriddifferentsigmaslambdas_BIGMAXLEVEL.pdf", dpi=450, bbox_inches='tight')
    plt.show()


    # TEST V: Preconditioned conjugate gradient vs non-preconditioned conjugate gradient. 
    res_ratio_vals = []

    sigma = 1.0
    lamb  = 5.0

    Ix, Iy, It, rhsu, rhsv, h = preprocessing_image(sigma, dx, dy, dt)    
    n, m = Ix.shape
    x, y = np.zeros((n, m)), np.zeros((n, m))

    start = perf_counter()
    cg_res = cg(x, 
                y, 
                Ix, 
                Iy, 
                lamb, 
                rhsu, 
                rhsv)
    stop = perf_counter()
    print(f'Time non-preconditioned CG  = {stop - start}')
  
    u, v, res_ratio = cg_res[0], cg_res[1], cg_res[2]
 
    res_ratio_vals.append(res_ratio)
            
    result = mycomputeColor(u, v)
    plt.imshow(result)
    plt.show()


    start = perf_counter()
    cg_res = pcg(x,y, Ix, Iy, lamb, rhsu, rhsv, 15, 15, 5)
    stop = perf_counter()
    print(f'Time preconditioned CG = {stop - start}')

    u, v, res_ratio = cg_res[0], cg_res[1], cg_res[2]

    res_ratio_vals.append(res_ratio)

    result = mycomputeColor(u,v)
    plt.imshow(result)
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
 

    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = "on-the-grid cg", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = "preconditioned on-the-grid-cg", color = "blue", linewidth = 1.5, zorder = 3)
 
    ax.set_yscale("log")

    plt.axhline(
    y=1e-8,
    color="black",
    linestyle=":",
    linewidth=1.0
    )


    plt.title(fr"Non-preconditioned vs. preconditioned conjugate gradient, $\sigma$ = {sigma}, $\lambda$ = {lamb}")
    ax.legend() 
    plt.savefig("preconvsnonprecond.pdf", dpi=450, bbox_inches='tight')
    plt.show()

