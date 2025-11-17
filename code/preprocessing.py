# Imports:
from helper_functions import generate_test_image
import cv2
import matplotlib.pyplot as plt
import scipy as sci
import numpy as np


def preprocessing_test_images(
    dx: float,
    dy: float,
    dt: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Preprocessing of frames and computation of spatial derivative and temporal derivative.

    Args:
    ---
    sigma : int
           Standard deviation for Gaussian kernel
    dx : int
        Grid spacing in x-dimension
    dy : int
        Grid spacing in y-dimension
    dt : int
        Time-difference between img0 and img1
    k : int
       Scaling parameter for image-size
    """
    assert dx == dy == dt

    h = dx
    # Synthetic test-cases
    I0, I1 = generate_test_image(100, 2)
    I0 = cv2.resize(I0, (2**k, 2**k))
    I1 = cv2.resize(I1, (2**k, 2**k))
    #plt.imshow(I0)
    #plt.imshow(I1)
    plt.show()

    #print("I0: ", I0.shape)
    I0 = np.astype(I0, np.float64)
    I1 = np.astype(I1, np.float64)

    # Initialize result vectors
    I0x = np.zeros_like(I0)
    I1x = np.zeros_like(I0)
    I0y = np.zeros_like(I0)
    I1y = np.zeros_like(I0)
    It = np.zeros_like(I0)

    # Interior Derivatives (Forward diff.)
    I0x[:, :-1] = (I0[:, 1:] - I0[:, :-1]) / dx  # I0[i,j] = I0[y,x]
    I1x[:, :-1] = (I1[:, 1:] - I1[:, :-1]) / dx
    I0y[:-1, :] = (I0[1:, :] - I0[:-1, :]) / dy
    I1y[:-1, :] = (I1[1:, :] - I1[:-1, :]) / dy

    # Boundary Derivatives (Backwards diff.)
    I0x[:, -1] = (I0[:, -1] - I0[:, -2]) / dx
    I1x[:, -1] = (I1[:, -1] - I1[:, -2]) / dx
    I0y[-1, :] = (I0[-1, :] - I0[-2, :]) / dy
    I1y[-1, :] = (I1[-1, :] - I1[-2, :]) / dy

    # Averaging
    Ix = 1 / 2 * (I0x + I1x)
    Iy = 1 / 2 * (I0y + I1y)

    # Time derivative (frame difference)
    It = (I1 - I0) / dt

    # Right hand sides of dicretized equations
    rhsu = -It * Ix
    rhsv = -It * Iy

    return (
        Ix,
        Iy,
        It,
        rhsu,
        rhsv,
        h,
    )


def preprocessing_image(
    sigma: float,
    dx: float,
    dy: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Preprocessing of frames and computation of spatial derivative and temporal derivative.

    Args:
    ---
    sigma : int
           Standard deviation for Gaussian kernel
    dx : int
        Grid spacing in x-dimension
    dy : int
        Grid spacing in y-dimension
    dt : int
        Time-difference between img0 and img1
    k : int
       Scaling parameter for image-size
    """
    assert dx == dy == dt

    h = dx
    #print(plt.imread("../reference_frames/frame10.png", "png"))
    # Reading, smoothing and scaling of images
    I0 = sci.ndimage.gaussian_filter(
        plt.imread("../reference_frames/frame10.png", "png"), sigma
    )
    I1 = sci.ndimage.gaussian_filter(
        plt.imread("../reference_frames/frame11.png", "png"), sigma
    )

    #print("I0: ", I0.shape)
    I0 = np.astype(I0, np.float64)
    I1 = np.astype(I1, np.float64)

    # Initialize result vectors
    I0x = np.zeros_like(I0)
    I1x = np.zeros_like(I0)
    I0y = np.zeros_like(I0)
    I1y = np.zeros_like(I0)
    It = np.zeros_like(I0)

    # Interior Derivatives (Forward diff.)
    I0x[:, :-1] = (I0[:, 1:] - I0[:, :-1]) / dx  # I0[i,j] = I0[y,x]
    I1x[:, :-1] = (I1[:, 1:] - I1[:, :-1]) / dx
    I0y[:-1, :] = (I0[1:, :] - I0[:-1, :]) / dy
    I1y[:-1, :] = (I1[1:, :] - I1[:-1, :]) / dy

    # Boundary Derivatives (Backwards diff.)
    I0x[:, -1] = (I0[:, -1] - I0[:, -2]) / dx
    I1x[:, -1] = (I1[:, -1] - I1[:, -2]) / dx
    I0y[-1, :] = (I0[-1, :] - I0[-2, :]) / dy
    I1y[-1, :] = (I1[-1, :] - I1[-2, :]) / dy

    # Averaging
    Ix = 1 / 2 * (I0x + I1x)
    Iy = 1 / 2 * (I0y + I1y)

    # Time derivative (frame difference)
    It = (I1 - I0) / dt

    # Right hand sides of dicretized equations
    rhsu = -It * Ix
    rhsv = -It * Iy

    # Plotting
    # plt.imshow(I0)
    # plt.show()
    # plt.imshow(I1)
    # plt.show()

    return (
        Ix,
        Iy,
        It,
        rhsu,
        rhsv,
        h,
    )
