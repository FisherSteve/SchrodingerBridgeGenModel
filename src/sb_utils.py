import numpy as np


def default_kernel(x, H):
    """Default kernel used in Schrodinger Bridge simulations."""
    return np.where(np.abs(x) < H, (H * H - x * x) ** 2, 0.0)


def gaussian_kernel(x, H):
    """Gaussian kernel."""
    return np.exp(-(x ** 2) / (2 * H ** 2)) / (np.sqrt(2 * np.pi * H ** 2))


def laplacian_kernel(x, H):
    """Laplacian kernel."""
    return np.exp(-np.abs(x) / H) / (2 * H)


def polynomial_kernel(x, H, degree=3):
    """Polynomial kernel."""
    return (1 + x / H) ** degree


def schedule(timeEuler, maturity, timestep):
    """Append time points to ``timeEuler`` from 0 to ``maturity``."""
    timeEuler.extend(np.arange(0, maturity, timestep))
    timeEuler.append(maturity)
