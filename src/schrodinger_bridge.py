"""Schr\u00f6dinger bridge model implementations."""

from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from .sb_utils import (
    default_kernel,
    gaussian_kernel,
    laplacian_kernel,
    polynomial_kernel,
    schedule as sb_schedule,
)

class SchrodingerBridgeMulti:
    """Vectorised multi-dimensional Schr\u00f6dinger bridge simulation."""

    def __init__(
        self,
        dist_size: int,
        nb_paths: int,
        dimension: int,
        time_series_data_vector: Iterable[Iterable[Iterable[float]]],
        kernel_type: str = "default",
    ) -> None:
        """Construct the simulator.

        Parameters
        ----------
        dist_size:
            Number of time intervals in each path.
        nb_paths:
            Number of reference paths.
        dimension:
            Dimension of each path observation.
        time_series_data_vector:
            Reference trajectories of shape ``(nb_paths, dist_size + 1, dimension)``.
        kernel_type:
            Name of the kernel to use.
        """

        self.dist_size = dist_size
        self.nb_paths = nb_paths
        self.dimension = dimension
        self.time_series_data_vector = np.array(time_series_data_vector)

        self.time_series_vector = np.zeros((dist_size + 1, dimension))
        self.time_series_vector[0, :] = self.time_series_data_vector[0, 0, :]

        self.weights = np.ones(nb_paths) / nb_paths
        self.weights_tilde = np.zeros(nb_paths)

        # Kernel selection
        if kernel_type == 'default':
            self.kernel = default_kernel
        elif kernel_type == 'gaussian':
            self.kernel = gaussian_kernel
        elif kernel_type == 'laplacian':
            self.kernel = laplacian_kernel
        elif kernel_type == 'polynomial':
            self.kernel = polynomial_kernel
        else:
            raise ValueError("Unsupported kernel type: {}".format(kernel_type))

    def schedule(self, time_euler: list, maturity: float, timestep: float) -> None:
        """Populate ``time_euler`` with a schedule of time points."""

        sb_schedule(time_euler, maturity, timestep)

    def simulate_kernel_vectorized(
        self,
        nb_steps_per_deltati: int,
        h: float,
        delta_ti: float,
    ) -> np.ndarray:
        """Simulate one path using the vectorised algorithm."""

        vtimestepEuler = np.arange(0, delta_ti + delta_ti / nb_steps_per_deltati, delta_ti / nb_steps_per_deltati)
        Brownian = np.random.normal(0, 1, (self.dist_size * len(vtimestepEuler) - 1, self.dimension))

        X_ = np.zeros(self.dimension)
        index_ = 0

        for interval in tqdm(range(self.dist_size), desc="Intervals"):
            if interval > 0:
                diffs = self.time_series_data_vector[:, interval, :] - X_
                self.weights *= self.kernel(diffs, h).prod(axis=1)

            self.weights_tilde = self.weights * np.exp(
                np.sum((self.time_series_data_vector[:, interval + 1, :] - X_) ** 2, axis=1)
                / (2.0 * delta_ti)
            )

            for nbtime in range(len(vtimestepEuler) - 1):
                timeprev = vtimestepEuler[nbtime]
                timestep = vtimestepEuler[nbtime + 1] - vtimestepEuler[nbtime]

                exp_factors = np.exp(
                    -np.sum((self.time_series_data_vector[:, interval + 1, :] - X_) ** 2, axis=1)
                    / (2.0 * (delta_ti - timeprev))
                )
                expecX = np.dot(self.weights_tilde, exp_factors)
                numerator = np.dot(
                    self.weights_tilde * exp_factors,
                    self.time_series_data_vector[:, interval + 1, :] - X_,
                )

                timestepsqrt = np.sqrt(timestep)
                if expecX > 0.0:
                    drift = (1.0 / (delta_ti - timeprev)) * (numerator / expecX)
                else:
                    drift = np.zeros(self.dimension)

                X_ += drift * timestep + Brownian[index_] * timestepsqrt
                index_ += 1

            self.time_series_vector[interval + 1] = X_

        return self.time_series_vector



class SchrodingerBridge:
    """One-dimensional Schr\u00f6dinger bridge simulator."""

    def __init__(
        self,
        dist_size: int,
        nb_paths: int,
        time_series_data: Optional[Iterable[Iterable[float]]] = None,
        dimension: Optional[int] = None,
        kernel_type: str = "default",
    ) -> None:
        """Create a simulator for one-dimensional paths.

        Parameters
        ----------
        dist_size:
            Number of time intervals in each path.
        nb_paths:
            Number of reference paths.
        time_series_data:
            Array-like containing the reference trajectories.
        dimension:
            Dimension of the reference data when ``time_series_data`` is not provided.
        kernel_type:
            Name of the kernel to use.
        """

        self.dist_size = dist_size
        self.nb_paths = nb_paths
        self.dimension = dimension

        if time_series_data is not None:
            self.time_series_data = np.array(time_series_data)
        else:
            self.time_series_data = None

        self.weights = np.ones(nb_paths) / nb_paths
        self.weights_tilde = np.zeros(nb_paths)
    
        # Kernel selection
        if kernel_type == 'default':
            self.kernel = default_kernel
        elif kernel_type == 'gaussian':
            self.kernel = gaussian_kernel
        elif kernel_type == 'laplacian':
            self.kernel = laplacian_kernel
        elif kernel_type == 'polynomial':
            self.kernel = polynomial_kernel
        else:
            raise ValueError("Unsupported kernel type: {}".format(kernel_type))

    def simulate_kernel(
        self,
        nb_steps_per_deltati: int,
        h: float,
        delta_ti: float,
    ) -> np.ndarray:
        """Simulate one path using the original iterative algorithm."""

        vtimestepEuler = np.arange(0, delta_ti + delta_ti / nb_steps_per_deltati, delta_ti / nb_steps_per_deltati)
        Brownian = np.random.normal(0, 1, self.dist_size * len(vtimestepEuler) - 1)

        time_series = np.zeros(self.dist_size + 1)
        time_series[0] = self.time_series_data[0, 0]
        X_ = time_series[0]
        index_ = 0

        for interval in range(self.dist_size):
            for particle in range(self.time_series_data.shape[0]):
                if interval == 0:
                    self.weights[particle] = 1.0 / self.nb_paths
                else:
                    self.weights[particle] *= self.kernel(
                        self.time_series_data[particle, interval] - X_, h
                    )

                self.weights_tilde[particle] = self.weights[particle] * np.exp(
                    (self.time_series_data[particle, interval + 1] - X_) ** 2 / (2.0 * delta_ti)
                )

            for nbtime in range(len(vtimestepEuler) - 1):
                expecY = 0.0
                expecX = 0.0
                timeprev = vtimestepEuler[nbtime]
                timestep = vtimestepEuler[nbtime + 1] - vtimestepEuler[nbtime]

                for particle in range(self.time_series_data.shape[0]):
                    if nbtime == 0:
                        expecX += self.weights[particle]
                        expecY += self.weights[particle] * (
                            self.time_series_data[particle, interval + 1] - X_
                        )
                    else:
                        termtoadd = -(
                            self.time_series_data[particle, interval + 1] - X_
                        ) ** 2 / (2.0 * (delta_ti - timeprev))
                        termtoadd = self.weights_tilde[particle] * np.exp(termtoadd)
                        expecX += termtoadd
                        expecY += termtoadd * (
                            self.time_series_data[particle, interval + 1] - X_
                        )

                drift = (1.0 / (delta_ti - timeprev)) * (expecY / expecX) if expecX > 0.0 else 0.0
                X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep)
                index_ += 1

            time_series[interval + 1] = X_

        return time_series

    def schedule(self, time_euler: list, maturity: float, timestep: float) -> None:
        """Populate ``time_euler`` with a schedule of time points."""

        sb_schedule(time_euler, maturity, timestep)
