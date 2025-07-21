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
    def __init__(self, distSize, nbpaths, dimension, timeSeriesDataVector, kernel_type='default'):
        
        self.distSize = distSize
        self.nbpaths = nbpaths
        self.dimension = dimension
        self.timeSeriesDataVector = np.array(timeSeriesDataVector)

        self.timeSeriesVector = np.zeros((distSize + 1, dimension))
        self.timeSeriesVector[0, :] = self.timeSeriesDataVector[0, 0, :]

        self.weights = np.ones(nbpaths) / nbpaths
        self.weights_tilde = np.zeros(nbpaths)

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

    def schedule(self, timeEuler, maturity, timestep):
        sb_schedule(timeEuler, maturity, timestep)

    def simulate_kernel_vectorized(self, nbStepsPerDeltati, H, deltati):
        vtimestepEuler = np.arange(0, deltati + deltati / nbStepsPerDeltati, deltati / nbStepsPerDeltati)
        Brownian = np.random.normal(0, 1, (self.distSize * len(vtimestepEuler) - 1, self.dimension))

        X_ = np.zeros(self.dimension)
        index_ = 0

        for interval in tqdm(range(self.distSize), desc="Intervals"):
            if interval > 0:
                diffs = self.timeSeriesDataVector[:, interval, :] - X_
                self.weights *= self.kernel(diffs, H).prod(axis=1)

            self.weights_tilde = self.weights * np.exp(np.sum((self.timeSeriesDataVector[:, interval + 1, :] - X_) ** 2, axis=1) / (2.0 * deltati))

            for nbtime in range(len(vtimestepEuler) - 1):
                timeprev = vtimestepEuler[nbtime]
                timestep = vtimestepEuler[nbtime + 1] - vtimestepEuler[nbtime]

                exp_factors = np.exp(-np.sum((self.timeSeriesDataVector[:, interval + 1, :] - X_) ** 2, axis=1) / (2.0 * (deltati - timeprev)))
                expecX = np.dot(self.weights_tilde, exp_factors)
                numerator = np.dot(self.weights_tilde * exp_factors, self.timeSeriesDataVector[:, interval + 1, :] - X_)

                timestepsqrt = np.sqrt(timestep)
                if expecX > 0.0:
                    drift = (1.0 / (deltati - timeprev)) * (numerator / expecX)
                else:
                    drift = np.zeros(self.dimension)

                X_ += drift * timestep + Brownian[index_] * timestepsqrt
                index_ += 1

            self.timeSeriesVector[interval + 1] = X_

        return self.timeSeriesVector



class SchrodingerBridge:
    def __init__(self, distSize, nbpaths, timeSeriesData=None, dimension=None, kernel_type='default'):
        self.distSize = distSize
        self.nbpaths = nbpaths
        self.dimension = dimension

        if timeSeriesData is not None:
            self.timeSeriesData = np.array(timeSeriesData)
        else:
            self.timeSeriesData = None

        self.weights = np.ones(nbpaths) / nbpaths
        self.weights_tilde = np.zeros(nbpaths)
    
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

    def simulate_kernel(self, nbStepsPerDeltati, H, deltati):
        vtimestepEuler = np.arange(0, deltati + deltati / nbStepsPerDeltati, deltati / nbStepsPerDeltati)
        Brownian = np.random.normal(0, 1, self.distSize * len(vtimestepEuler) - 1)

        timeSeries = np.zeros(self.distSize + 1)
        timeSeries[0] = self.timeSeriesData[0, 0]
        X_ = timeSeries[0]
        index_ = 0

        for interval in range(self.distSize):
            for particle in range(self.timeSeriesData.shape[0]):
                if interval == 0:
                    self.weights[particle] = 1.0 / self.nbpaths
                else:
                    self.weights[particle] *= self.kernel(self.timeSeriesData[particle, interval] - X_, H)

                self.weights_tilde[particle] = self.weights[particle] * np.exp((self.timeSeriesData[particle, interval + 1] - X_) ** 2 / (2.0 * deltati))

            for nbtime in range(len(vtimestepEuler) - 1):
                expecY = 0.0
                expecX = 0.0
                timeprev = vtimestepEuler[nbtime]
                timestep = vtimestepEuler[nbtime + 1] - vtimestepEuler[nbtime]

                for particle in range(self.timeSeriesData.shape[0]):
                    if nbtime == 0:
                        expecX += self.weights[particle]
                        expecY += self.weights[particle] * (self.timeSeriesData[particle, interval + 1] - X_)
                    else:
                        termtoadd = -(self.timeSeriesData[particle, interval + 1] - X_) ** 2 / (2.0 * (deltati - timeprev))
                        termtoadd = self.weights_tilde[particle] * np.exp(termtoadd)
                        expecX += termtoadd
                        expecY += termtoadd * (self.timeSeriesData[particle, interval + 1] - X_)

                drift = (1.0 / (deltati - timeprev)) * (expecY / expecX) if expecX > 0.0 else 0.0
                X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep)
                index_ += 1

            timeSeries[interval + 1] = X_

        return timeSeries

    def schedule(self, timeEuler, maturity, timestep):
        sb_schedule(timeEuler, maturity, timestep)
