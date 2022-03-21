from model.kernel_ridge_regressors import LaplacianKRR, \
        PeriodicLaplacianKRR, PowerLaplacianKRR, RadialBasisKRR
from .weighted_average import GaussianWeightedAverage


models = [GaussianWeightedAverage,
          LaplacianKRR,
          PeriodicLaplacianKRR,
          PowerLaplacianKRR,
          RadialBasisKRR]
