from model.kernel_ridge_regressors import LaplacianKRR, \
        PeriodicLaplacianKRR, PowerLaplacianKRR, RadialBasisKRR
from .weighted_average import GaussianWeightedAverage
from .test_model import TestModel


models = [TestModel,
          GaussianWeightedAverage,
          LaplacianKRR,
          PeriodicLaplacianKRR,
          PowerLaplacianKRR,
          RadialBasisKRR]
