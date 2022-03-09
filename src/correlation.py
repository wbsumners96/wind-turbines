"""
File for me playing around with turbine-turbine
Correlation functions
"""

import argparse
import copy
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from tqdm import tqdm

from load_data import *
from model.weighted_average import *
from models import *
from visualize import *

mpl.rcParams['savefig.directory'] = ''

# Create dummy class such that diagonal entries in
# correlation function matrix just return their inputs
class DummyPredictor:
    def fit(self, X, Y):
        print('Trying to fit turbine i to turbine i')

    def score(self, X, Y):
        return 1

    def predict(self, X):
        return X


# parser = argparse.ArgumentParser(description='Slow dancing with wind turbines.')
# parser.add_argument('data_path', 
#         help='path to the directory in which the data is located.')
# parser.add_argument('--type', help='type of data to load (ARD or CAU).')

# args = parser.parse_args()
# data = TurbineData(args.data_path, args.type)

# data.to_tensor()
# data.select_baseline()


def correlation_functions(data, D=10, filename='cor_func'):
    """
    Fits a pairwise interaction correlation function to each pair of turbines
    """
    N = data.n_turbines
    its = list(range(N))
    k_mat = np.empty((N, N), dtype='object')
    regressor = KernelRidge(kernel='laplacian', alpha=0.001, gamma=0.001)
    for i in (its):
        for j in (its[:i] + its[i+1:]):
            k_mat[i, j] = TransformedTargetRegressor(regressor=regressor, 
                    func=np.log1p, inverse_func=np.expm1)
    r_mat = np.zeros((N, N))
    
    for i in its:
        k_mat[i,i] = DummyPredictor()
        r_mat[i,i] = 1
    
    for i in tqdm(its):
        for j in tqdm(its[:i] + its[i + 1:]): # don't fit turbine j to turbine j
            data_copy = copy.deepcopy(data)
            data_copy.select_turbine([i, j])
            data_copy.select_normal_operation_times()

            x = data_copy.data[:, 1, 2] # reference power
            y = data_copy.data[:, 0, 2] # target power
            xa = data_copy.data[:, 1, 4] # reference angle
            ya = data_copy.data[:, 0, 4] # target angle

            X = np.stack((x, xa), axis=1)
            Y = np.stack((y, ya), axis=1)

            k_mat[i, j].fit(X[::D], Y[::D])
            r_mat[i, j] = k_mat[i, j].score(X, Y)

    dump(k_mat, filename + '.joblib')
    np.save(filename + '_r2.npy', r_mat)

    return k_mat, r_mat


# k_mat = load('cor_func_baseline_log_ARD.joblib')
# r_mat = np.load('cor_func_baseline_log_ARD_r2.npy')
# 
# plt.imshow(r_mat)
# plt.show()
# 
# average_power_gain_curve(data, k_mat)
# 
# I=3
# J=4
# 
# data_copy = copy.deepcopy(data)
# data_copy.select_turbine([I, J])
# data_copy.select_normal_operation_times()
# data_copy.select_unsaturated_times()
# 
# x = data_copy.data[:, 1, 2] # reference power
# y = data_copy.data[:, 0, 2] # target power
# xa = data_copy.data[:, 1, 4] # reference angle
# ya = data_copy.data[:, 0, 4] # target angle
# 
# X = np.stack((x, xa), axis=1)
# Y = np.stack((y, ya), axis=1)
# 
# r2 = k_mat[I, J].score(X, Y)
# ys = k_mat[I, J].predict(X)


def test_kernel_ridge(data):
    data.to_tensor()
    kr = KernelRidge(kernel='laplacian', alpha=0.001, gamma=0.001)
    krs = np.array([kr, kr, kr, kr, kr, kr], dtype='object')
    for i in range(6):
        data_copy = copy.deepcopy(data)
        data_copy.select_turbine([i, i+1])
        data_copy.select_normal_operation_times()
        data_copy.select_unsaturated_times()

        x = data_copy.data[:, 0, 2] # reference power
        y = data_copy.data[:, 1, 2] # target power
        xa = data_copy.data[:, 0, 4] # reference angle
        ya = data_copy.data[:, 1, 4] # target angle

        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

        X = np.stack((x, xa), axis=1)
        Y = np.stack((y, ya), axis=1)

        krs[i].fit(X[::10], Y[::10])
        r2 = krs[i].score(X, Y)
        ys = krs[i].predict(X)

        #measured vs predicted powers
        plt.scatter(x[::10], y[::10], label='Measured', alpha=0.05)
        plt.scatter(x[1::10], ys[1::10, 0], label='Prediction (on test data)', 
                alpha=0.05)
        plt.xlabel('Reference power')
        plt.legend()
        plt.ylabel('Target power')
        plt.show()

        plt.scatter(xa[::10], ya[::10], label='Measured', alpha=0.05)
        plt.scatter(xa[1::10], ys[1::10, 1], label='Prediction (on test data)',
                alpha=0.05)

        plt.xlabel('Reference angle')
        plt.legend()
        plt.ylabel('Target angle')
        plt.show()

        _, ax = plt.subplots(2, 1, sharex=True, sharey=True)

        ax[0].hist2d(xa, xa - ya, bins=100, range=[[np.min(xa), np.max(xa)], 
            [-20, 20]])
        ax[0].set_ylabel('Reference angle - measured target angle')
        ax[0].set_title('Measured angle correlation')

        ax[1].hist2d(xa, xa - ys[:, 1], bins=100,
                range=[[np.min(xa), np.max(xa)], [-20, 20]])
        ax[1].set_xlabel('Reference angle')
        ax[1].set_ylabel('Reference angle - predicted target angle')
        ax[1].set_title('Predicted angle correlation')
        plt.show()

        _, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].hist2d((xa + ya)/2, x - y, bins=100)
        ax[0].set_ylabel(r'$P_t-P_r$')
        ax[0].set_title('Measured power difference vs mean angle correlation')

        ax[1].hist2d((xa + ys[:, 1])/2, x - ys[:, 0], bins=100)
        ax[1].set_xlabel('Average angle of target/reference pair')
        ax[1].set_ylabel(r'$P_t-P_r$')
        ax[1].set_title('Predicted power difference vs mean angle correlation')
        plt.show()

        plt.hist2d(ya, y - ys[:, 0], bins=100)
        plt.xlabel('Measured target angle')
        plt.ylabel('Measured target power - predicted target power')
        plt.show()

        plt.scatter((xa[::10] + ya[::10])/2, x[::10] - y[::10], 
                label='Measured', alpha=0.05)
        plt.scatter((xa[::10] + ys[::10, 1])/2, x[::10] - ys[::10, 0], 
                label="Prediction", alpha=0.05)
        plt.xlabel('Average angle of target/reference pair')
        plt.ylabel(r'$P_t-P_r$')
        plt.title('Power difference vs mean angle correlation')
        plt.legend()
        plt.show()

        plt.hist2d(y, y - ys[:, 0], bins=100)
        plt.xlabel('Measured target power')
        plt.ylabel('Measured - predicted target power')
        plt.show()

