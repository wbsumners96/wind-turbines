from abc import abstractmethod
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .predictor import Predictor


def periodic_kernel(theta_i, theta_j, variance=1, length=1, period=2*np.pi):
    angle = (np.pi/period)*np.abs(theta_i - theta_j)
    z = (-2/length*length)*(np.sin(angle)**2)

    return variance*np.exp(z)

def linear_kernel(power_i, power_j):
    return power_i*power_j

def laplacian_kernel(power_i, power_j):
    return np.exp(-0.01*np.linalg.norm(power_i - power_j, ord=1))

def turbine_kernel(x_i, x_j):
    power_ref_i, theta_ref_i, theta_tar_i = x_i.T
    power_ref_j, theta_ref_j, theta_tar_j = x_j.T

    # power_cov = linear_kernel(power_ref_i, power_ref_j.reshape((-1, 1)))
    power_cov = laplacian_kernel(power_ref_i, power_ref_j.reshape((-1, 1)))
    theta_ref_cov = periodic_kernel(theta_ref_i,
                                    theta_ref_j.reshape((-1, 1)),
                                    length=10)
    theta_tar_cov = periodic_kernel(theta_tar_i,
                                    theta_tar_j.reshape((-1, 1)),
                                    length=10)

    # return np.exp(-0.01*np.linalg.norm(x_i[np.newaxis, :, :] - x_j[:,
    #   np.newaxis, :], ord=1, axis=2))
    
    return power_cov*theta_ref_cov*theta_tar_cov


class KernelRidgeRegressor(Predictor):
    @abstractmethod
    def kernel(self, x_i, x_j):
        """
        Compute kernel matrix given two lists of features. 

        Parameters
        ----------
        x_i : ndarray of shape (n_samples, n_features)
            Array of features.
        x_j : ndarray of shape (m_samples, n_features) 
            Array of features.

        Returns
        -------
        k_ij : ndarray of shape (n_samples, m_samples)
            Kernel matrix.
        """
        raise NotImplementedError()

    def predict(self, data, targets, references, times):
        prediction_data = {'ts': [],
                           'target_id': [],
                           'reference_id': [],
                           'target_power': [],
                           'predicted_power': []}
        for target_number in targets:
            target_id = f'ARD_WTG{target_number:02}' 
            target = data.select_turbine(target_id)
            
            target_data = target[['ts',
                                  'instanceID', 'Power',
                                  'Wind_direction_calibrated']]
            target_data.rename({'instanceID': 'target_id',
                                'Power': 'target_power',
                                'Wind_direction_calibrated': 'target_angle'},
                               inplace=True)

            target_regressors = load(f'regressors/' + \
                    '{target_number}_kernel_ridge_regressors.joblib')
            for reference_number in references:
                reference_id = f'ARD_WTG{reference_number:02}'
                reference = data.select_turbine(reference_id)
                
                reference_data = reference[['ts',
                                            'instanceID', 
                                            'Power',
                                            'Wind_direction_calibrated']]
                reference_data.rename({'instanceID': 'reference_id',
                                       'Power': 'reference_power',
                                       'Wind_direction_calibrated':
                                               'reference_angle'},
                                      inplace=True)

                merged_data = pd.merge(target_data, reference_data, on='ts')
                ts = merged_data['ts'].to_numpy()
                target_id = merged_data['target_id'].to_numpy()
                reference_id = merged_data['reference_id'].to_numpy()
                target_power = merged_data['target_power'].to_numpy()
                reference_power = merged_data['reference_power'].to_numpy()
                target_angle = merged_data['target_angle'].to_numpy()
                reference_angle = merged_data['reference_angle'].to_numpy()

                features = np.column_stack([reference_power,
                                            target_angle,
                                            reference_angle])
                kernel_gram = self.kernel(features, self.features_train)

                regressor = target_regressors[reference_id]
                predicted_power = regressor.predict(kernel_gram)

                prediction_data['ts'].extend(ts)
                prediction_data['target_id'].extend(target_id)
                prediction_data['reference_id'].extend(reference_id)
                prediction_data['target_power'].extend(target_power)
                prediction_data['predicted_power'].extend(predicted_power)

        return pd.DataFrame(prediction_data)

    def __init__(self, data):
        """
        Fit a collection of kernel ridge models of the form target_power = 
        f(reference_power, target_angle, reference_angle) for each pair of target
        and reference turbines in the farm.
        """
        target = data.select_turbine('ARD_WTG01')
        target = target[['ts', 'instanceID', 'Power', 'Wind_direction_calibrated']]
        target.rename(columns={'instanceID': 'target_id',
                               'Power': 'target_power',
                               'Wind_direction_calibrated': 'target_angle'},
                      inplace=True)

        self.features_train = {}
        for target_number in tqdm(range(1, 16), desc='Target'):
            target_id = f'ARD_WTG{target_number:02}'
            target = data.select_turbine(target_id)
            target = target[['ts',
                             'instanceID',
                             'Power',                 
                             'Wind_direction_calibrated']]
            target.rename(columns={'instanceID': 'target_id',
                                   'Power': 'target_power',
                                   'Wind_direction_calibrated': 'target_angle'},
                             inplace=True)

            target_regressors = {}
            target_scores = {}
            target_features_train = {}
            for reference_number in tqdm(range(1, 16), desc='Reference'):
                reference_id = f'ARD_WTG{reference_number:02}'
                if target_number == reference_number:
                    target_regressors[reference_id] = None
                    target_scores[reference_id] = None

                    continue

                reference = data.select_turbine(reference_id)
                reference = reference[['ts',
                                       'instanceID',
                                       'Power',                 
                                       'Wind_direction_calibrated']]
                reference.rename(columns={'instanceID': 'reference_id',
                                          'Power': 'reference_power',
                                          'Wind_direction_calibrated':
                                                  'reference_angle'},
                                 inplace=True)

                merged_data = pd.merge(target, reference, on='ts')

                # labels for ml.
                target_power = merged_data['target_power'].to_numpy()
                label = target_power

                # features for ml.
                target_angle = merged_data['target_angle'].to_numpy()
                reference_power = merged_data['reference_power'].to_numpy()
                reference_angle = merged_data['reference_angle'].to_numpy()
                features = np.column_stack([reference_power,
                                            target_angle,
                                            reference_angle])
                
                features_train, features_test, label_train, label_test = \
                        train_test_split(features, label)
                target_features_train[reference_id] = features_train

                kernel_train = self.kernel(features_train, features_train)
                kernel_test = self.kernel(features_train, features_test)

                kernel_ridge_regressor = KernelRidge(kernel='precomputed',
                                                     alpha=0.001)
                regressor = TransformedTargetRegressor(
                        regressor=kernel_ridge_regressor,
                        func=np.log1p,
                        inverse_func=np.expm1)
                regressor.fit(kernel_train, label_train)

                target_regressors[reference_id] = regressor
                target_scores[reference_id] = regressor.score(kernel_test,
                                                              label_test)

            dump(target_regressors,
                 f'regressors/{target_number}_kernel_ridge_regressors.joblib')
            dump(target_scores, 
                 f'scores/{target_number}_kernel_ridge_scores.joblib')

            self.features_train[target_id] = target_features_train


class LaplacianKRR(KernelRidgeRegressor):
    def kernel(self, x_i, x_j):
        return np.exp(-0.01*np.linalg.norm(x_i[np.newaxis, :, :] - x_j[:,
                np.newaxis, :], ord=1, axis=2))

