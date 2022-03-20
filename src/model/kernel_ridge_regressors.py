from abc import abstractmethod
from datetime import datetime
import math
from joblib import dump, load
import numpy as np
import os
import pandas as pd
from pathlib import Path
from pandas.core.reshape.merge import merge
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .predictor import Predictor


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

    def predict(self, data, targets, references, times=None):
        target_ids = [f'{data.farm}_WTG{target_number:02}'
                      for target_number
                      in targets]
        reference_ids = [f'{data.farm}_WTG{reference_number:02}'
                         for reference_number 
                         in references]

        targets = data.data.query('instanceID == @target_ids')
        references = data.data.query('instanceID == @reference_ids')

        targets = targets[['ts',
                           'instanceID',
                           'Power',
                           'Wind_direction_calibrated']]
        targets.rename(columns={'instanceID': 'target_id',
                                'Power': 'target_power',
                                'Wind_direction_calibrated': 'target_angle'},
                       inplace=True)

        references = references[['ts',
                                 'instanceID', 
                                 'Power',
                                 'Wind_direction_calibrated']]
        references.rename(columns={'instanceID': 'reference_id',
                                   'Power': 'reference_power',
                                   'Wind_direction_calibrated': 
                                           'reference_angle'},
                          inplace=True)

        merged_data = pd.merge(targets, references, on='ts')
        merged_data['predicted_power'] = 0
        
        for target_id in tqdm(target_ids, desc='Target', leave=False):
            regressors_path = Path('~/.turbines/regressors/').expanduser() \
                    / f'{target_id}_kernel_ridge_regressors.joblib'
            regressors = load(regressors_path)

            for reference_id in tqdm(reference_ids, desc='Reference',
                    leave=False):
                if target_id == reference_id:
                    predictions = merged_data['target_power']
                else:
                    select_data = merged_data.query('target_id == @target_id '+\
                            'and reference_id == @reference_id')

                    reference_powers = select_data['reference_power'].to_numpy()
                    target_angle = select_data['target_angle'].to_numpy()
                    reference_angle = select_data['reference_angle'].to_numpy()

                    features = np.column_stack([reference_powers,
                                                target_angle,
                                                reference_angle])
                    training_features = self.features_train[target_id] \
                            [reference_id]

                    kernel_gram = self.kernel(training_features, features)

                    regressor = regressors[reference_id]
                    predictions = regressor.predict(kernel_gram)

                mask = (merged_data['target_id'] == target_id) & \
                       (merged_data['reference_id'] == reference_id)
                merged_data.loc[mask, 'predicted_power'] = predictions

        merged_data.drop(columns=['reference_power', 'target_angle', 'reference_angle'],
                         inplace=True)

        if self.aggregation == 'r2':
            weightings = {}
            for target_id in target_ids:
                r2_path = Path('~/.turbines/scores/').expanduser() / \
                        f'{target_id}_kernel_ridge_scores.joblib'
                target_r2_scores = load(r2_path)
                target_r2_scores = { reference_id : target_r2_scores[reference_id] for
                        reference_id in reference_ids }

                target_weightings = { key: np.exp(value) for key, value in
                        target_r2_scores.items() }

                weightings[target_id] = target_weightings

            merged_data['weighting'] = 0
            for target_id in target_ids:
                for reference_id in reference_ids:
                    mask = (merged_data['target_id'] == target_id) & \
                           (merged_data['reference_id'] == reference_id)
                    merged_data.loc[mask, 'weighting'] = \
                            weightings[target_id][reference_id]

            table = merged_data.pivot(index=['ts', 'target_id'],
                                      columns=['reference_id'],
                                      values=['target_power', 'predicted_power', 'weighting'])
            
            predicted_powers = table['predicted_power'].to_numpy(na_value=0.0)
            weightings = table['weighting'].to_numpy(na_value=0.0)

            predicted_powers = np.average(predicted_powers, weights=weightings, axis=1)

            table['predicted_power'] = predicted_powers

            table = table.stack(['reference_id'])
            table.dropna(inplace=True)
            table = table.reset_index()   

            table.drop(columns=['reference_id', 'weighting'], inplace=True)

            return table
        else:
            return merged_data

    def __init__(self, aggregation='none'):
        """
        A model which fits functions of the form
            f(x) = sum a_j k(x, x_j)
        for some kernel k.

        Parameters
        ----------
        aggregation : 'none', 'r2', 'lm'
            Determine how to aggregate the pairwise predictions into a single
            prediction on a target. If 'none', no aggregation is done. If 'r2',
            takes a weighted average with weights proportional to the
            exponential of the R^2 score of the pairwise predictions. If 'lm',
            an additional step of fitting a linear function of the form
                target_power = sum a_j f_j(reference_power_j, target_angle,
                        reference_angle_j)
            is done.
        """
        self.aggregation = aggregation

        features_train_path = \
                Path('~/.turbines/training_features.joblib') \
                        .expanduser()
        if features_train_path.is_file():
            self.features_train = load(features_train_path)

    def fit(self, data):
        """
        Fit a collection of kernel ridge models of the form target_power = 
        f(reference_power, target_angle, reference_angle) for each pair of
        target and reference turbines in the farm.
        """
        self.features_train = {}
        turbines_dir = Path('~/.turbines').expanduser()
        if not turbines_dir.is_dir():
            os.mkdir(turbines_dir)

        regressor_dir = Path('~/.turbines/regressors').expanduser()
        if not regressor_dir.is_dir():
            os.mkdir(regressor_dir) 

        scores_dir = Path('~/.turbines/scores/').expanduser()
        if not scores_dir.is_dir():
            os.mkdir(scores_dir)

        turbine_ids = data.data['instanceID'].drop_duplicates()
        for target_id in tqdm(turbine_ids, desc='Target', leave=False):
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
            for reference_id in tqdm(turbine_ids,
                                     desc='Reference',
                                     leave=False):
                if target_id == reference_id:
                    target_regressors[reference_id] = None
                    target_scores[reference_id] = 1.0

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

            target_regressors_path = Path('~/.turbines/regressors/' + \
                    f'{target_id}_kernel_ridge_regressors.joblib') \
                    .expanduser()
            target_scores_path = Path('~/.turbines/scores/' + \
                    f'{target_id}_kernel_ridge_scores.joblib') \
                    .expanduser()
            dump(target_regressors, target_regressors_path)
            dump(target_scores, target_scores_path)

            self.features_train[target_id] = target_features_train

        training_features_path = Path('~/.turbines/' + \
                'training_features.joblib') \
                .expanduser()
        dump(self.features_train, training_features_path)

    def scores(self, data):
        scores = {}
        target_ids = data.data['instanceID'].drop_duplicates()
        for target_id in target_ids:
            target_scores_path = Path('~/.turbines/scores/' + \
                    f'{target_id}_kernel_ridge_scores.joblib') \
                    .expanduser()
            target_scores = load(target_scores_path)

            scores[target_id] = target_scores

        return scores


class LaplacianKRR(KernelRidgeRegressor):
    def __init__(self, aggregation):
        if math.isclose(aggregation, 1.0):
            aggregation = 'r2'
        else:
            aggregation = 'none'

        super().__init__(aggregation)

    def kernel(self, x_i, x_j):
        return np.exp(-0.01*np.linalg.norm(x_i[np.newaxis, :, :] - x_j[:,
                np.newaxis, :], ord=1, axis=2))


class PowerLaplacianKRR(KernelRidgeRegressor):
    def __init__(self, aggregation):
        if math.isclose(aggregation, 1.0):
            aggregation = 'r2'
        else:
            aggregation = 'none'

        super().__init__(aggregation)

    def kernel(self, x_i, x_j):
        power_ref_i, _, _ = x_i.T
        power_ref_j, _, _ = x_j.T

        def laplacian_kernel(power_i, power_j):
            return np.exp(-0.01*np.abs(power_i - power_j))

        return laplacian_kernel(power_ref_i, power_ref_j.reshape((-1, 1)))


class RadialBasisKRR(KernelRidgeRegressor):
    def __init__(self, aggregation):
        if math.isclose(aggregation, 1.0):
            aggregation = 'r2'
        else:
            aggregation = 'none'

        super().__init__(aggregation)

    def kernel(self, x_i, x_j):
        return np.exp(-0.01*np.linalg.norm(x_i[np.newaxis, :, :] - x_j[:,
            np.newaxis, :], axis=2))


class PeriodicLaplacianKRR(KernelRidgeRegressor):
    def __init__(self, aggregation):
        if math.isclose(aggregation, 1.0):
            aggregation = 'r2'
        else:
            aggregation = 'none'

        super().__init__(aggregation)

    def kernel(self, x_i, x_j):
        def periodic_kernel(theta_i, theta_j, 
                            variance=1, 
                            length=1, 
                            period=2*np.pi):
            angle = (np.pi/period)*np.abs(theta_i - theta_j)
            z = (-2/length*length)*(np.sin(angle)**2)

            return variance*np.exp(z)

        def laplacian_kernel(power_i, power_j):
            return np.exp(-0.01*np.abs(power_i - power_j))

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

        return power_cov * theta_ref_cov * theta_tar_cov

