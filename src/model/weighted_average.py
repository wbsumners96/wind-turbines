from model.predictor import Predictor

import numpy as np


class WeightedAverage(Predictor):
    def __init__(self, weighting):
        self.weighting = weighting

    def predict(self, data, targets, references, times):
        """
        Predict the power of the target turbine at the specified time.

        The power is predicted by taking a weighted average of the powers of the
        given reference turbines at that time. The coefficients in the weighted
        average are given by a function of distance from the target turbine.

        Parameters
        ----------
        data : pd.DataFrame
            Wind turbine data.
        weighting : (distance: positive real float) -> positive real float
            Function that determines the coefficient of linear combination.
        targets : list of int
            ID of target turbine.
        references : list of int
            IDs of reference turbines.
        time : str
            Target timestamp to predict, with datetime format
            'DD-MMM-YYYY hh:mm:ss'.

        Returns
        -------
        target_power : numpy.ndarray (real numbers)
            True power output of target turbines at given time.
        predicted_power : numpy.ndarray (real numbers)
            Predicted power output of the target turbines.

        Raises
        ------
        ValueError
            If data type is not 'ARD' or 'CAU'.
        """
        # Generate string IDs of turbines
        # First learn the type of the data
        data = data.data
        first_id = data['instanceID'][0]
        if first_id.startswith('ARD'):
            type = 'ARD'
        elif first_id.startswith('CAU'):
            type = 'CAU'
        else:
            raise ValueError('Data is of an unexpected type.')

        # Target_id = f'{type}_WTG{target:02d}'
        target_ids = [f'{type}_WTG{target:02d}' for target in targets]
        reference_ids = [f'{type}_WTG{reference:02d}' 
                         for reference in references]

        # Restrict data to given time and separate into targets and references
        print(times)
        current_data = data.query('ts == @times')
        print(current_data)
        target_data = current_data.query('instanceID == @target_ids')
        reference_data = current_data.query('instanceID == @reference_ids')

        # Get vector of distances from target turbines to reference turbines
        target_positions = target_data[['Easting', 'Northing']].to_numpy()
        reference_positions = reference_data[['Easting', 'Northing']].to_numpy()

        distances = np.sqrt(np.sum((target_positions[:,np.newaxis,:]
                                   - reference_positions) ** 2, axis=-1))
        # Get vector of weights
        weights = np.vectorize(self.weighting)(distances)
        
        # Calculate predicted power as w_1 f(p_1) + ... + w_n f(p_n)
        target_powers = target_data['Power'].to_numpy()
        reference_powers = reference_data['Power'].to_numpy()
        predicted_powers = np.einsum('ij, j->i', weights, reference_powers) \
                           / np.sum(weights, axis=1)

        return target_powers, predicted_powers


class GaussianWeightedAverage(WeightedAverage):
    def __init__(self, gamma):
        def weighting(distance):
            return np.exp(-gamma*distance*distance)

        super().__init__(weighting)
