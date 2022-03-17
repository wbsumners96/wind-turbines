from math import dist
from model.predictor import Predictor

import numpy as np
import pandas as pd


class WeightedAverage(Predictor):
    def __init__(self, weighting, reference_column=False):
        self.reference_column = reference_column
        self.weighting = weighting

    def fit(self, data):
        pass

    def predict(self,data,targets,references,times=None):
        """
        Calls predict_tensor or predict_pd depending on datatype
        """
        if data.data_type == "pd.DataFrame":
            return self.predict_pd(data, targets, references, None)
        if data.data_type == "np.ndarray":
            return self.predict_tensor(data, targets, references)

    def predict_tensor(self,data,tar_mask,ref_mask,verbose=False):
        """
        Predict the power of the specified wind turbines.
        Needs data as a numpy array, parallel over time axis
        

        Parameters
        ----------
        data : TurbineData (with numpy.ndarray data)
            Wind turbine data.
        targets : list of int
            ID of target turbine.
        references : list of int
            IDs of reference turbines.
        verbose : bool
            Choose whether to display heatmap of weight matrix

        Returns
        -------
        pred_power : numpy.ndarray
            Estimated power output of the target turbines.
        tar_power : numpy.ndarray
            Measured power output of the target turbines.
        """

        if data.data_type != 'np.ndarray':
            raise TypeError('Data must be numpy array, run .to_tensor() first')
        data = data.data

        print(np.shape(data))
        print(np.shape(data[:,tar_mask]))
        print(np.shape(data[:,ref_mask]))

        tars = data[:,tar_mask]
        refs = data[:,ref_mask]

        if not np.all(tars[:,-1]):
            print("Warning: some target turbines are faulty")
        if not np.all(refs[:,-1]):
            print("Warning: some reference turbines are faulty")

        # Position data
        tar_pos = tars[0,:,5:6] # turbines don't move
        ref_pos = refs[0,:,5:6]
        # Power data
        tar_power = tars[:,:,2]
        ref_power = refs[:,:,2]

        # Calculate euclidean distance between all target-reference pairs
        ds = np.sqrt(np.sum((tar_pos[:,np.newaxis,:]-ref_pos)**2,axis=-1))

        ws = np.vectorize(self.weighting)(ds)
        if verbose:
            plt.imshow(ws)
            plt.title('Weight matrix')
            plt.show()

        def f(power):
            # Dummy function to change later if we want something more complex 
            return power

        vf = np.vectorize(f)
        pred_power = np.einsum('ij, kj->ki', ws, ref_power)/np.sum(ws, axis=1)
        
        return pred_power, tar_power

    def predict_pd(self, data, targets, references, times):
        """
        Predict the power of the target turbine at the specified time.
    
        The power is predicted by taking a weighted average of the powers of the
        given reference turbines at that time. The coefficients in the weighted
        average are given by a function of distance from the target turbine.

        Parameters
        ----------
        data : TurbineData (with pd.DataFrame data)
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
        predictions : pd.DataFrame
            Dataframe with columns 'ts' representing timestamp, 'target_id'
            giving the ID of the target turbine, 'target_power' giving the true
            power of the target turbine at that time, and 'predicted_power'
            giving the predicted power of the target turbine at that time.

        Raises
        ------
        ValueError
            If data type is not 'ARD' or 'CAU'.
        """
        # Generate string IDs of turbines
        # First learn the type of the data
        if data.data_type != 'pd.DataFrame':
            raise TypeError('Data must be pandas dataframe')

        data = data.data
        first_id = data['instanceID'].iloc[0]
        if first_id.startswith('ARD'):
            type = 'ARD'
        elif first_id.startswith('CAU'):
            type = 'CAU'
        else:
            raise ValueError('Data is of an unexpected type.')

        target_ids = [f'{type}_WTG{target:02d}' for target in targets]
        reference_ids = [f'{type}_WTG{reference:02d}' 
                         for reference in references]

        # Restrict data to given time and separate into targets and references
        if times is not None:
            data = data.query('ts == @times')

        target_data = data.query('instanceID == @target_ids')
        reference_data = data.query('instanceID == @reference_ids')

        target_data = target_data[['ts',
                               'instanceID',
                               'Power',
                               'Easting',
                               'Northing']]
        target_data.rename(columns={'instanceID': 'target_id',
                                    'Power': 'target_power',
                                    'Easting': 'target_easting',
                                    'Northing': 'target_northing'},
                           inplace=True)

        reference_data = reference_data[['ts', 
                                 'instanceID', 
                                 'Power',
                                 'Easting',
                                 'Northing']]
        reference_data.rename(columns={'instanceID': 'reference_id',
                                   'Power': 'reference_power',
                                   'Easting': 'reference_easting',
                                   'Northing': 'reference_northing'},
                          inplace=True)

        merged_data = pd.merge(target_data, reference_data, on='ts')
        merged_data['weighting'] = 0

        def distance(row):
            target_position = np.array([row['target_easting'],
                                        row['target_northing']])
            reference_position = np.array([row['reference_easting'],
                                           row['reference_northing']])

            displacement = target_position - reference_position
            row['weighting'] = self.weighting(np.linalg.norm(displacement))

            return row

        merged_data = merged_data.apply(distance, axis=1)
        merged_data.drop(columns=['target_easting',
                                  'target_northing',
                                  'reference_easting',
                                  'reference_northing'],
                         inplace=True)
        
        merged_data['weighted_power'] = merged_data['weighting'] * \
                                        merged_data['reference_power']
        merged_data.drop(columns=['reference_power', 'weighting'], inplace=True)

        table = pd.pivot_table(merged_data, index=['ts', 'target_id'],
                aggfunc=np.average)

        table = pd.DataFrame(table.to_records())
        table.rename({'weighted_power': 'predicted_power'}, inplace=True)

        if self.reference_column:
            table['reference_id'] = None

        return table


class GaussianWeightedAverage(WeightedAverage):
    def __init__(self, gamma):
        def weighting(distance):
            return np.exp(-gamma*distance*distance)

        super().__init__(weighting)

