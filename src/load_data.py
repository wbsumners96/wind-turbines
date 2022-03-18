from datetime import datetime
from dateutil import parser
import os
from pathlib import Path

import pandas as pd
import numpy as np 
from tqdm import tqdm


class TurbineData:
    def __init__(self, path, farm):
        """
        Class constructor to load wind turbine data, relative position data
		and normal operation flags.
        
        Parameters
        ----------
        path : str
            The path to the directory in which the data is located.
        type : str
            Which data to load, either 'ARD' or 'CAU'.

        Throws
        ------
        TypeError
            If type is not 'ARD' or 'CAU'.
        """
        if farm not in ('ARD', 'CAU'):
            raise TypeError('Argument \'farm\' must be \'ARD\' or '
                            + '\'CAU\'.')

        data_file = farm + '_Data.csv'
        data = pd.read_csv(os.path.join(path, data_file))
        assert type(data) is pd.DataFrame

        pos_file = farm + '_Turbine_Positions.csv'
        pos = pd.read_csv(os.path.join(path, pos_file))

        data_joined = pd.merge(data, 
                               pos, 
                               left_on=['instanceID'],
                               right_on=['Obstical'])
        flag_file = farm + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        data_complete = pd.merge(data_joined, 
                                 normal_operation,
                                 on=['ts', 'instanceID'])

        self.farm = farm
        if self.farm == 'ARD':
            self.n_turbines = 15
        elif self.farm == 'CAU':
            self.n_turbines = 21

        self.data = data_complete
        self.data_type = 'pd.DataFrame'

    def to_tensor(self):
        """
        Converts pd.dataframe to 2 rank 3 tensors.

        One tensor is a float containing all the actual data,
        one is of strings containing labels of datatime and turbine. Rank 3
		tensors are indexed by time, turbine and attribute.
        ----------
        self.data is now an array indexed as follows:
        data[time,turbine,attribute]
        Attributes index order:
            0 - Turbulence Intensity
            1 - Wind Speed
            2 - Power
            3 - ambient temperature
            4 - Wind direction callibrated
            5 - Easting
            6 - Northing
            7 - Hub Height
            8 - Diameter
            9 - Altitude
            10 - Normal operation flag
        self.data_labels is indexed as follows:
        data_labels[time,turbine,label]
        label index:
            0 - datatime
            1 - instanceID
        """
        data_numpy = self.data.to_numpy()
        if self.farm == 'ARD':
            n_turbines = 15
        elif self.farm == 'CAU':
            n_turbines = 21
        
        data_tensor = data_numpy.reshape((n_turbines, -1, 15))
        data_tensor = np.einsum('ijk->jik', data_tensor)

        mask = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], 
                dtype=bool)
        self.data = data_tensor[:, :, mask].astype(float)
        self.data_label = data_tensor[:, :, :2]
        self.data_type = 'np.ndarray'

    def to_dataframe(self):
        """
        Converts tensor form to dataframe.
        """
        if self.data_type != 'pd.DataFrame':
            data = np.dstack((self.data, self.data_label))
            data = np.einsum('jik->ijk', data)
            data = data.reshape((-1, 13))

            loctype = ['EastNorth'] * data.shape[0]
            self.data = pd.DataFrame({'ts': data[:, 11],
                                      'instanceID': data[:, 12],
                                      'TI': data[:, 0],
                                      'Wind_speed': data[:, 1],
                                      'Power': data[:, 2],
                                      'Ambient_temperature': data[:, 3],
                                      'Wind_direction_calibrated': data[:, 4],
                                      'Obstical': data[:, 12],
                                      'LocType': loctype,
                                      'Easting': data[:, 5],
                                      'Northing': data[:, 6],
                                      'HubHeight': data[:, 7],
                                      'Diameter': data[:, 8],
                                      'Altitude': data[:, 9],
                                      'value': data[:, 10]})

            self.data_type = 'pd.DataFrame'

    def select_baseline(self, inplace=False):
        """
        Selects data before the configuration changes
        For ARD:  < 01/06/2020
        For CAU:  < 30/06/2020
        """
        if self.data_type == 'pd.DataFrame':
            self.data['ts'] = pd.to_datetime(self.data['ts'], 
                                             format='%d-%b-%Y %H:%M:%S')

            if self.farm == 'ARD':
                baseline = parser.parse('01-Jun-2020 00:00:00')
            elif self.farm == 'CAU':
                baseline = parser.parse('30-Jun-2020 00:00:00')

            timestamps = self.data['ts'] <= baseline
            baseline_data = self.data[timestamps]
            
            if inplace:
                self.data = baseline_data

            return baseline_data
        elif self.data_type == 'np.ndarray':
            if not inplace:
                print('Selecting baseline configuration datetimes for \
                        numpy arrays has not been implemented in the non- \
                        inplace case')

                return
            if self.farm == 'ARD':
                i = np.argwhere(self.data_label[:, 0, 0] 
                        == '01-Jun-2020 00:00:00')[0, 0]
            elif self.farm == 'CAU':
                i = np.argwhere(self.data_label[:, 0, 0]
                        == '30-Jun-2020 00:00:00')[0,0]

            self.data = self.data[:i]
            self.data_label = self.data_label[:i]

    def select_new_phase(self):
        """
        Selects data before the configuration changes
        For ARD:  < 01/06/2020
        For CAU:  < 30/06/2020
        """
        if self.data_type == 'pd.DataFrame':
            print('Not yet implemented for DataFrame')
            
            return
        elif self.data_type == 'np.ndarray':
            if self.farm == 'ARD':
                condition = self.data_label[:, 0, 0] == '10-Sep-2020 00:00:00'
                i = np.argwhere(condition)[0,0]
            elif self.farm == 'CAU':
                condition = self.data_label[:, 0, 0] == '19-Oct-2020 00:00:00'
                i = np.argwhere(condition)[0,0]

            self.data = self.data[i:]
            self.data_label = self.data_label[i:]

    def select_time(self, time, verbose=False):
        """
        Return the data for all wind turbines at the specified time.

        Parameters
        ----------
        time : str
        verbose : bool, default=False

        Returns
        -------
        pd.Series
        """
        data = self.data
        if verbose:
            print('Selected time: ' + str(data.ts[time]))
        if self.data_type == 'pd.DataFrame':
            return data[data.ts == data.ts[time]]
        elif self.data_type == 'np.ndarray':
            self.data = data[time]
            self.data_label = self.data_label[time]

    def select_turbine(self, turbine, inplace=False, verbose=False):
        """
        Return the data for one wind turbine (or a set of turbines) across all 
        times.

        Parameters
        ----------
        time : str
        verbose : bool, default=False

        Returns
        -------
        pd.Series or numpy array, depending on current type of data stored in 
        class
        """
        data = self.data
        if verbose:
            print(f'Selected turbine {data.instanceID[turbine]}')

        if self.data_type == 'pd.DataFrame':
            turbine_data = data[data.instanceID == turbine]
            if inplace:
                self.data = turbine_data

            return turbine_data
        elif self.data_type == 'np.ndarray':
            turbine_data = data[:, turbine]
            turbine_data_label = self.data_label[:, turbine]
            if inplace:
                self.data = turbine_data
                self.data_label = turbine_data_label

            return turbine_data, turbine_data_label

    def select_wind_direction(self, direction, width, verbose=False):
        """
        Selects only times when the average wind direction is 
        within "width" of "direction"

        Parameters
        ----------
        direction : float (0,360)
            Average wind direction desired
        width : float 
            How far from the average wind direction is okay
        verbose : bool
            Print things
        """  
        if self.data_type == 'pd.DataFrame':
            raise TypeError('Data needs to be in numpy array, call .to_tensor()\
                    first')

        wind_dir_mean = np.mean(self.data[:, :, 4], axis=1)
        diff = np.abs(wind_dir_mean - direction)
        # correct for modular nature of angles
        wind_diff = np.minimum(360 - diff, diff) 
        wind_mask = wind_diff < width

        self.data = self.data[wind_mask]
        self.data_label = self.data_label[wind_mask]

    def select_normal_operation_times(self, verbose=False):
        """
        Removes times where any turbine is not functioning normaly.
        Best to run after running select turbines
        """
        if self.data_type == 'np.ndarray':
            if verbose:
                print(f'Data shape before selecting normal operation turbines: \
                        {self.data.shape}')

            flag = np.all((self.data[:, :, -1]).astype(bool), axis=1)
            self.data = self.data[flag]
            self.data_label = self.data_label[flag]

            if verbose:
                print(f'Data shape after selecting normal operation turbines: \
                        {self.data.shape}')
        elif self.data_type == 'pd.DataFrame':
            self.data.query('value == 1', inplace=True)

    def select_unsaturated_times(self, cutoff=1900, verbose=False):
        """
        Removes times where any turbine is obviously saturated in the power 
        curve.
        Best to run after running select turbines
        """
        if self.data_type == 'np.ndarray':
            if verbose:
                print(f'Data shape before selecting unsaturated turbines: \
                        {self.data.shape}')

            flag = np.all((self.data[:, :, 2] < cutoff).astype(bool), axis=1)
            self.data = self.data[flag]
            self.data_label = self.data_label[flag]

            if verbose:
                print(f'Data shape after selecting unsaturated turbines: \
                        {self.data.shape}')
        elif self.data_type == 'pd.DataFrame':
            unsaturated = self.data['Power'] < cutoff
            self.data = self.data[unsaturated]

    def select_power_min(self,cutoff=10,verbose=False):
        """
        Removes times where any turbine has very low power.
        Best to run after running select turbines
        """
        if verbose:
            print(f'Data shape before selecting higher power turbines: \
                    {self.data.shape}')

        flag = np.all((self.data[:, :, 2] > cutoff).astype(bool), axis=1)
        self.data = self.data[flag]
        self.data_label = self.data_label[flag]
        if verbose:
            print(f'Data shape after selecting higher power turbines: \
                    {self.data.shape}')

    def nan_to_zero(self):
        """
        Sets NANs to 0 - risky
        """
        self.data = np.nan_to_num(self.data)

    def clear_wake_affected_turbines(self):
        """
        Remove all data points of turbines lying in the wake of  
        non-operational turbines.

        That is, if at time t, turbine A is non-operational, and turbine B lies
        in the wake of turbine A according to IEC 61400-12-2 with the wind
        heading set to be that measured by turbine A, then the data point of 
        turbine B at time t will be removed.
        """
        def lies_in_wake(turbine_position, other_position, wind_heading,
                blade_diameter=80):
            """
            Determine if turbine 'other' lies in the wake of turbine 'turbine'
            according to IEC 61400-12-2.

            Parameters
            ----------
            turbine_position: length 2 numpy vector
                Position of turbine, with easting (in meters) in the first
                component and northing (in meters) in the second.

            other_position: length 2 numpy vector
                Position of other.

            wind_heading: float
                Direction of the wind in degrees from north.

            blade_diameter: float (default = 80)
                Diameter of turbine blades in meters.
            """
            def iec_function(blade_diameters):
                return 1.3*np.arctan(2.5/blade_diameters + 0.15)\
                        + np.pi*10/180

            wind_vector = np.array([np.sin(np.pi*wind_heading/180),
                                    np.cos(np.pi*wind_heading/180)])
            turbine_displacement = other_position - turbine_position
            angle_to_wind = np.arccos(np.dot(turbine_displacement, 
                    wind_vector)/np.linalg.norm(turbine_displacement))

            turbine_distance = np.linalg.norm(turbine_displacement)\
                    /blade_diameter 

            if turbine_distance <= 2:
                return True
            if turbine_distance > 20:
                return False

            return angle_to_wind <= iec_function(turbine_distance)

        if self.data_type != 'pd.DataFrame':
            print('Convert to a pandas DataFrame using self.to_dataframe().')

            return

        turbines_path = Path('~/.turbines').expanduser()
        if not turbines_path.is_dir():
            os.mkdir(turbines_path)

        wake_affected_path = Path('~/.turbines/wake_affected').expanduser()
        if not wake_affected_path.is_dir():
            os.mkdir(wake_affected_path)
        else:
            self.merge_wake_affected_data()

            return

        non_operational = self.data[self.data.value == 0]
        non_operational = non_operational[['ts', 'instanceID', 'Easting',
            'Northing', 'Diameter', 'Wind_direction_calibrated']]
        turbine_positions = {'other_id': self.data['instanceID'],
                             'other_easting': self.data['Easting'],
                             'other_northing': self.data['Northing']}
        turbine_positions = pd.DataFrame(turbine_positions).drop_duplicates()

        # an unfortunate hack to perform a cross join.
        non_operational['cross'] = 0
        turbine_positions['cross'] = 0
        df = pd.merge(non_operational, turbine_positions, on='cross')
        df['affected'] = 0

        def f(row):
            if row['instanceID'] == row['other_id']:
                return row

            turbine_position = np.array([row['Easting'], row['Northing']])
            other_position = np.array([row['other_easting'],
                                       row['other_northing']])
            blade_diameter = row['Diameter']
            wind_heading = row['Wind_direction_calibrated']

            if lies_in_wake(turbine_position,
                            other_position,
                            wind_heading,
                            blade_diameter):
                row['affected'] = 1

            return row

        dfs = np.array_split(df, 200)
        # print(dfs)
        fr = 0
        for frame in tqdm(dfs, leave=False):
            fr = fr + 1

            frame = frame.apply(f, axis=1)
            # print(frame)
            frame = frame[frame['affected'] == 1]
            frame_prime = frame[['ts', 'other_id']]
            
            frame_prime.to_csv(f'~/.turbines/wake_affected/{self.farm}_{fr}_wake_affected.csv')
            print(f'Saved frame {fr}')

        self.merge_wake_affected_data()

    def merge_wake_affected_data(self):
        dfs = []
        for fr in range(1, 201):
            dfs.append(pd.read_csv('~/.turbines/wake_affected/' + \
                    f'{self.farm}_{fr}_wake_affected.csv'))

        df = pd.concat(dfs)
        df.drop(columns=['Unnamed: 0'])

        self.data = pd.merge(self.data, df,
                             left_on=['ts', 'instanceID'],
                             right_on=['ts', 'other_id'],
                             how='outer',
                             indicator=True) \
                      .query('_merge == "left_only"') \
                      .drop(['_merge', 'other_id', 'Unnamed: 0'], axis=1)

    def sample(self, frac=0.1, inplace=False):
        """
        Selects a random subset of the data.
        """
        sample = self.data.sample(frac=frac)
        if inplace:
            self.data = sample

        return sample


def load_data(path: str, data_type: str, flag: bool = False):
    """
    Load wind turbine data.
    
    Parameters
    ----------
    path : str
        The path to the directory in which the data is located.
    data_type : str
        Which data to load, either 'ARD' or 'CAU'.
    flag : bool, default=False
        Whether to apply the 'normal operation' flag.

    Returns
    -------
    pd.DataFrame
        Loaded CSV data.

    Throws
    ------
    TypeError
        If data_type is not 'ARD' or 'CAU'.
    """
    if data_type not in ('ARD', 'CAU'):
        raise TypeError('Argument \'data_type\' must be \'ARD\' or ' \
						+ '\'CAU\'.')

    data_file = data_type + '_Data.csv'
    data = pd.read_csv(os.path.join(path, data_file))
    assert type(data) is pd.DataFrame
    
    if flag:
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        joined_data = data.join(normal_operation, lsuffix='', rsuffix='f')
        queried_data = joined_data.query('value == 1')
		# Added as removing the flagged data messes up the data indexing
        queried_data.reset_index(drop=True, inplace=True)
        assert type(queried_data) is pd.DataFrame
        return queried_data

    return data


def load_positions(path, data_type, flag=False):
    """
    Load wind turbine relative position data.
    
    Parameters
    ----------
    path : str
        The path to the directory in which the data is located.
    data_type : str
        Which data to load, either 'ARD' or 'CAU'.
    flag : bool, default=False
        Whether to apply the 'normal operation' flag.

    Returns
    -------
    pd.DataFrame
        Loaded xlsx data.

    Throws
    ------
    ValueError
        If type is not 'ARD' or 'CAU'.
    """    
    if data_type not in ('ARD', 'CAU'):
        raise ValueError('Argument \'data_type\' must be \'ARD\' or ' \
						 + '\'CAU\'.')

    data_file = data_type + '_Turbine_Positions.csv'
    data = pd.read_csv(os.path.join(path, data_file))
    assert type(data) is pd.DataFrame

    if flag:
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        joined_data = data.join(normal_operation, lsuffix='', rsuffix='f')
		# Added as removing the flagged data messes up the data indexing
        joined_data.reset_index(drop=True,inplace=True)
        return joined_data.query('value == 1')

    return data


def load_data_positions(path, data_type, flag=False):
    """
    Load wind turbine data and relative position data combined.
    
    Parameters
    ----------
    path : str
        The path to the directory in which the data is located.
    type : str
        Which data to load, either 'ARD' or 'CAU'.

    Returns
    -------
    pd.DataFrame
        Loaded xlsx data.

    Throws
    ------
    TypeError
        If type is not 'ARD' or 'CAU'.
    """    

    if data_type not in ('ARD', 'CAU'):
        raise TypeError('Argument \'data_type\' must be \'ARD\' or ' \
						+ '\'CAU\'.')
    
    data_file = data_type + '_Data.csv'
    data = pd.read_csv(os.path.join(path, data_file))
    assert type(data) is pd.DataFrame

    pos_file = data_type + '_Turbine_Positions.csv'
    pos = pd.read_csv(os.path.join(path, pos_file))

    if flag:
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        data = data.join(normal_operation, lsuffix='', rsuffix='f')
		# Added as removing the flagged data messes up the data indexing
        data.reset_index(drop=True,inplace=True)

    data_joined = pd.merge(data, pos, left_on=["instanceID"],
						   right_on=["Obstical"])
    # data_joined.reset_index(drop=True,inplace=True)
    return data_joined
    
