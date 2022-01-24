import os
import pandas as pd
import numpy as np 

class TurbineData:
    def __init__(self, path, data_type):
        """
        Class constructor to load wind turbine data, relative position data and 
        normal operation flags.
        
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
        if data_type not in ('ARD', 'CAU'):
            raise TypeError('Argument \'data_type\' must be \'ARD\' or ' \
                            + '\'CAU\'.')
        
        data_file = data_type + '_Data.csv'
        data = pd.read_csv(os.path.join(path, data_file))
        assert type(data) is pd.DataFrame

        pos_file = data_type + '_Turbine_Positions.csv'
        pos = pd.read_csv(os.path.join(path, pos_file))

        #if flag:
        #    data = data.join(normal_operation, lsuffix='', rsuffix='f')
        #    # Added as removing the flagged data messes up the data indexing
        #    data.reset_index(drop=True,inplace=True)

        data_joined = pd.merge(data, 
                               pos, 
                               left_on=["instanceID"],
                               right_on=["Obstical"])
        
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        data_complete = pd.merge(data_joined, 
                                 normal_operation,
                                 on=["ts","instanceID"])

        self.data = data_complete
        self.farm = data_type
        self.datatype = "pd.DataFrame"

    def to_tensor(self):
        """
        Converts pd.dataframe to 2 rank 3 tensors, indexed by time, turbine and 
        attribute
        One tensor is a float containing all the actual data,
        one is of strings containing labels of datatime and turbine
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
        if self.farm == "ARD":
            n_turbines = 15
        elif self.farm == "CAU":
            n_turbines = 39
        data_tensor = data_numpy.reshape((n_turbines,-1,15))
        data_tensor = np.einsum("ijk->jik",data_tensor)
        print(data_tensor.shape)
        mask = np.array([0,0,1,1,1,1,1,0,0,1,1,1,1,1,1],dtype=bool)
        self.data = data_tensor[:,:,mask].astype(float)
        self.data_label = data_tensor[:,:,:2]
        self.datatype = "np.ndarray"

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
            print("Selected time: " + str(data.ts[time]))
        if self.datatype=="pd.DataFrame":
            return data[data.ts == data.ts[time]]
        elif self.datatype=="np.ndarray":
            self.data = data[time]
    def select_turbine(self, turbine, verbose=False):
        """
        Return the data for one wind turbine across all times.

        Parameters
        ----------
        time : str
        verbose : bool, default=False

        Returns
        -------
        pd.Series or numpy array, depending on current type of data stored in class
        """
        data = self.data
        if verbose:
            print("Selected turbine " + str(data.instanceID[turbine]))
        if self.datatype=="pd.DataFrame":
            return data[data.instanceID == data.instanceID[turbine]]
        elif self.datatype=="np.ndarray":
            self.data = data[:,turbine]
    
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
    
