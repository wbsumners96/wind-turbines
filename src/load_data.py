import os
import pandas as pd


def load_data(path, data_type, flag=False):
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
    
def select_time(data, time, verbose=False):
    """
    Return the data for all wind turbines at the specified time.

	Parameters
	----------
	data : pd.DataFrame
	time : str
	verbose : bool, default=False

	Returns
	-------
	pd.Series
    """
    if verbose:
        print("Selected time: " + str(data.ts[time]))
    return data[data.ts == data.ts[time]]

def select_turbine(data, turbine, verbose=False):
    """
    Return the data for one wind turbine across all times.

	Parameters
	----------
	data : pd.DataFrame
	time : str
	verbose : bool, default=False

	Returns
	-------
	pd.Series
    """
    if verbose:
        print("Selected turbine " + str(data.instanceID[turbine]))
    return data[data.instanceID == data.instanceID[turbine]]
