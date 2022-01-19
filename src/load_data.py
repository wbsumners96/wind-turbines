import pandas as pd
import os


def load_data(path: str, data_type: str, flag: bool = False):
    """
    Load wind turbine data.
    
    Parameters
    ----------
    path : str
        the path to the directory in which the data is located.
    data_type : 'ARD' or 'CAU'
        which data to load.
    flag : bool, default = False
        whether to apply the 'normal operation' flag.

    Returns
    -------
    pd.DataFrame
        loaded CSV data.

    Throws
    ------
    TypeError
        if data_type is not 'ARD' or 'CAU'
    """
    if data_type not in ('ARD', 'CAU'):
        raise TypeError('Argument \'data_type\' must be \'ARD\' or \'CAU\'.')

    data_file = data_type + '_Data.csv'
    data = pd.read_csv(os.path.join(path, data_file))

    assert type(data) is pd.DataFrame
    
    if flag:
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        joined_data = data.join(normal_operation, lsuffix='', rsuffix='f')
        queried_data = joined_data.query('value == 1')
        queried_data.reset_index(drop=True,inplace=True) # Added as removing the flagged data messes up the indexing of the data
        assert type(queried_data) is pd.DataFrame

        return queried_data

    return data


def load_positions(path: str, data_type: str, flag: bool = False):
    """
    Load wind turbine relative position data.
    
    Parameters
    ----------
    path : str
        the path to the directory in which the data is located.
    type : 'ARD' or 'CAU'
        which data to load.

    Returns
    -------
    pd.DataFrame
        loaded xlsx data.

    Throws
    ------
    ValueError
        if type is not 'ARD' or 'CAU'
    """    
    if data_type not in ('ARD', 'CAU'):
        raise ValueError('Argument \'data_type\' must be \'ARD\' or \'CAU\'.')
    data_file = data_type + '_Turbine_Positions.csv'
    data = pd.read_csv(os.path.join(path, data_file))

    assert type(data) is pd.DataFrame
    if flag:
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        joined_data = data.join(normal_operation, lsuffix='', rsuffix='f')
        joined_data.reset_index(drop=True,inplace=True) # Added as removing the flagged data messes up the indexing of the data
        return joined_data.query('value == 1')

    return data


def load_data_positions(path: str, data_type: str, flag: bool = False):
    """
    Load wind turbine data and relative position data, combine them as one dataframe
    
    Parameters
    ----------
    path : str
        the path to the directory in which the data is located.
    type : 'ARD' or 'CAU'
        which data to load.

    Returns
    -------
    pd.DataFrame
        loaded xlsx data.

    Throws
    ------
    TypeError
        if type is not 'ARD' or 'CAU'
    """    

    if data_type not in ('ARD', 'CAU'):
        raise TypeError('Argument \'data_type\' must be \'ARD\' or \'CAU\'.')
    
    data_file = data_type + '_Data.csv'
    data = pd.read_csv(os.path.join(path, data_file))

    pos_file = data_type + '_Turbine_Positions.csv'
    pos = pd.read_csv(os.path.join(path, pos_file))

    assert type(data) is pd.DataFrame
    if flag:
        flag_file = data_type + '_Flag.csv'
        normal_operation = pd.read_csv(os.path.join(path, flag_file))
        data = data.join(normal_operation, lsuffix='', rsuffix='f')
        data.reset_index(drop=True,inplace=True) # Added as removing the flagged data messes up the indexing of the data

    data_joined = pd.merge(data,pos,left_on=["instanceID"],right_on=["Obstical"])
    #data_joined.reset_index(drop=True,inplace=True)
    return data_joined
    
def select_time(data,time,verbose=False):
    """
    Returns all wind turbines at a given time
    """
    if verbose:
        print("Selected time: "+str(data.ts[time]))
    return data[data.ts==data.ts[time]]

def select_turbine(data,turbine,verbose=False):
    """
    Returns one wind turbine's data across all times
    """
    if verbose:
        print("Selected turbine "+str(data.instanceID[turbine]))
    return data[data.instanceID==data.instanceID[turbine]]
