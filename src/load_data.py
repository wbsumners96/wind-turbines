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
    flag : bool
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
        
        return joined_data.query('value == 1')

    return data

