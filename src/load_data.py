import pandas as pd


def load_data(path: str, data_type: str):
    """
    Load wind turbine data.
    
    Parameters
    ----------
    path : str
        the path to the directory in which the data is located.
    data_type : 'ARD' or 'CAU'
        which data to load.

    Returns
    -------
    pd.DataFrame
        loaded CSV data.

    Throws
    ------
    TypeError
        if data_type is not 'ARD' or 'CAU'
    """
    if data_type == 'ARD':
        data = pd.read_csv(path + 'ARD_Data.csv')
    elif data_type == 'CAU':
        data = pd.read_csv(path + 'CAU_Data.csv')
    else:
        raise TypeError('Argument \'data_type\' must be \'ARD\' or \'CAU\'.')

    assert type(data) is pd.DataFrame
    
    return data

