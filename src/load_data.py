import pandas as pd


def load_data(path: str, type: str):
    """
    Load wind turbine data.
    
    Parameters
    ----------
    path : str
        the path to the directory in which the data is located.
    type : 'ARD' or 'CAU'
        which data to load.

    Returns
    -------
    pd.DataFrame
        loaded CSV data.

    Throws
    ------
    TypeError
        if type is not 'ARD' or 'CAU'
    """
    if type == 'ARD':
        return pd.read_csv(path + 'ARD_Data.csv')
    elif type == 'CAU':
        return pd.read_csv(path + 'CAU_Data.csv')

    raise TypeError('Argument \'type\' must be \'ARD\' or \'CAU\'.')

