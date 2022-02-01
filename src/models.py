import matplotlib.pyplot as plt
import numpy as np


def weighted_average_and_knuckles(data, weighting, targets,
                                  references, time):
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
    current_data = data.query('ts == @time')
    target_data = current_data.query('instanceID == @target_ids')
    reference_data = current_data.query('instanceID == @reference_ids')

    # Get vector of distances from target turbines to reference turbines
    target_positions = target_data[['Easting', 'Northing']].to_numpy()
    reference_positions = reference_data[['Easting', 'Northing']].to_numpy()

    distances = np.sqrt(np.sum((target_positions[:, np.newaxis, :]
                                - reference_positions) ** 2, axis=-1))
    # Get vector of weights
    weights = np.vectorize(weighting)(distances)

    # Calculate predicted power as w_1 f(p_1) + ... + w_n f(p_n)
    target_powers = target_data['Power'].to_numpy()
    reference_powers = reference_data['Power'].to_numpy()
    predicted_powers = np.einsum('ij, j->i', weights, reference_powers) \
        / np.sum(weights, axis=1)

    return target_powers, predicted_powers


def model_error(model, data):
    """
    Return the normalised absolute error between predicted and actual power.

    Computes the error for one dataset and instance of model.

    Parameters
    ----------
    model : (data : pd.DataFrame) -> list[float], list[float]
            Prediction model with all other parameters set.
    data : pd.DataFrame
            Wind turbine data.

    Returns
    -------
    error : float
    """
    actual, predictions = model(data)
    return np.mean(np.abs(actual - predictions) / actual)


def model_error_averaged(model, data, date_range, turbine_refs,
                         turbine_targets):
    """
    Calculate average model error.

    Calculates the model error averaged over different choices of target
    turbines, reference turbines, and times.

    Parameters
    ----------
    model : (data : pd.DataFrame,
                     targets : list[int],
                     references : list[int],
                     time : str with datetime format 'DD-MM-YYYY hh:mm:ss)
                         -> list[float],list[float]
            Model with only weight function specified.
    data : pd.DataFrame
            Turbine data.
    date_range : str
            Range of dates to average over with 2 datatime formats,
            'DD-MM-YYYY hh:mm:ss : DD-MM-YYYY hh:mm:ss'.
    turbine_refs : int
            Number of reference turbines.
    turbine_targets : int
            Number of target turbines.

    Returns
    -------
    error : float
    """
    data = data.loc[date_range]
    print(data)


def weighted_average(data, weighting, tar_mask, ref_mask, verbose=False):
    """
    Predict the power of the specified wind turbines.
    Needs data as a numpy array, parallel over time axis


    Parameters
    ----------
    data : numpy.ndarray
            Wind turbine data.
    weighting : (distance: positive float) -> positive float
        Function that determines the coefficient of linear combination.
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

    if data.datatype != 'np.ndarray':
        raise TypeError('Data must be numpy array, run .to_tensor() first')
    data = data.data

    tars = data[:, tar_mask]
    refs = data[:, ref_mask]

    if not np.all(tars[:, -1]):
        print("Warning: some target turbines are faulty")
    if not np.all(refs[:, -1]):
        print("Warning: some reference turbines are faulty")

    # Position data
    tar_pos = tars[0, :, 5:6]  # turbines don't move
    ref_pos = refs[0, :, 5:6]
    # Power data
    tar_power = tars[:, :, 2]
    ref_power = refs[:, :, 2]

    # Calculate euclidean distance between all target-reference pairs
    ds = np.sqrt(np.sum((tar_pos[:, np.newaxis, :] - ref_pos)**2, axis=-1))

    ws = np.vectorize(weighting)(ds)
    if verbose:
        plt.imshow(ws)
        plt.title('Weight matrix')
        plt.show()

    def f(power):
        # Dummy function to change later if we want something more complex
        return power

    vf = np.vectorize(f)
    pred_power = np.einsum('ij, kj->ki', ws, ref_power) / np.sum(ws, axis=1)

    return tar_power, pred_power
