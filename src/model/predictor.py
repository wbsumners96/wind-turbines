from abc import abstractmethod
import numpy as np
import scipy as sp

class Predictor:
    """
    A predictor is an instance of a model, and is an object which should be
    capable of predicting the power output at a given time of a set of target
    turbines in a particular farm from a disjoint set of reference turbines.

    The predictor can make use of the data for the reference turbines at the
    given times.

    The constructor of a predictor must not have any positional-only parameters,
    and each parameter must be a float.
    """
    @abstractmethod
    def predict(self, data, targets, references, times):
        """
        Predict the output power of a given set of target turbines given a 
        separate set of reference turbines at a collection of times.

        Parameters
        ----------
        data : TurbineData 
            Full turbine data for a particular farm.
        targets : list[int]
            List of target turbine IDs.
        references : list[int]
            List of reference turbine IDs.
        times : list[str]
            List of interested times (format 'DD-MMM-YYYY hh:mm:ss').

        Returns
        -------
        target_powers : 2D array
            True power of target turbines at given times.
        predicted_powers : 2D array
            Predicted power.

        Raises
        ------
        ValueError
            Targets or references or times is empty.
        MissingTurbineError
            At least one target or reference turbine ID doesn't exist in data.
        MissingTimeError
            At least one time doesn't exist in data. 
        """
        return NotImplementedError()

    @abstractmethod
    def fit(self, data):
        """
        Fit a model against some data.
        """
        raise NotImplementedError()
    
    def predict_abs_error(self, data, targets, references, times=None):
        """
        Run the predict() function, and output it's results alongside
        information about the error between prediction and target

        Parameters
        ----------
        data : TurbineData 
            Full turbine data for a particular farm.
        targets : list[int]
            List of target turbine IDs.
        references : list[int]
            List of reference turbine IDs.
        times : list[str]
            List of interested times (format 'DD-MMM-YYYY hh:mm:ss').

        Returns
        -------
        target_powers : 2D array
            True power of target turbines at given times.
        predicted_powers : 2D array
            Predicted power of target turbines at given times
        abs_err : 2D array
            Magnitude of difference between measured and predicted
            turbine powers for target turbines
        abs_err_turbine_average : 1D array
            abs_err, but averaged over the target turbines
        abs_err_time_average : 1D array
            abs_err, but averaged over times
        abs_err_total_average : float
            abs_err averaged over all times and turbines

        """


        tar_powers,pred_powers = self.predict(data,targets,references,times)
        abs_err = np.abs(tar_powers-pred_powers)
        abs_err_turbine_average = np.mean(abs_err,axis=-1)
        abs_err_time_average = np.mean(abs_err,axis=0)
        abs_err_total_average= np.mean(abs_err)
        return tar_powers,pred_powers,abs_err,abs_err_turbine_average,abs_err_time_average,abs_err_total_average
