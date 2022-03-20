from joblib import dump
import sys
import time
import itertools
from pathlib import Path
import threading

from load_data import TurbineData


class TurbineApp:
    """
    Loads data and runs predictions for specified models.

    Attributes
    ----------
    data_path : str
        The path to the directory where the data is located.
    farm : str
        The wind farm to use (either ARD or CAU).
    data: TurbineData
        The data to run the models with.    
    models : list of str
        List of the desired models to run.
    remove_wake_affected: boolean
        Whether to remove wake affected turbines in data cleaning.
    predictor_parameters : list of dict
        List of dictionaries where each dictionary contains the parameters
        needed to create a predictor.
    targets : list of int
        List of ID numbers of target turbines.
    references : list of int
        List of ID numbers of reference turbines.
    times : list of datetime
        List of times over which the models should be run.
    predictors : list of Predictor objects
        The predictors created.
    """
    def __init__(self, data_path, farm):
        self.data_path = data_path
        self.farm = farm
        self.data = None

        self.remove_wake_affected = False
        self.sample_fraction = 1.0

        self.models = []
        self.predictor_parameters = {}
        self.predictors = []

        self.targets = []
        self.references = []
        self.times = []

    def load_data(self):
        """
        Loads the data and returns a TurbineData object.
        """
        loading_complete = False 
        def show_loading_animation():
            for c in itertools.cycle(['⠟', '⠯', '⠷', '⠾', '⠽', '⠻']):
                if loading_complete:
                    break
                
                sys.stdout.write('\r' + c + ' Loading turbine data...')
                sys.stdout.flush()
                time.sleep(0.1)

        animation_thread = threading.Thread(target = show_loading_animation)
        animation_thread.start()

        self.data = TurbineData(self.data_path, self.farm)
        sys.stdout.write('\rLoading complete.                        \n')
        sys.stdout.flush()
        loading_complete = True

        print(self.data.data.info())
        
    def clean_data(self):
        """
        Performs pre-processing operations on the data.
        """
        self.data.sample(frac=self.sample_fraction, inplace=True)
        if self.remove_wake_affected:
            print('Removing wake affected turbines...', end='')
            self.data.clear_wake_affected_turbines()
            print('Success.')
        print('Removing abnormally operating turbines...', end='')
        self.data.select_normal_operation_times()
        print('Success.')
        print('Selecting times up to baseline configuration date...', end='')
        self.data.select_baseline(inplace=True)
        print('Success.')

    def create_predictors(self):
        """
        Creates a predictor object for each model.
        """
        for i in range(len(self.models)):
            new_predictor = self.models[i](**self.predictor_parameters[i])
            self.predictors.append(new_predictor)

    def run_predictions(self, filepath, filename):
        """
        Returns a list of predictions (one for each model).
        """
        predictions = []
        for predictor in self.predictors:
            predictor.fit(self.data)
            prediction = predictor.predict(self.data,
                                           self.targets,
                                           self.references,
                                           self.times)

            dump(prediction,
                 filepath/f'{predictor.__class__.__name__}_{filename}.joblib')

            predictions.append(prediction)

        return predictions
        
    def run(self):
        """
        Cleans the data, creates predictor objects, and runs predictions.
        """
        print('\nStarting data cleaning...')
        self.clean_data()
        print('Data cleaning complete.\n\nCreating predictors...', end='')
        self.create_predictors()
        print('Success.\nRunning predictions...')

        path = Path('~/.turbines/app_predictions_testing/').expanduser()
        self.run_predictions(path, 'ARD')
        print('Predictions complete.')
