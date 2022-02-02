import sys
import re
import datetime
import inspect

from turbine_app import TurbineApp
from model import models


def main():
    data_path = '~/Documents/academia/current_courses/wind_project/DataPack'
    farm_options = ['ARD', 'CAU']

    # Get user input for which farm to use
    farm = select_farm(farm_options)
    print(f'Farm selected: {farm}\n')

    # Initialize the app and load data
    app = TurbineApp(data_path, farm)
    app.load_data()

    # Get user input for models
    app.models = select_models()
    print(f'\nModels selected:')
    for model in app.models:
        print(model.__name__)

    # Get user input for target and reference turbines
    app.targets, app.references = select_turbines()
    print(f'Targets selected: {app.targets}')
    print(f'References selected: {app.references}')

    # Get user input for times
    app.times = select_times(app)
    print('Times selected:')
    for time in app.times:
        print(time)

    # Run predictions
    app.predictor_parameters = select_predictor_parameters(app)
    app.create_predictors()
    print('\nPredictor creation successful. Running predictions...')
    results = app.run_predictions()
    # app.minimize_errors()

    # Display results
    print('\nResults:', end='')
    for result in results:
        model_name = app.models[results.index(result)].__name__
        if result == None:
            print('\nNo result returned for the ' + model_name + ' model.')
        else:
            print('\n' + model_name)
            print('Powers of the target turbine(s) predicted by the '
                  + model_name + ' model:')
            for value in result[0]:
                print(value)

            print('Powers of the target turbine(s) measured by the '
                  + model_name + ' model:')
            for value in result[1]:
                print(value)


def select_farm(farms):
    """
    Returns the user's choice for which farm to work on.

    Parameters
    ----------
    farms : list of str
            List of the wind farm codes.

    Returns
    -------
    str
            The code for the selected farm.
    """
    # Initial prompt
    print('Choose the farm to work on: ')
    for i, farm in enumerate(farms):
        print(f'({i+1}) {farm}')
    print('(Q) Quit')
    user_selection = input()

    # Terminate if the user quits
    if user_selection == 'Q' or user_selection == 'q':
        sys.exit()

    # As long as the user enters invalid input, display an error message and
    # display the prompt again
    try:
        selected_index = int(user_selection) - 1
        while selected_index not in range(len(farms)):
            print('Invalid selection.')
            print('Choose the farm to work on: ')
            for i, farm in enumerate(farms):
                print(f'({i+1}) {farm}')
            user_selection = input()
            selected_index = int(user_selection) - 1
    except ValueError:
        print('ValueError: Input must be an integer.\n')

    # Return the validated user farm selection
    return farms[selected_index]


def select_models():
    """
    Returns a list of the user's choices of models to run.
    """
    selected_models = []
    user_selection = 'Y'

    while user_selection == 'Y' or user_selection == 'y':
        # Initial prompt
        print('\nChoose from the following available models: ')
        for i, model in enumerate(models):
            print(f'({i+1}) {model.__name__}')
        print('(Q) Quit')
        user_selection = input()

        # Terminate if the user quits
        if user_selection == 'Q' or user_selection == 'q':
            sys.exit()

        # As long as the user enters invalid input, display an error message and
        # display the prompt again
        try:
            selected_index = int(user_selection) - 1
            while selected_index not in range(len(models)):
                print('Invalid selection. Choose from the following available '
					  + 'models: ')
                for i, model in enumerate(models):
                    print(f'({i+1}) {model.__name__}')
                user_selection = input()
                selected_index = int(user_selection) - 1
        except ValueError:
            print('ValueError: Input must be an integer.\n')

        # Add the user's selection to the list of selected models
        selected_models.append(models[selected_index])

        # Prompt to select additional models
        print('Select an additional model? (Y/N): ', end='')
        user_selection = input()

    # Return the validated list of selected models
    return selected_models


def select_turbines():
    """
    Returns two int lists of turbine IDs, for targets and references.
    """
    targets = []
    references = []

    # Prompt for target turbine IDs
    print('\nEnter target turbine IDs separated by commas, or (Q) to quit: ',
          end='')
    user_input = input()

    # Terminate if the user quits
    if user_input == 'Q' or user_input == 'q':
        sys.exit()

    # Populate the target list with the user's selections
    try:
        user_targets = user_input.replace(' ', '').split(',')
        for user_target in user_targets:
            targets.append(int(user_target))
    except ValueError:
        print('ValueError: Input must be integers separated by commas.\n')

    # Prompt for reference turbine IDs
    print('Enter reference turbine IDs separated by commas, or (Q) to quit: ',
          end='')
    user_input = input()

    # Terminate if the user quits
    if user_input == 'Q' or user_input == 'q':
        sys.exit()

    # Populate the reference list with the user's selections
    try:
        user_references = user_input.replace(' ', '').split(',')
        for user_reference in user_references:
            references.append(int(user_reference))
    except ValueError:
        print('ValueError: Input must be integers separated by commas.\n')

    # Return the lists of targets and references
    return targets, references


def select_times(app):
    """
    Returns a datetime list of times selected by the user.

    Parameters
    ----------
    app : TurbineApp

    Returns
    -------
    list of datetime
        The times over which to run the predictions.
    """
    selected_times = []

    # Prompt to choose the format of time selection
    print('\nChoose a time selection method:')
    print('(A) All times\n(R) Range of times\n(I) Individual times\n(Q) Quit')
    user_input = input()

    # As long as the user does not select an option, display an error message
    # and display the prompt again
    while user_input != 'A' and user_input != 'a' and user_input != 'R' \
                            and user_input != 'r' and user_input != 'I' \
                            and user_input != 'i' and user_input != 'Q' \
                            and user_input != 'q':
        print('Invalid selection.')
        print('\nChoose a time selection method:')
        print('(A) All times\n(R) Range of times\n(I) Individual times\n'
              + '(Q) Quit')
        user_input = input()

    # Terminate if the user quits
    if user_input == 'Q' or user_input == 'q':
        sys.exit()
    elif user_input == 'A' or user_input == 'a':
        selected_times.append(app.data.data)
    elif user_input == 'R' or user_input == 'r':
        # Prompt for the start and end times
        print('Enter the start time of the range in the format [YYYY-MM-DD '
              + 'hh:mm:ss]: ', end='')
        start_time = input()
        print('Enter the end time of the range in the format [YYYY-MM-DD '
              + 'hh:mm:ss]: ', end='')
        end_time = input()

        # Get the desired time slice from the data
        try:
            start_time = datetime.datetime.fromisoformat(start_time)
            end_time = datetime.datetime.fromisoformat(end_time)
            selected_times.append(app.data.data[start_time:end_time])
        except ValueError:
            print('ValueError: Invalid time format.\n')
    else:
        # Prompt for individual times
        print('Enter times in the format [YYYY-MM-DD hh:mm:ss] separated by'
              + ' commas: ', end='')
        user_input = input()

        # Get the user's list of times and convert them to datetime objects
        try:
            user_times = user_input.replace(', ', ',').split(',')
            for time in user_times:
                selected_times.append(datetime.datetime.fromisoformat(time))
        except ValueError:
            print('ValueError: Invalid time format.\n')

    return selected_times


def select_predictor_parameters(app):
    """
    Returns the parameters the user specifies for each selected model.

    Parameters
    ----------
    app : TurbineApp

    Returns
    -------
    list of dict
            The parameters to be used in the creation of the predictors, wher
            each dictionary element in the list contains the parameters
            corresponding to a selected model.
    """
    predictor_parameters = []
    # Create a dictionary of parameters for each model
    for model in app.models:
        user_parameters = {}
        # print(model.__init__.__doc__)
        print(f'\nSetting the parameters for the {model.__name__} model...')

        for parameter in inspect.signature(model.__init__).parameters.keys():
            # Automatically set the parameters required for all predictors

            # elif parameter == 'data':
            # 	user_parameters[parameter] = app.data
            # 	print('Automatically set data parameter to the loaded data.')
            # elif parameter == 'targets':
            # 	user_parameters[parameter] = app.targets
            # 	print(f'Automatically set targets to {targets}.')
            # elif parameter == 'references':
            # 	user_parameters[parameter] == app.references
            # 	print(f'Automatically set references to {references}.')
            # elif parameter == 'times':
            # 	user_parameters[parameter] == app.times
            # 	print(f'Automatically set times to {times}.')
            # Get user input for any remaining parameters
            # else:
            if parameter != 'self':
                print(f'Enter a value for {parameter}: ', end='')
                user_input = input()
                try:
                    user_parameters[parameter] = float(user_input)
                    print(f'Set {parameter} to {user_input}.')
                except ValueError:
                    print('ValueError: Input must be a float.\n')

        predictor_parameters.append(user_parameters)
    return predictor_parameters


main()
