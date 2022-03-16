import sys
import re
import datetime
import inspect
import matplotlib.pyplot as plt

from turbine_app import TurbineApp
from model import models


def main():

	if len(sys.argv) != 2:
		print('Usage: python turbine_ui.py data_path')
	else:
		data_path = sys.argv[1]


	# Get user input for which farm to use
	farm_options = ['ARD', 'CAU']
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

	# Get user input for target and reference turbines and times
	app.targets, app.references = select_turbines()
	print(f'Targets selected: {app.targets}')
	print(f'References selected: {app.references}')
	app.times = app.data.data.loc[:,'ts'] # selecting all times
	print('Running predictions over all times by default.')

	# Run predictions
	app.predictor_parameters = select_predictor_parameters(app)
	print('\nCreating predictors and running predictions...')
	results = app.run() # results is a list of dataframes

	# TODO: replace with a call to visualize
	# Display results
	for result in results:
		model_name = app.models[results.index(result)].__name__
		print(f'\nGenerating the figure for {model_name}...')
		display_results(result, model_name)
		print(f'Successfully created the {model_name} results figure.')


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
		user_selection = input()

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
	print('\nEnter target turbine IDs separated by commas: ',
		  end='')
	user_input = input()

	# Populate the target list with the user's selections
	try:
		user_targets = user_input.replace(' ', '').split(',')
		for user_target in user_targets:
			targets.append(int(user_target))
	except ValueError:
		print('ValueError: Input must be integers separated by commas.\n')

	# Prompt for reference turbine IDs
	print('Enter reference turbine IDs separated by commas: ',
		  end='')
	user_input = input()

	# Populate the reference list with the user's selections
	try:
		user_references = user_input.replace(' ', '').split(',')
		for user_reference in user_references:
			references.append(int(user_reference))
	except ValueError:
		print('ValueError: Input must be integers separated by commas.\n')

	# Return the lists of targets and references
	return targets, references


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


def display_results(result, model_name):
	plt.hist(result.loc[:, 'predicted_power'],
			 # histtype='stepfilled',
			 color='hotpink',
			 label='Predicted power',
			 alpha=0.6)
	plt.hist(result.loc[:, 'true_power'],
			 # histtype='stepfilled',
			 color='cornflowerblue',
			 label='Measured power',
			 alpha=0.6)
	plt.title(model_name + ' Predicted and Measured Power')
	plt.ylabel('Power (kW)')
	plt.legend()
	plt.show()


main()
