import sys
import time
import itertools
import threading

from load_data import TurbineData


class TurbineApp:
	"""
	Loads data and runs predictions for specified models.

	Attributes
	----------
	models : list of str
		List of the desired models to run.
	predictor_parameters : list of dict
		List of dictionaries where each dictionary contains the parameters
		needed to create a predictor.
	path : str
		The path to the directory where the data is located.
	farm : str
		The wind farm to use (either ARD or CAU).
	targets : list of int
		List of ID numbers of target turbines.
	references : list of int
		List of ID numbers of reference turbines.
	times : list of datetime
		List of times over which the models should be run.
	data: TurbineData
		The data to run the models with.	
	predictors : list of Predictor objects
		The predictors created.
	"""
	def __init__(self, data_path, farm):
		self.data_path = data_path
		self.farm = farm
		self.data = None
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
		sys.stdout.write('\rLoading complete.						\n')
		sys.stdout.flush()
		loading_complete = True

		print(self.data.data.info())
		print('Converting to tensor...')
		self.data.to_tensor()
		print('Conversion successful.')
	
	def trim_data(self):
		"""
		Trims the data to only the times and turbines of interest.
		"""
		self.data.select_turbine(self.targets + self.references)
		self.data.select_normal_operation_times()
		self.data.select_unsaturated_times()
		# self.data.select_time(self.times)
		

	def create_predictors(self):
		"""
		Creates predictor objects.
		"""
		for i in range(len(self.models)):
			new_predictor = self.models[i](**self.predictor_parameters[i])
			self.predictors.append(new_predictor)

	def run_predictions(self):
		"""
		Returns a list of predictions (one from each predictor object).
		"""
		results = []
		iso_times = []

		for time in self.times:
			iso_time = time.strftime(r'%d-%b-%Y %H:%M:%S')
			iso_times.append(iso_time)

		for predictor in self.predictors:
			results.append(predictor.predict(self.data,
											 self.targets,
											 self.references,
											 iso_times))
		return results

	def minimize_errors(self):
		"""
		Minimizes the errors for each prediction.
		"""
