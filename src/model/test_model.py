import pandas as pd
import numpy as np
from .predictor import Predictor


class TestModel(Predictor):
	def __init__(self, funniness):
		"""
		this model is for losers and nerds.
		"""
		self.funniness = funniness

	def predict(self, data, targets, references, times):
		col_names = ['target_id', 'reference_id', 'ts', 'true_power',
					 'predicted_power']
		test_targets = np.zeros((500, 1))
		test_refs = np.zeros((500, 1))
		test_ts = np.zeros((500, 1))
		test_true_power = np.random.normal(loc=1, size=(500, 1))
		test_pred_power = np.random.normal(loc=-1, size=(500, 1))
		test_data = np.concatenate((test_targets, test_refs, test_ts,
					 				test_true_power, test_pred_power),
									axis=1)
		result = pd.DataFrame(test_data, columns=col_names)
		return result

	def __str__(self):
		return f'i have {self.funniness} funny points'

