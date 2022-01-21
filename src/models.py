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

	distances = np.sqrt(np.sum((target_positions[:,np.newaxis,:]
				- reference_positions) ** 2, axis=-1))
	# Get vector of weights
	weights = np.vectorize(weighting)(distances)
	
	# Calculate predicted power as w_1 f(p_1) + ... + w_n f(p_n)
	target_powers = target_data['Power'].to_numpy()
	reference_powers = reference_data['Power'].to_numpy()
	predicted_powers = np.einsum('ij, j->i', weights, reference_powers) \
					   / np.sum(weights, axis=1)

	return target_powers, predicted_powers
    

def weighted_average(data, target='ARD_WTG01'):
	"""
	Predict the power of the specified wind turbine.
	
	First attempt at a crude model, where the power is predicted by a
	weighted average (by distance) of all other turbines at that point in
	time.

	Parameters
	----------
	data : pd.DataFrame
		Wind turbine data.
	target : str
		Target turbine ID.

	Returns
	-------
	powers_estimated : numpy.ndarray
		Estimated power output of the target turbine.
	powers_measured : numpy.ndarray
		Measured power output of the target turbine.
	"""

	# Calculate matrix of seprations between turbines i and j
	positions = data[['Easting', 'Northing']].to_numpy()
	dxs = (positions[:, 0][:, np.newaxis] - positions[:, 0])
	dys = (positions[:, 1][:, np.newaxis] - positions[:, 1])
	distances = np.sqrt(dxs * dxs + dys * dys)

	# Set by hand, too small and every wind turbine contributes similarly,
	# too big and no other wind turbine matters
	D = 0.00001
	# Gaussian-like weight matrix, with diagonals set to 0 so no wind
	# turbine estimate is based on self-measurement
	weights = np.exp(-D * distances * distances) \
			  - np.eye(distances.shape[0])
	plt.imshow(weights)
	plt.title('Weight matrix')
	plt.show()
	# print(positions)


	# Weighted average of f(powers) is just a matrix multiplication with
	# the weight matrix
	def f(power):
		# Dummy function to change later if we want something more complex 
		return power
	

	vf = np.vectorize(f)
	powers_measured = data['Power'].to_numpy()
	powers_estimated = np.einsum('i, ij->j', vf(powers_measured), weights) \
					   / np.sum(weights, axis=0)
	
	# Display results. So far a bit rubbish
	xs = np.arange(distances.shape[0])
	plt.scatter(xs, powers_estimated, label='Predicted')
	plt.scatter(xs,powers_measured,label='Correct')
	plt.xlabel('Turbine')
	plt.ylabel('Power')
	plt.legend()
	plt.show()

	return powers_estimated, powers_measured
