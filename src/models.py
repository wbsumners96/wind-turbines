import matplotlib.pyplot as plt
import numpy as np 


def weighted_average_and_knuckles(data, weighting, targets, references, time):
	"""
	Predict power of a target turbine at a given time by taking a weighted average of the powers of given reference turbines at that time.

	The coefficients in the weighted average is given by a function of distance from the target turbine.

	Parameters
	----------
	data : pd.DataFrame
	    turbine data.
	weighting : (distance: positive float) -> positive float
	    coefficient of linear combination. 
	target : list[int]
	    ID of target turbines.
	references : list[int]
	    IDs of reference turbines.
	time : str with datetime format 'DD-MM-YYYY hh:mm:ss'
	    target timestamp to predict.

	Returns
	-------
	target_power : float
	    true power output of target turbine at given time.
	predicted_power : float
	    predicted power output of target turbine.

	Raises
	------
	ValueError
	    data type is not 'ARD' or 'CAU'
	"""
	# generate string ids of turbines
	# first learn the type of the data
	first_id = data['instanceID'][0]
	if first_id.startswith('ARD'):
		type = 'ARD'
	elif first_id.startswith('CAU'):
		type = 'CAU'
	else:
		raise ValueError('data is of an unexpected type.')

	#target_id = f'{type}_WTG{target:02d}'
	target_ids = [f'{type}_WTG{target:02d}' for target in targets]
	reference_ids = [f'{type}_WTG{reference:02d}' for reference in references]

	# restrict data to given time and separate into target and reference
	current_data = data.query('ts == @time')
	target_data = current_data.query('instanceID == @target_ids')
	reference_data = current_data.query('instanceID == @reference_ids')

	# get vector of distances from target turbine to reference turbines
	target_positions = target_data[['Easting', 'Northing']].to_numpy()
	reference_positions = reference_data[['Easting', 'Northing']].to_numpy()

	rs = np.sqrt(np.sum((target_positions[:,np.newaxis,:]-reference_positions)**2,axis=-1))

	# get vector of weights
	ws = np.vectorize(weighting)(rs)

	# calculate predicted power as w_1 f(p_1) + ... + w_n f(p_n)
	target_powers = target_data['Power'].to_numpy()
	reference_powers = reference_data['Power'].to_numpy()
	predicted_powers = np.einsum("ij,j->i",ws,reference_powers)/np.sum(ws,axis=1)

	return target_powers, predicted_powers
    

def model_error(model,data):
	"""
	Returns the normalised absolute error between predicted and actual powers, for one dataset and instance of model


	Parameters
	----------
	model : (data: pd.DataFrame) -> list[float],list[float]
		prediction model with all other parameters set
	data : pd.DataFrame
		turbine data

	Returns
	-------
	error : float

	"""
	actual,predictions = model(data)
	return np.mean(np.abs(actual-predictions)/actual)

def model_error_averaged(model,data,date_range,turbine_refs,turbine_targets):
	"""
	Calculates model error averaged over different choices of target and reference turbines, and times


	Parameters
	----------
	model : (data: pd.DataFrame, 
			 targets: list[ints], 
			 references: list[ints], 
			 time: str with datetime format 'DD-MM-YYYY hh:mm:ss) -> list[float],list[float]
		model but with only weight function specified
	data: pd.DataFrame
		turbine data
	date_range: str with 2 datatime format 'DD-MM-YYYY hh:mm:ss : DD-MM-YYYY hh:mm:ss'
		range of dates to average over
	turbine_refs: int
		number of referene turbines
	turbine_targets: int
		number of target turbines

	Returns
	-------
	error: float
	"""
	data = data.loc[date_range]
	print(data)


def weighted_average(data,target="ARD_WTG01"):
	"""
	First attempt at crude model. Predict power of wind turbine by weighted (by distance) average of all other turbines at that point in time
	"""

	#--- Calculate matrix of seprations between turbines i and j
	positions = data[["Easting","Northing"]].to_numpy()
	dxs = (positions[:,0][:,np.newaxis]-positions[:,0])
	dys = (positions[:,1][:,np.newaxis]-positions[:,1])
	rs = np.sqrt(dxs**2+dys**2)

	#--- Gaussian like weight matrix, with diagonals set to 0 so no wind turbine estimate based on self measurement
	D=0.00001 # Set by hand, too small and every wind turbine contributes similarly, too big and no other wind turbine matters
	ws = np.exp(-D*rs**2)-np.eye(rs.shape[0])
	plt.imshow(ws)
	plt.title("Weight matrix")
	plt.show()
	#print(positions)

	#--- Weighted average of f(powers) is just a matrix multiplication with weight matrix
	def f(power):
		return power # dummy function for changing later if we want something more complex 
	vf = np.vectorize(f)
	powers_measured = data["Power"].to_numpy()
	powers_estimated = np.einsum("i,ij->j",vf(powers_measured),ws) / np.sum(ws,axis=0)
	
	#--- Display results. So far a bit rubbish
	xs = np.arange(rs.shape[0])
	plt.scatter(xs,powers_estimated,label="Predicted")
	plt.scatter(xs,powers_measured,label="Correct")
	plt.xlabel("Turbine")
	plt.ylabel("Power")
	plt.legend()
	plt.show()
	return powers_estimated, powers_measured
