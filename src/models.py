import matplotlib.pyplot as plt
import numpy as np 


def linear_combination(data, weighting, target, references, time):
    """
    Predict power of a target turbine at a given time by taking a linear combination of the powers of given reference turbines at that time.

    The coefficients in the linear combination is given by a function of distance from the target turbine.

    Parameters
    ----------
    data : pd.DataFrame
        turbine data.
    weighting : (distance: positive real) -> positive real
        coefficient of linear combination. 
    target : str with format '<type ('ARD' or 'CAU')>_WTGxx'
        ID of target turbine.
    references : list[str with format above]
        IDs of reference turbines.
    time : str with datetime format 'DD-MMM-YYYY hh:mm:ss'
        target timestamp to predict.

    Returns
    -------
    target_power : real
        true power output of target turbine at given time.
    predicted_power : real
        predicted power output of target turbine.
    """
    # restrict data to given time
    current_data = data.query('ts == @time')
    target_data = current_data.query('instanceID == @target')
    reference_data = current_data.query('instanceID == @references')

    # get vector of distances from target turbine to reference turbines
    target_position = target_data[['Easting', 'Northing']].to_numpy()
    reference_positions = reference_data[['Easting', 'Northing']].to_numpy()
    deltas = reference_positions - target_position
    
    rs = np.array([dx*dx + dy*dy for [dx, dy] in deltas])

    # get vector of weights
    ws = np.vectorize(weighting)(rs)

    # calculate predicted power as w_1 f(p_1) + ... + w_n f(p_n)
    target_power = target_data['Power'].to_numpy()
    reference_powers = reference_data['Power'].to_numpy()
    predicted_power = np.dot(ws, reference_powers)

    return target_power, predicted_power
    

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
