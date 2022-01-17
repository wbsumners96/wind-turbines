import matplotlib.pyplot as plt
import numpy as np 

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
