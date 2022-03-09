import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import copy
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
import matplotlib as mpl

def wind_direction_location(data_positions, time, targets, references, filename=None):
	"""
	Display scatterplot of turbine locations with wind speed and direction.

	Takes the output of load_data_positions() from load_data.py, and
	displays points as turbine locations with wind speed and direction as
	vectors.

	Parameters
	----------
	data_positions : pd.DataFrame
		Wind turbine position data. 
	time : str
		Timestamp with the datetime format 'DD-MMM-YYYY hh:mm:ss'.
	filename : str
		The name of the file containing the resulting plot.
	"""
	if data_positions.data_type=="np.ndarray":
		data = data_positions.data[time]
		xs = data[:,5]
		ys = data[:,6]
		vxs = data[:,1]*np.sin(data[:,4]*np.pi/180)
		vys = data[:,1]*np.cos(data[:,4]*np.pi/180)
		plt.quiver(xs,ys,vxs,vys)
		plt.scatter(xs,ys)
		plt.scatter(xs[targets],ys[targets],label="Target turbine")
		plt.scatter(xs[references],ys[references],label="Reference turbine")
		plt.legend()
		if filename != None:
			plt.savefig(filename, format='png')
		plt.show()





		#raise TypeError("Data needs to be in pd.DataFrame format")
	if data_positions.data_type=="pd.DataFrame":
		data_positions = data_positions.data
		data_time0 = (data_positions[data_positions.ts 
					  == data_positions.ts[time]])# .sort_values("Easting")
		#print(data_time0)
		xs = data_time0['Easting'].to_numpy()
		ys = data_time0['Northing'].to_numpy()
		#print(xs)
		vxs = (data_time0['Wind_speed'].to_numpy()
			   * np.sin(data_time0['Wind_direction_calibrated']
			   * np.pi / 180)).to_numpy()
		vys = (data_time0['Wind_speed'].to_numpy()
			   * np.cos(data_time0['Wind_direction_calibrated']
			   * np.pi / 180)).to_numpy()
		#print(vxs)
		plt.quiver(xs,ys,vxs,vys)
		plt.scatter(xs,ys)
		plt.scatter(xs[targets],ys[targets],label="Target turbine")
		plt.scatter(xs[references],ys[references],label="Reference turbine")
		plt.legend()
		if filename != None:
			plt.savefig(filename, format='png')
		plt.show()


def direction_power_histogram(data):
	"""
	Plot histogram of wind directions and power outputs.

	For a given dataframe, bin all wind directions and corresponding power
	outputs and plot a 2D histogram.

	Parameters
	----------
	data : pd.DataFrame
	"""
	

	if data.data_type=="np.ndarray":
		_dir = data.data[:,:,4].reshape(-1)
		_pow = data.data[:,:,2].reshape(-1)
		data_np = np.stack((_dir,_pow),axis=-1)
		print(data_np.shape)
	if data.data_type=="pd.DataFrame":
		data_np = data.data[['Wind_direction_calibrated', 'Power']].to_numpy()
		print(data_np.shape)

	n_bins = 100
	
	#print(np.nanmax(data_np[:,0]))
	
	plt.hist2d(data_np[:,0],data_np[:,1],bins=100,cmap="inferno")
	plt.xlabel("Turbine angle (degrees)")
	plt.ylabel("Turbine power (kW)")
	plt.title(data.farm)
	plt.colorbar()
	plt.show()

	
	"""
	hist, xedges, yedges = np.histogram2d(
						 data_np[:, 1], data_np[:, 0], bins=n_bins,
						 range=[[0, np.nanmax(data_np[:, 1])],[0, 360]])
	#print(xedges.shape)

	fig,ax = plt.subplots(1,1,figsize=(10,8))
	img = ax.imshow(hist,origin="lower")
	ax.set_xticks(np.arange(n_bins+1)[::10])
	ax.set_yticks(np.arange(n_bins+1)[::10])
	ax.set_xticklabels(yedges[::10].astype(int))
	ax.set_yticklabels(xedges[::10].astype(int))
	ax.set_title("ARD")
	ax.set_xlabel("Turbine angle (degrees)")
	ax.set_ylabel("Power output (kW)")
	#ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
	ax.tick_params(axis="x",labelrotation=-90)
	fig.colorbar(img)
	plt.show()
	"""

def prediction_measured_histogram(predictions,measurements):
	#plt.scatter(measurements,predictions,alpha=0.4)
	plt.hist2d(measurements,predictions,bins=100,norm=mpl.colors.LogNorm(),cmap="binary")
	plt.xlabel("Measured Power (kW)")
	plt.ylabel("Predicted Power (kW)")
	plt.show()




def visualize_cor_func_behaviour(X,Y,ys):
	cmap="binary"
	x = X[:,0]
	xa = X[:,1]
	y = Y[:,0]
	ya = Y[:,1]
	print(y.shape)
	print(ys[:,0].shape)
	#measured vs predicted powers
	plt.scatter(x[::10],y[::10],label="Measured",alpha=0.05)
	plt.scatter(x[1::10],ys[1::10,0],label="Prediction (on test data)",alpha=0.05)
	plt.xlabel("Reference power")
	plt.legend()
	plt.ylabel("Target power")
	plt.show()

	plt.scatter(xa[::10],ya[::10],label="Measured",alpha=0.05)
	plt.scatter(xa[1::10],ys[1::10,1],label="Prediction (on test data)",alpha=0.05)

	plt.xlabel("Reference angle")
	plt.legend()
	plt.ylabel("Target angle")
	plt.show()




	fig,ax = plt.subplots(2,1,sharex=True,sharey=True)

	ax[0].hist2d(xa,xa-ya,bins=100,range=[[np.min(xa),np.max(xa)],[-20,20]],norm=mpl.colors.LogNorm(),cmap=cmap)
	#ax[0].set_xlabel("Reference angle")
	ax[0].set_ylabel(r"$\theta_j-\theta_i$")
	ax[0].set_title("Measured")

	ax[1].hist2d(xa,xa-ys[:,1],bins=100,range=[[np.min(xa),np.max(xa)],[-20,20]],norm=mpl.colors.LogNorm(),cmap=cmap)
	ax[1].set_xlabel(r"$\theta_j$")
	ax[1].set_ylabel(r"$\theta_j - \hat{\theta}_i$")
	ax[1].set_title("Predicted")
	plt.show()




	fig,ax = plt.subplots(2,1,sharex=True,sharey=True)
	ax[0].hist2d(xa,(x-y),bins=100,norm=mpl.colors.LogNorm(),cmap=cmap)
	#ax[0].set_xlabel("Average angle of target/reference pair")
	ax[0].set_ylabel(r"$P_i-P_j$")
	ax[0].set_title("Measured")

	ax[1].hist2d(xa,(x-ys[:,0]),norm=mpl.colors.LogNorm(),bins=100,cmap=cmap)
	ax[1].set_xlabel(r"$\theta_j$")
	#ax[1].set_xlabel(r"$\frac{\theta_i+\theta_j}{2}$")
	ax[1].set_ylabel(r"$P_i-\hat{P}_j$")
	ax[1].set_title("Predicted")
	#plt.title("Power difference vs mean angle correlation")
	plt.show()


	plt.hist2d(ya,y-ys[:,0],bins=100,norm=mpl.colors.LogNorm(),cmap=cmap)
	plt.xlabel(r"$\theta_j$")
	plt.ylabel(r"$P_j - \hat{P}_j$")
	plt.show()






	plt.scatter((xa[::10]+ya[::10])/2,(x[::10]-y[::10]),label="Measured",alpha=0.05)
	plt.scatter((xa[::10]+ys[::10,1])/2,(x[::10]-ys[::10,0]),label="Prediction",alpha=0.05)
	plt.xlabel(r"$\frac{\theta_i+\theta_j}{2}$")
	plt.ylabel(r"$P_i-P_j$")
	plt.title("Power difference vs mean angle correlation")
	plt.legend()
	plt.show()

	plt.hist2d(y,(ys[:,0]),bins=100,norm=mpl.colors.LogNorm(),cmap=cmap)
	plt.xlabel(r"$P_j$")
	plt.ylabel(r"$\hat{P}_j$")
	plt.show()



	plt.scatter(y,(y-ys[:,0])/y,alpha=0.1)
	plt.xlabel(r"$P_j$")
	plt.ylabel(r"$\frac{P_j - \hat{P}_j}{P_j}$")
	plt.show()


	plt.hist2d(y,(y-ys[:,0])/(1+y),bins=100,norm=mpl.colors.LogNorm(),cmap=cmap)
	plt.xlabel(r"$P_j$")
	plt.ylabel(r"$\frac{P_j - \hat{P}_j}{P_j}$")
	plt.show()

	h,xedges,yedges=np.histogram2d(y,(y-ys[:,0]),bins=50)
	print(yedges)
	print(h)
	plt.imshow(h.T,origin="lower",
				   extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
				   interpolation="gaussian",
				   cmap="Greens")
	h_av = np.zeros((xedges.shape[0]-1))
	h_var = np.zeros((xedges.shape[0]-1))
	
	for i in range(h_av.shape[0]):
		h_av[i] = np.average(yedges[1:],weights=h[i])
		h_var[i]= np.sqrt(np.average((yedges[1:]-h_av[i])**2,weights=h[i]))
	plt.plot(xedges[:-1],h_av,color="red")
	plt.plot(xedges[:-1],h_av+h_var,color="red",alpha=0.2)
	plt.plot(xedges[:-1],h_av-h_var,color="red",alpha=0.2)
	plt.show()

	plt.hist(y,bins=100,alpha=0.4,label="Measured",density=True)
	plt.hist(ys[:,0],bins=100,alpha=0.4,label="Predicted",density=True)
	plt.xlabel("Target turbine power (kW)")
	plt.legend()
	plt.ylabel("Count")
	plt.show()


def average_power_gain_curve(data,k_mat):
	def all_predictions(data,k_mat):
		N=data.n_turbines
		#N=2
		its = list(range(N))
		predictions = [[] for n in range(N)]
		measurements = [[] for n in range(N)]
		errors = [[] for n in range(N)]
		for i in tqdm(its):
			for j in tqdm(its[:i]+its[i+1:]):
				data_copy = copy.deepcopy(data)
				data_copy.select_turbine([i,j])
				data_copy.select_normal_operation_times()
				data_copy.select_unsaturated_times()
				data_copy.select_power_min()
				x = data_copy.data[:,1,2] # reference power
				xa = data_copy.data[:,1,4] # reference angle
				y = data_copy.data[:,0,2] # target power
				#ya = data_copy.data[:,0,4] # target angle
				X = np.stack((x,xa),axis=1)
				#Y = np.stack((y,ya),axis=1)

				#print("Available datapoints: "+str(x.shape[0]))
				p = k_mat[i,j].predict(X)
				predictions[i].append(p[:,0])
				measurements[i].append(y)
				errors[i].append(y-p[:,0])
		print(predictions)
		return predictions,measurements,errors

	_,measured,errors=all_predictions(data,k_mat)
	#Plots power errors as functions of measured powers, averaged over turbines, like Oli showed in meeting
	M = np.array(measured,dtype=object).flatten()
	E = np.array(errors,dtype=object).flatten()

	M = np.array(M,dtype=float)
	E = np.array(E,dtype=float)
	#P = predicted.reshape(-1)
	print(E)
	print(M)

	h,xedges,yedges=np.histogram2d(M,E,bins=100,range=[[M.min(),M.max()],[E.min(),E.max()]])
	print(yedges)
	print(h)
	plt.imshow(h.T,origin="lower",
				   extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
				   interpolation="gaussian",
				   cmap="Greens")
	h_av = np.zeros((xedges.shape[0]-1))
	h_var = np.zeros((xedges.shape[0]-1))
	
	for i in range(h_av.shape[0]):
		h_av[i] = np.average(yedges[1:],weights=h[i])
		h_var[i]= np.sqrt(np.average((yedges[1:]-h_av[i])**2,weights=h[i]))
	plt.plot(xedges[:-1],h_av,color="red",label="Weighted average")
	plt.plot(xedges[:-1],h_av+h_var,color="red",alpha=0.2,label="Standard deviation")
	plt.plot(xedges[:-1],h_av-h_var,color="red",alpha=0.2)
	plt.xlabel("$P$")
	plt.ylabel("$\delta P$")
	plt.colorbar()
	plt.legend()
	plt.show()
