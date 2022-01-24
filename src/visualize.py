import matplotlib.pyplot as plt
import numpy as np 


def wind_direction_location(data_positions, time, filename=None):
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
	if data_positions.datatype=="np.ndarray":
		raise TypeError("Data needs to be in pd.DataFrame format")
	data_positions = data_positions.data
	data_time0 = (data_positions[data_positions.ts 
				  == data_positions.ts[time]])# .sort_values("Easting")
	print(data_time0)
	xs = data_time0['Easting'].to_numpy()
	ys = data_time0['Northing'].to_numpy()
	print(xs)
	vxs = (data_time0['Wind_speed'].to_numpy()
		   * np.cos(data_time0['Wind_direction_calibrated']
		   * np.pi / 180)).to_numpy()
	vys = (data_time0['Wind_speed'].to_numpy()
		   * np.sin(data_time0['Wind_direction_calibrated']
		   * np.pi / 180)).to_numpy()
	print(vxs)
	plt.quiver(xs,ys,vxs,vys)
	plt.scatter(xs,ys)
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
	if data.datatype=="np.ndarray":
		raise TypeError("Data needs to be in pd.DataFrame format")
	data = data.data
	n_bins = 100
	data_np = data[['Wind_direction_calibrated', 'Power']].to_numpy()
	print(np.nanmax(data_np[:,0]))
	
	hist, xedges, yedges = np.histogram2d(
						 data_np[:, 1], data_np[:, 0], bins=n_bins,
						 range=[[0, np.nanmax(data_np[:, 1])],[0, 360]])
	print(xedges.shape)

	fig,ax = plt.subplots(1,1)
	img = ax.imshow(hist,origin="lower")
	ax.set_xticks(np.arange(n_bins+1)[::10])
	ax.set_yticks(np.arange(n_bins+1)[::10])
	ax.set_xticklabels(yedges[::10])
	ax.set_yticklabels(xedges[::10])
	ax.set_title("ARD")
	ax.set_xlabel("Turbine angle")
	ax.set_ylabel("Power output")
	ax.tick_params(axis="x",labelrotation=-90)
	fig.colorbar(img)
	plt.show()
