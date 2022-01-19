import matplotlib.pyplot as plt
import numpy as np 



def wind_direction_location(data_positions,time,filename=None):
	"""
	Takes output of load_data_positions() from load_data.py, and displays scatterplot of turbine locations with wind speed/directions as vectors
	"""
	data_time0 = (data_positions[data_positions.ts==data_positions.ts[time]])#.sort_values("Easting")
	print(data_time0)
	xs = data_time0["Easting"].to_numpy()
	ys = data_time0["Northing"].to_numpy()
	print(xs)
	vxs = (data_time0["Wind_speed"].to_numpy()*np.cos(data_time0["Wind_direction_calibrated"]*np.pi/180)).to_numpy()
	vys = (data_time0["Wind_speed"].to_numpy()*np.sin(data_time0["Wind_direction_calibrated"]*np.pi/180)).to_numpy()
	print(vxs)
	plt.quiver(xs,ys,vxs,vys)
	plt.scatter(xs,ys)
	if filename!=None:
		plt.savefig(filename,format="png")
	plt.show()

def direction_power_histogram(data):
	"""
	For a given dataframe, bin all wind directions and corresponding power outputs and plot 2d histogram
	"""
	n_bins=100
	data_np = data[["Wind_direction_calibrated","Power"]].to_numpy()

	print(np.nanmax(data_np[:,0]))
	
	hist,xedges,yedges = np.histogram2d(data_np[:,0],data_np[:,1],bins=n_bins,range=[[0,360],[0,np.nanmax(data_np[:,1])]])
	print(xedges.shape)

	fig,ax = plt.subplots(1,1)
	img = ax.imshow(hist,origin="lower")
	ax.set_xticks(np.arange(n_bins+1)[::10])
	ax.set_yticks(np.arange(n_bins+1)[::10])
	ax.set_xticklabels(xedges[::10])
	ax.set_yticklabels(yedges[::10])
	ax.set_xlabel("Turbine angle")
	ax.set_ylabel("Power output")
	ax.tick_params(axis="x",labelrotation=-90)
	fig.colorbar(img)
	plt.show()