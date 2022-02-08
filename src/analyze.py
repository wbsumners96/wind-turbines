from load_data import *#load_data,load_positions,load_data_positions
from visualize import *#wind_direction_location,direction_power_histogram
from models import *
import argparse
import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
from model.weighted_average import *#GaussianWeightedAverage


parser = argparse.ArgumentParser(description='Slow dancing with wind turbines.')
parser.add_argument('data_path', help='path to the directory in which the data is located.')
parser.add_argument('--type', help='type of data to load (ARD or CAU).')





args = parser.parse_args()
data = TurbineData(args.data_path,args.type)
I=0

data.to_tensor()
data.nan_to_zero()
data.select_wind_direction(0,20)
#wind_direction_location(data,I,0,np.arange(1,15))
wind_direction_location(data,I,9,[4,5,8,11,12])
data.select_turbine([9,4,5,8,11,12])
data.select_normal_operation_times()
data.select_unsaturated_times(verbose=True)

direction_power_histogram(data)
direction_power_histogram(data)



#print(data.select_time(0))
#data.select_time(slice(I,I+1000))
print(data.data.shape)
#print(data.data)







def model_minimize(data,targets,references):
	def f(x,*args):
		_,_,_,_,_,error = GaussianWeightedAverage(x).predict_abs_error(args[0],args[1],args[2])
		return (error)
	args = [data,targets,references]
	#bounds = sp.Bounds(np.array(0),np.inf)
	#opt = (sp.minimize(f,x0=1e-5,args=args,bounds=[(0,None)],options={"disp":True}))
	opt = sp.shgo(f,args=args,bounds=[(0,1)],options={"disp":True})
	
	print(opt)
	return opt.x

def model_error(data,targets,references):
	def f(x,args):
		_,_,_,_,_,error = GaussianWeightedAverage(x).predict_abs_error(args[0],args[1],args[2])
		return error

	args = [data,targets,references]
	N=1000
	ers = np.zeros(N)
	xs = np.logspace(-10,-3,N)
	for i in range(N):
		ers[i] = f(xs[i],args)
	print(np.nanmin(ers))
	min_loc = np.arange(1000)[ers==np.nanmin(ers)][0]
	print(min_loc)
	

	plt.plot(xs,ers)
	plt.scatter(xs[min_loc],np.nanmin(ers))
	plt.xscale("log")
	plt.xlabel("gamma")
	plt.ylabel("Mean absolute error")
	plt.show()
	return xs[np.argmin(ers)]
#x = model_minimize(data,[0],np.arange(1,6))
x = model_error(data,[0],np.arange(1,6))
print("Minimised Parameters: "+str(x))


model = GaussianWeightedAverage(x)
tars,preds,err,m_err,_,_ = model.predict_abs_error(data,[0],np.arange(1,6))
plt.plot(err, label="error")
plt.plot(tars,label="Measured power",alpha=0.4)
plt.plot(preds,label="Predicted power",alpha=0.4)
plt.legend()
plt.show()


plt.hist(tars,label="Measured Power",alpha=0.4,bins=100)
plt.hist(preds,label="Predicted power",alpha=0.4,bins=100)
plt.xlabel("Power")
plt.ylabel("Count of times")
plt.title("Gaussian Weighted Average")
plt.legend()
plt.show()
print(tars.shape)
print(preds.shape)
prediction_measured_histogram(preds[:,0],tars[:,0])

#data = load_data_positions(args.data_path, args.type,False)
#print(data[3])
#data = load_data(args.data_path, args.type,True)
#data = select_time(data,30000,True)
#print(data)
#direction_power_histogram(data)




"""
D = 0.001
def weighting(x):
	return np.exp(-x*x*D)
target_turbines = [2,5]
preds,tars = weighted_average(data,weighting,[2,5],[3,4,6,7,8])

ers = np.abs(preds-tars)
print(ers.shape)
for i in range(len(target_turbines)):
	plt.plot(ers[:,i],label="Turbine "+str(target_turbines[i]))
plt.legend()
plt.ylabel("$\|P - P_m\|$")
plt.show()
#model_error_averaged(0,data,"10-Jan-2019 00:20:00",0,0)
"""
"""
for D in np.linspace(1e-5,1e-4,100):
	model = lambda x:weighted_average_and_knuckles(x,weighting,[1,3,9],[2,4,5,6,7],"10-Jan-2019 00:20:00")

	#target, predicted = weighted_average_and_knuckles(data,weighting,[1,3,9],[2,4,5,6,7],"10-Jan-2018 00:20:00")
	print(model_error(model,data))
"""
#print(target.shape)
#print(predicted.shape)
#print(target)
#print(predicted)


#print(weighted_average(data)[0])




#positions = load_positions(args.data_path, args.type)

#wind_direction_location(data_positions,100000)




