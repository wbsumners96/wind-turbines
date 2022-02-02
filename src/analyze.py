from load_data import *#load_data,load_positions,load_data_positions
from visualize import wind_direction_location,direction_power_histogram
from models import *
import argparse
import matplotlib.pyplot as plt
import numpy as np 


parser = argparse.ArgumentParser(description='Slow dancing with wind turbines.')
parser.add_argument('data_path', help='path to the directory in which the data is located.')
parser.add_argument('--type', help='type of data to load (ARD or CAU).')


args = parser.parse_args()
data = load_data_positions(args.data_path, args.type,False)
#print(data[3])
#data = load_data(args.data_path, args.type,True)
#data = select_time(data,30000,True)
#print(data)
direction_power_histogram(data)
D = 0.001
model_error_averaged(0,data,"10-Jan-2019 00:20:00",0,0)
"""
for D in np.linspace(1e-5,1e-4,100):
	def weighting(x):
		return np.exp(-x*x*D)
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




