from model.predictor import Predictor

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from joblib import dump, load
import copy
from tqdm import tqdm


# Create dummy class such that diagonal entries in
# correlation function matrix just return their inputs
class DummyPredictor(object):
	def __init__(self):
		pass
	def fit(self,X,Y):
		print("Trying to fit turbine i to turbine i")
	def score(self,X,Y):
		return 1
	def predict(self,X):
		return X

class CorFuncAverage(Predictor):
	def __init__(self,filename):
		self.filename=filename
	def predict(self,data,targets,references):
		if data.data_type=="np.ndarray":
			#Load fitted correlation functions
			farm = data.farm
			func_mat = load(self.filename+'_'+str(farm)+'.joblib')
			r_mat = np.load(self.filename+'_'+str(farm)+'_r2.npy')
			
			#Load and reshape reference and target data
			x = data.data[:,references,2]
			xa = data.data[:,references,4]
			y = data.data[:,targets,2]
			ya = data.data[:,targets,4]
			X = np.stack((x,xa),axis=2)
			Y = np.stack((y,ya),axis=2)
			
			#Iterate over each target-reference pair
			preds = np.zeros((len(targets),len(references),x.shape[0],2))
			for i in range(len(targets)):
				for j in range(len(references)):
					preds[i,j] = func_mat[targets[i],references[j]].predict(X[:,j])
			
			print(r_mat)
			r_mat_masked = r_mat[targets][:,references]
			print(r_mat_masked)
			#r_mat_norm = (r_mat_masked/np.sum(r_mat_masked,axis=-1))
			#print(str(r_mat_norm))
			print("Weight matrix: "+str(r_mat_masked.shape))
			print("Predictions tensor: "+str(preds.shape))
			pred_av = np.zeros((preds.shape[0],preds.shape[2],preds.shape[3]))
			print(pred_av.shape)
			for i in range(len(targets)):
				print(r_mat_masked[i])
				print(preds[i].shape)
				pred_av[i] = np.average(preds[i],weights=r_mat_masked[i],axis=0)
			pred_av = (np.einsum("tda->dta",pred_av))
			print("Averaged prediction: "+str(pred_av.shape))
			"""
			
			"""
			return y,pred_av
		else:
			print("Convert to numpy array first")
	def train(self,data,D=10):
		"""
		Fits a pairwise interaction correlation function to each pair of turbines
		"""

		N = data.n_turbines
		its = list(range(N))
		k_mat = np.empty((N,N),dtype="object")
		for i in (its):
			for j in (its[:i]+its[i+1:]):
				k_mat[i,j] =  TransformedTargetRegressor(regressor=KernelRidge(kernel="laplacian",
																		  	   alpha=0.001,
																		  	   gamma=0.001),
														 func=np.log1p,
														 inverse_func=np.expm1)
		r_mat = np.zeros((N,N))
		for i in its:
			k_mat[i,i] = DummyPredictor()
			r_mat[i,i] = 1
		#print(k_mat.shape)
		
		for i in tqdm(its):
			for j in tqdm(its[:i]+its[i+1:]): # don't fit turbine j to turbine j
				#print("Fitting target "+str(i)+" to reference "+str(j))
				data_copy = copy.deepcopy(data)
				data_copy.select_turbine([i,j])
				data_copy.select_normal_operation_times()
				data_copy.select_unsaturated_times()
				x = data_copy.data[:,1,2] # reference power
				y = data_copy.data[:,0,2] # target power
				xa = data_copy.data[:,1,4] # reference angle
				ya = data_copy.data[:,0,4] # target angle
				#print("Available datapoints: "+str(x.shape[0]))
				X = np.stack((x,xa),axis=1)
				Y = np.stack((y,ya),axis=1)
				#Data transformation

				k_mat[i,j].fit(X[::D],Y[::D])
				r_mat[i,j] = k_mat[i,j].score(X,Y)
		dump(k_mat,self.filename+'_'+str(data.farm)+".joblib")
		np.save(self.filename+'_'+str(data.farm)+"_r2.npy",r_mat)