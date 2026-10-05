### SVD decomposition classes  ###
### P. Vagovic                 ###


import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys


parameters = {'figure.figsize': [10,10],
              'axes.labelsize': 16,
              'axes.titlesize': 18,
              'axes.labelsize': 16,
              'axes.titlesize':16,
              'xtick.labelsize':16,
              'ytick.labelsize':16
              }
plt.rcParams.update(parameters)








#class RandSVD(object):
#	"""docstring for RandSVD"""
#	def __init__(self):
#		"""
#			here we decide how to read data
#			.. 1) get size of dataset
#			   2) create chunks (bufers for calculation of QT * X)
#				each chunk have list of blocks to loop throug
#				Buffer for X   update X with following block
#
#
#
#		"""
#
#	def set_data_matrix(self, ImageSequence):
#		"""
#		Expecting 4D matrix NTrain, NBuffer, NX, NY 
#		Every time we set data matrix we update the buffer 
#		with Z matrix  
#
#		"""
#	
#def rSVD(X,r,p,q)



class ImageSVD(object):
	"""docstring for SVD"""
	


	def __init__(self):
		self.U = 0
		self.S = 0
		self.VT = 0
		self.have_truncated_matrixes = False
		self.have_full_svd = False

	def set_data_matrix(self, image_stack):
		"""
		 expecting [Nz,Nx,Ny] coordinates

		"""
		self.NZ,self.NX,self.NY = image_stack.shape
		self.X = image_stack.reshape(self.NZ,-1).T
		self.X_mean = self.X.mean()

		self.X = self.X - self.X_mean  #### center the data 

		print('Reshaping data matrix from: {} to {}'.format(image_stack.shape, self.X.shape))

	def calculate_svd(self):
		# Singular value decomposition
		self.U, self.S, self.VT = np.linalg.svd(self.X,full_matrices=False) #compute economic SVD
		#S- eigenvalues
		self.S_diag = np.diag(self.S) # create diagonal matrix for multiplications

		self.have_full_svd = True ### this resets to False when loading from file

		print('##########################\n')
		print('A = S x U x V.T')
		print('U shape: {}'.format(self.U.shape))
		print('S shape: {}'.format(self.S.shape))
		print('S_diag shape: {}'.format(self.S_diag.shape))
		print('VT shape: {}'.format(self.VT.shape))
		print('\n##########################\n')

	def plot_metrics(self,truncated=False):

		### this works when we have full SVD
		### If SVD is loaded from the file only trucated matrixes are loaded
		### need to add switch for that


		if truncated == False:  ## if ve have SVD plot up to maximum rank (size of data matrix)
			plt.figure(1)
			plt.plot(self.S/np.sum(self.S),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(2)
			plt.semilogy(self.S/np.sum(self.S),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(3)
			plt.semilogy(np.cumsum(self.S) / np.sum(self.S),'o')
			plt.title('Singular Values: Cumulative sum')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()

		else: ### use only trucated matrixes
			plt.figure(1)
			plt.plot(self.Sr/np.sum(self.Sr),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(2)
			plt.semilogy(self.Sr/np.sum(self.Sr),'o')
			plt.title('Singular Values, log plot')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(3)
			plt.semilogy(np.cumsum(self.Sr) / np.sum(self.Sr),'o')
			plt.title('Singular Values: Cumulative sum')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
		


	def get_mode(self, r):


		if self.have_full_svd == True:

			if r > self.NZ: # rank of economy SVD is NZ
				print('Error: Mode number above the range. Max. number of modes:{}'.format(self.NZ))
				sys.exit(0)

			# Use matrixes from full SVD
			Xr = self.U[:,r:r+1] @ self.S_diag[r:r+1,r:r+1] @ self.VT[r:r+1,r:r+1]
			return(Xr.reshape(self.NX, self.NY))

		else:

			if r > self.r: # rank of economy SVD is NZ
				print('Error: Mode number above the range. Max. number of modes:{}'.format(self.r))
				sys.exit(0)

			# Use matrixes from full SVD
			Xr = self.Ur[:,r:r+1] @ self.Sr_diag[r:r+1,r:r+1] @ self.VTr[r:r+1,r:r+1]
			return(Xr.reshape(self.NX, self.NY))


	def truncate(self, r):

		"""
		Trucate matrixes

		"""

		self.r = r  # set trucation rank for model 
		
		#### truncate matrixes
		self.Ur = self.U[:,0:self.r]
		self.Sr = self.S[0:self.r]
		self.Sr_diag = self.S_diag[0:self.r,0:self.r]
		self.VTr = self.VT[0:self.r,0:self.r]
		
	def get_truncated_model(self):

		self.Xr = self.Ur @ self.Sr_diag @ self.VTr ### data still reshaped in colums
		print('X approx shape: {} '.format(self.Xr.shape))

		
		### this is reshaped to original coordinates
		self.X_model = (self.Xr.T ).reshape(self.r, self.NX,self.NY) # reshaping to original

		self.have_truncated_matrixes = True
		
	
	def reconstruct(self, Image):
		"""
		Image to reconstruct using library stored 
		r - is the rank to use from library 
		"""
		NX, NY = Image.shape ### this should be same size as library
		### add if else 

		X = Image.flatten()
		#print('X.shape: {}'.format(X.shape))
		#print('Ur shape: {}'.format(self.Ur.shape))
		#print('Selecting rank: {}'.format(r))
		#Ur = np.copy(self.Ur[:,0:r])
		#print('Ur shape: {}'.format(Ur.shape))
		UrT = self.Ur.T
		#print('UrT shape: {}'.format(UrT.shape))

		alpha = UrT @ X
		#print('alpha shape: {}'.format(alpha.shape))

		tmp = self.X_mean + self.Ur @ alpha

		return(tmp.reshape(NX,NY))


	def plot_modes_sum(self,r1,r2):

		"""
		sum r1 to r3 modes and plot
		"""
		X_model_sum = np.sum(self.X_model[r1:r2,:],axis=0)
		plt.figure()
		plt.imshow(X_model_sum)
		plt.colorbar(orientation='vertical',fraction=0.03, pad=0.01)
		plt.show() 

	def plot_modes(self):

		for i in range(self.r):

			Xmodel_v = self.Ur[:,i:i+1] @ self.Sr_diag[i:i+1,i:i+1] @ self.VTr[i:i+1,i:i+1]
			Xmodel = Xmodel_v.reshape(self.NX, self.NY)
			plt.figure()
			plt.imshow(Xmodel,cmap='bone')
			plt.title('Mode: {}'.format(i))
			plt.colorbar(orientation='vertical',fraction=0.03, pad=0.01)
			plt.show()


	def save_truncated_model(self, file):
		"""
			This saves SVD library into hdf5 file 
		"""
		
		print("Trying to create file %s" % (file) )

		try:
			self.hf = h5py.File(file,'a')
		except Exception as e:
			print(e)
		
		
		#Original space of the dataset
		self.hf.create_dataset('NX', data=self.NX)
		self.hf.create_dataset('NY', data=self.NY)
		self.hf.create_dataset('NZ', data=self.NZ)

		# Truncation rank r: 
		self.hf.create_dataset('r',data=self.r)
		

		# mean value for reconstruction A0
		self.hf.create_dataset('X_mean',data=self.X_mean)
		
		#truncated matrixes: 
		self.hf.create_dataset('Ur',data=self.Ur)
		self.hf.create_dataset('Sr',data=self.Sr)
		self.hf.create_dataset('Sr_diag',data=self.Sr_diag)
		self.hf.create_dataset('VTr',data=self.VTr)


		print('Saving done ...')
		self.hf.close()

	def read_truncated_model(self,file):
		print("Trying to read file %s" % (file) )

		try:
			self.hf = h5py.File(file,'r')
		except Exception as e:
			print(e)

		self.NX = self.hf['NX'][()]
		self.NY = self.hf['NY'][()]
		self.NZ = self.hf['NZ'][()]
		
		self.r = self.hf['r'][()]
		self.X_mean = self.hf['X_mean'][()]

		self.Ur = self.hf['Ur'][()]
		self.Sr = self.hf['Sr'][()]
		self.Sr_diag = self.hf['Sr_diag'][()]
		self.VTr = self.hf['VTr'][()]

		print('Reading done')
		self.hf.close()

		self.have_full_svd=False
		self.have_truncated_matrixes=True




class SVD_1D(object):
	"""
		input is 1D data for example from gotthard
	"""
	


	def __init__(self):
		self.U = 0
		self.S = 0
		self.VT = 0
		self.have_truncated_matrixes = False
		self.have_full_svd = False

	def set_data_matrix(self, image_stack):
		"""
		 expecting [Nz,Nx,Ny] coordinates

		"""
		self.NZ,self.NX,self.NY = image_stack.shape
		self.X = image_stack.reshape(self.NZ,-1).T
		self.X_mean = self.X.mean()

		self.X = self.X - self.X_mean  #### center the data 

		print('Reshaping data matrix from: {} to {}'.format(image_stack.shape, self.X.shape))

	def calculate_svd(self):
		# Singular value decomposition
		self.U, self.S, self.VT = np.linalg.svd(self.X,full_matrices=False) #compute economic SVD
		#S- eigenvalues
		self.S_diag = np.diag(self.S) # create diagonal matrix for multiplications

		self.have_full_svd = True ### this resets to False when loading from file

		print('##########################\n')
		print('A = S x U x V.T')
		print('U shape: {}'.format(self.U.shape))
		print('S shape: {}'.format(self.S.shape))
		print('S_diag shape: {}'.format(self.S_diag.shape))
		print('VT shape: {}'.format(self.VT.shape))
		print('\n##########################\n')

	def plot_metrics(self,truncated=False):

		### this works when we have full SVD
		### If SVD is loaded from the file only trucated matrixes are loaded
		### need to add switch for that


		if truncated == False:  ## if ve have SVD plot up to maximum rank (size of data matrix)
			plt.figure(1)
			plt.plot(self.S/np.sum(self.S),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(2)
			plt.semilogy(self.S/np.sum(self.S),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(3)
			plt.semilogy(np.cumsum(self.S) / np.sum(self.S),'o')
			plt.title('Singular Values: Cumulative sum')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()

		else: ### use only trucated matrixes
			plt.figure(1)
			plt.plot(self.Sr/np.sum(self.Sr),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(2)
			plt.semilogy(self.Sr/np.sum(self.Sr),'o')
			plt.title('Singular Values')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
	
			plt.figure(3)
			plt.semilogy(np.cumsum(self.Sr) / np.sum(self.Sr),'o')
			plt.title('Singular Values: Cumulative sum')
			plt.grid(True, which="both")
			plt.xlabel('Mode number')
			plt.show()
		


	def get_mode(self, r):


		if self.have_full_svd == True:

			if r > self.NZ: # rank of economy SVD is NZ
				print('Error: Mode number above the range. Max. number of modes:{}'.format(self.NZ))
				sys.exit(0)

			# Use matrixes from full SVD
			Xr = self.U[:,r:r+1] @ self.S_diag[r:r+1,r:r+1] @ self.VT[r:r+1,r:r+1]
			return(Xr.reshape(self.NX, self.NY))

		else:

			if r > self.r: # rank of economy SVD is NZ
				print('Error: Mode number above the range. Max. number of modes:{}'.format(self.r))
				sys.exit(0)

			# Use matrixes from full SVD
			Xr = self.Ur[:,r:r+1] @ self.Sr_diag[r:r+1,r:r+1] @ self.VTr[r:r+1,r:r+1]
			return(Xr.reshape(self.NX, self.NY))


	def truncate(self, r):

		"""
		Trucate matrixes

		"""

		self.r = r  # set trucation rank for model 
		
		#### truncate matrixes
		self.Ur = self.U[:,0:self.r]
		self.Sr = self.S[0:self.r]
		self.Sr_diag = self.S_diag[0:self.r,0:self.r]
		self.VTr = self.VT[0:self.r,0:self.r]
		
	def get_truncated_model(self):

		self.Xr = self.Ur @ self.Sr_diag @ self.VTr ### data still reshaped in colums
		print('X approx shape: {} '.format(self.Xr.shape))

		
		### this is reshaped to original coordinates
		self.X_model = (self.Xr.T ).reshape(self.r, self.NX,self.NY) # reshaping to original

		self.have_truncated_matrixes = True
		
	
	def reconstruct(self, Image,r):
		"""
		Image to reconstruct using library stored 
		r - is the rank to use from library 
		"""
		NX, NY = Image.shape ### this should be same size as library
		### add if else 

		X = Image.flatten()
		#print('X.shape: {}'.format(X.shape))
		#print('Ur shape: {}'.format(self.Ur.shape))
		#print('Selecting rank: {}'.format(r))
		Ur = np.copy(self.Ur[:,0:r])
		#print('Ur shape: {}'.format(Ur.shape))
		UrT = Ur.T
		#print('UrT shape: {}'.format(UrT.shape))

		alpha = UrT @ X
		#print('alpha shape: {}'.format(alpha.shape))

		tmp = self.X_mean + Ur @ alpha

		return(tmp.reshape(NX,NY))


	def plot_modes_sum(self):

		"""
		sum zero to r-1 modes and plot
		"""
		X_model_sum = np.sum(self.X_model,axis=0)
		plt.figure()
		plt.imshow(X_model_sum)
		plt.show() 

	def plot_modes(self):

		for i in range(self.r):

			Xmodel_v = self.Ur[:,i:i+1] @ self.Sr_diag[i:i+1,i:i+1] @ self.VTr[i:i+1,i:i+1]
			Xmodel = Xmodel_v.reshape(self.NX, self.NY)
			plt.figure()
			plt.imshow(Xmodel,cmap='bone')
			plt.title('Mode: {}'.format(i))
			plt.colorbar()
			plt.show()


	def save_truncated_model(self, file):
		"""
			This saves SVD library into hdf5 file 
		"""
		
		print("Trying to create file %s" % (file) )

		try:
			self.hf = h5py.File(file,'a')
		except Exception as e:
			print(e)
		
		
		#Original space of the dataset
		self.hf.create_dataset('NX', data=self.NX)
		self.hf.create_dataset('NY', data=self.NY)
		self.hf.create_dataset('NZ', data=self.NZ)

		# Truncation rank r: 
		self.hf.create_dataset('r',data=self.r)
		

		# mean value for reconstruction A0
		self.hf.create_dataset('X_mean',data=self.X_mean)
		
		#truncated matrixes: 
		self.hf.create_dataset('Ur',data=self.Ur)
		self.hf.create_dataset('Sr',data=self.Sr)
		self.hf.create_dataset('Sr_diag',data=self.Sr_diag)
		self.hf.create_dataset('VTr',data=self.VTr)


		print('Saving done ...')
		self.hf.close()

	def read_truncated_model(self,file):
		print("Trying to read file %s" % (file) )

		try:
			self.hf = h5py.File(file,'r')
		except Exception as e:
			print(e)

		self.NX = self.hf['NX'][()]
		self.NY = self.hf['NY'][()]
		self.NZ = self.hf['NZ'][()]
		
		self.r = self.hf['r'][()]
		self.X_mean = self.hf['X_mean'][()]

		self.Ur = self.hf['Ur'][()]
		self.Sr = self.hf['Sr'][()]
		self.Sr_diag = self.hf['Sr_diag'][()]
		self.VTr = self.hf['VTr'][()]

		print('Reading done')
		self.hf.close()

		self.have_full_svd=False
		self.have_truncated_matrixes=True












