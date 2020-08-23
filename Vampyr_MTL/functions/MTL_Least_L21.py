"""
.. module:: MTL_Least_L21
   :synopsis: MTL linear regression
.. moduleauthor:: Max J <https://github.com/DaPraxis>
"""
import numpy as np
from .init_opts import init_opts
from numpy import linalg as LA
from tqdm import tqdm
from tqdm import trange
import sys
import time

class MTL_Least_L21:
	""" MTL algorithm with least square regression and L21 penalty
	"""
	def __init__(self, opts, rho1=0.01):
		"""Initialization of MTL function

        Args:
            opts (opts): initalization class from opts
            rho1 (int, optional): L2,1-norm group Lasso parameter. Defaults to 0.01.
        """
		self.opts = init_opts(opts)
		self.rho1 = rho1
		self.rho_L2 = 0
		if hasattr(opts, 'rho_L2'):
			rho_L2 = opts.rho_L2

	def fit(self, X, Y, **kwargs):
		"""Fit with training samples and train
  
        t: task number
        
        n: number of entries
        
        d: data dimension

        Args:
            X ([np.array(np.array)]): t x n x d
            Y ([np.array(np.array)]): t x n x 1
        """
		if 'rho' in kwargs.keys():
			print(kwargs)
			self.rho1 = kwargs['rho']
		X_new = []
		for i in range(len(X)):
			X_new.append(np.transpose(X[i]))
		X = X_new
		self.X = X
		self.Y = Y
		# transpose to size: t x d x n

		self.task_num = len(X)
		self.dimension, _ = X[0].shape
		funcVal = []

		if hasattr(self.opts,'W0'):
			W0=self.opts.W0
		elif self.opts.init==2:
			W0 = np.zeros((self.dimension, self.task_num))
		elif self.opts.init == 0:
			W0 = np.zeros((self.dimension, self.task_num))
			for t_idx in range(self.task_num):
				W0[:,t_idx] = (X[t_idx]@Y[t_idx]).flatten()
		else:
			W0 = np.random.normal(0, 1, (self.dimension, self.task_num))

		# this flag checks if gradient descent only makes significant step
		bFlag=0 

		Wz= W0
		Wz_old = W0

		t = 1
		t_old = 0

		it = 0
		gamma = 1
		gamma_inc = 2

		for it in trange(self.opts.maxIter, file=sys.stdout, desc='Training'):
			alpha = (t_old - 1)/t
			Ws = (1 + alpha) * Wz - alpha * Wz_old
			# compute function value and gradients of the search point
			gWs = self.gradVal_eval(Ws)
			Fs = self.funVal_eval(Ws)
			in_it = 0

			while True:
				Wzp = self.FGLasso_projection(Ws - gWs/gamma, self.rho1 / gamma)
				Fzp = self.funVal_eval(Wzp)
				delta_Wzp = Wzp - Ws
				r_sum = LA.norm(delta_Wzp)**2
				Fzp_gamma = Fs + np.sum(delta_Wzp*gWs)+ gamma/2 * r_sum
				if (r_sum <=1e-20):
					bFlag=1 # this shows that, the gradient step makes little improvement
					break
				if (Fzp <= Fzp_gamma):
					break
				else:
					gamma = gamma * gamma_inc
					
			Wz_old = Wz
			Wz = Wzp
			funcVal.append(Fzp + self.nonsmooth_eval(Wz, self.rho1))

			if (bFlag):
				print('\n The program terminates as the gradient step changes the solution very small.')
				break
			
			if(self.opts.tFlag == 0):
				if it>=2:
					if (abs( funcVal[-1] - funcVal[-2]) <= self.opts.tol):
						break
					
			elif(self.opts.tFlag == 1):
				if it>=2:
					if (abs( funcVal[-1] - funcVal[-2] ) <= self.opts.tol* funcVal[-2]):
						break
					
			elif(self.opts.tFlag == 2):
				if ( funcVal[-1]<= self.opts.tol):
					break
				
			elif(self.opts.tFlag == 3):
				if it>=self.opts.maxIter:
					break
			
			t_old = t
			t = 0.5 * (1 + (1+ 4 * t**2)**0.5)

		self.W = Wzp
		self.funcVal = funcVal

	def FGLasso_projection (self, D, lmbd):
		"""Lasso projection for panelties

		Args:
			D (np.array(np.array)): Weight matrix
			lmbd (int): panelties param

		Returns:
			(np.array(np.array)): panelties
		"""
		# l2.1 norm projection.
		ss = np.sum(D**2,axis=1)
		sq = np.sqrt(ss.astype(float))
		tmp = np.tile(np.maximum(0, 1 - lmbd/sq),(D.shape[1], 1))
		return np.transpose(tmp)*D

	# smooth part gradient.
	def gradVal_eval(self, W):
		"""Gradient Decent

			Args:
				W (np.array(np.array)): Weight Matrix with shape (d, t)

			Returns:
				grad_W (np.array(np.array)): gradient matrix of weight, shape (d, t)
        """
		if self.opts.pFlag:
			# grad_W = zeros(zeros(W));
			# parfor i = 1:task_num
				# grad_W (i, :) = X{i}*(X{i}' * W(:,i)-Y{i})
			pass
		else:
			grad_W = []
			for i in range(self.task_num):
				g_w = np.reshape(self.X[i]@(np.reshape(np.transpose(self.X[i])@W[:,i], (self.Y[i].shape[0], 1))-np.reshape(self.Y[i], (self.Y[i].shape[0], 1))), (self.dimension,-1))
				grad_W.append(g_w)
			grad_W = np.concatenate(grad_W, axis=1)+ self.rho_L2 * 2 * W
			return grad_W

	def funVal_eval(self, W):
		"""Loss accumulation

        Args:
            W (np.array(np.array)): weight matrix of shape (d, t)

        Returns:
            funcval (float): loss
        """
		funcVal = 0
		if self.opts.pFlag:
			# parfor i = 1: task_num
			#     funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
			# end
			pass
		else:
			for i in range(self.task_num):
				funcVal = funcVal + 0.5 * LA.norm ((self.Y[i] - np.transpose(self.X[i]) @ W[:, i]), ord=2)**2
			funcVal = funcVal + self.rho_L2 * LA.norm(W)**2
		return funcVal

	def nonsmooth_eval(self, W, rho_1):
		"""non-smooth loss evaluation

		Args:
			W (np.array(np.array)): weight matrix of shape (d, t)
			rho1 (float): L2,1-norm group Lasso parameter

		Returns:
			(float): loss 
		"""
		non_smooth_value = 0
		if self.opts.pFlag:
			pass
		else:
			for i in range(W.shape[0]):
				w = W[i, :]
				non_smooth_value = non_smooth_value+ rho_1 * LA.norm(w, ord=2)
			return non_smooth_value

	def get_params(self, deep = False):
		"""Get inbult initalization params

		Args:
			deep (bool, optional): deep traverse. Defaults to False.

		Returns:
			(dict): dictionary of all inits
		"""
		return {'rho1':self.rho1, 'opts':self.opts}

	def predict(self, X):
		"""Predict with test data

		Args:
			X [(np.array(np.array))]: input to predict, shape (t, n, d)

		Returns:
			([np.array()]): predict matrix, shape (t, n ,1)
		"""
		pred = []
		for i in range(self.task_num):
			pred.append(np.reshape(X[i], (-1, self.dimension)) @ self.W[:, i])
		return pred



