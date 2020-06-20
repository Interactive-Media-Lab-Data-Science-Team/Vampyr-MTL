import numpy as np
from .init_opts import init_opts
from numpy import linalg as LA
from tqdm import tqdm
from tqdm import trange
import sys
import time
from ..evaluations.utils import opts


class MTL_Softmax_L21:
	def __init__(self, opts, rho1=0.01):
		"""
			rho1: L2,1-norm group Lasso parameter
			opts: config variables
		"""
		self.opts = init_opts(opts)
		self.rho1 = rho1
		self.rho_L2 = 0
		if hasattr(opts, 'rho_L2'):
			rho_L2 = opts.rho_L2

	def fit(self, X, Y, **kwargs):
		"""
			X: np.array: t x n x d
			Y: np.array t x n x 1, as type == int or obj
		"""
		
		self.task_num = len(X)
		_, self.dimension = X[0].shape
		task_num = self.task_num
		dimension = self.dimension
        
		if 'rho' in kwargs.keys():
			print(kwargs)
			self.rho1 = kwargs['rho']
		X_new = []
		for i in range(len(X)):
			X_new.append(np.transpose(X[i]))
		X = X_new

        # y type check:
		dt =Y[0].dtype
		Y_new = []
		m=0
		if(np.issubdtype(dt, np.number)):
			m = max(np.max(i) for i in Y)+1
			self.encode_mtx = None
		elif(np.issubdtype(dt, np.character) or np.issubdtype(dt, np.object)):
			unq = set([])
			for t in range(self.task_num):
				unq = unq.union(set(Y[t]))
			self.encode_mtx = {j:i for i,j in enumerate(unq)} # check encoding seq
			self.decode_mtx = {i:j for i,j in enumerate(unq)} # decode the seq
			m = len(unq)
			# change Y str to num, make a copy of Y to avoid addr change
			Y_cp=[0]*self.task_num
			for t in range(self.task_num):
				tmp_y = []
				for j in Y[t]:
					tmp_y.append(self.encode_mtx[j])
				Y_cp[t] = np.array(tmp_y)
			Y=Y_cp
		else:
			raise TypeError('Invalid target type')
		self.encoding = m
		for t in range(self.task_num):
			s = int(Y[t].size)
			Y_new_t = np.zeros((s, m))
			Y_new_t[np.arange(s),Y[t]] = 1
			Y_new.append(Y_new_t)
		Y_old = Y
		Y = Y_new
		self.X = X
		self.Y = Y
		# transpose X to size: t x d x n
		# encoding Y to size: t x n x m
		funcVal = []

		C0_prep = np.zeros((task_num, self.encoding))
		for t_idx in range(task_num):
			m1 = np.count_nonzero(Y_old[t_idx]==1)
			m2 = np.count_nonzero(Y_old[t_idx]==-1)
			if(m1==0 or m2==0):
				# inbalanced label
				C0_prep[t_idx] = 0
			else:
				C0_prep[t_idx] = np.log(m1/m2)

		if self.opts.init==2:
			W0 = np.zeros((dimension, task_num, self.encoding))
			C0 = np.zeros((task_num, self.encoding))
		elif self.opts.init == 0:
			W0 = np.random.randn(dimension, task_num, self.encoding)
			C0 = C0_prep
		else: 
			if hasattr(self.opts,'W0'):
				W0=self.opts.W0 
				if(W0.shape!=(dimension, task_num, self.encoding)):
					raise TypeError('\n Check input W0 size')
			else:
				W0 = np.zeros((dimension, task_num, self.encoding))
			if hasattr(self.opts, 'C0'):
				C0 = self.opts.C0
			else:
				C0 = C0_prep

		# this flag checks if gradient descent only makes significant step
		bFlag=0 

		Wz= W0
		Cz = C0
		Wz_old = W0
		Cz_old = C0

		t = 1
		t_old = 0

		gamma = 1
		gamma_inc = 2

		for it in trange(self.opts.maxIter, file=sys.stdout, desc='outer loop'):
			alpha = (t_old - 1)/t

			Ws = (1 + alpha) * Wz - alpha * Wz_old
			Cs = (1 + alpha) * Cz - alpha * Cz_old

			gWs, gCs, Fs = self.gradVal_eval(Ws, Cs)

			for in_it in trange(1000,file=sys.stdout, leave=False, unit_scale=True, desc='inner loop'):
				Wzp = self.FGLasso_projection(Ws - gWs/gamma, self.rho1 / gamma)
				Czp = Cs - gCs/gamma
				Fzp = self.funVal_eval(Wzp, Czp)

				delta_Wzp = Wzp - Ws
				delta_Czp = Czp - Cs
				nrm_delta_Wzp = LA.norm(delta_Wzp)**2
				nrm_delta_Czp = LA.norm(delta_Czp)**2
				r_sum = (nrm_delta_Czp + nrm_delta_Wzp)/2

				Fzp_gamma = Fs + np.sum(delta_Wzp*gWs) + np.sum(delta_Czp*gCs)+ gamma/2 * r_sum*2
				if (r_sum <=1e-20):
					bFlag=1 # this shows that, the gradient step makes little improvement
					break
				if (Fzp <= Fzp_gamma):
					break
				else:
					gamma = gamma * gamma_inc
					
			Wz_old = Wz
			Cz_old = Cz
			Wz = Wzp
			Cz = Czp

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
		'''
			D: dimension = d x t x m

		'''
		# l2.1 norm projection.
		ss = np.sum(D**2,axis=1)
		sq = np.sqrt(ss.astype(float))
		# of shape: d x m, sum in direction of task t
		tmp = np.tile(np.maximum(0, 1 - lmbd/sq),(D.shape[1], 1, 1)).transpose(1,0,2)
		return tmp*D

	# smooth part gradient.
	def gradVal_eval(self, W, C):
		grad_W = np.zeros((self.dimension, self.task_num, self.encoding))
		grad_C = np.zeros((self.task_num, self.encoding))
		lossValVect = np.zeros((1, self.task_num)) 
		if self.opts.pFlag:
			# grad_W = zeros(zeros(W));
			# parfor i = 1:task_num
				# grad_W (i, :) = X{i}*(X{i}' * W(:,i)-Y{i})
			pass
		else:
			for i in range(self.task_num):
				grad_W[:,i], grad_C[i], lossValVect[:,i] = self.unit_grad_eval(W[:,i], C[i], i)
			grad_W = grad_W+ self.rho_L2 * 2 * W
			funcVal = np.sum(lossValVect) + self.rho_L2 * LA.norm(W)**2
			return grad_W, grad_C, funcVal

	def funVal_eval(self, W, C):
		funcVal = 0
		if self.opts.pFlag:
			# parfor i = 1: task_num
			#     funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
			# end
			pass
		else:
			for i in range(self.task_num):
				funcVal = funcVal + self.unit_funcVal_eval(W[:,i], C[i], i)
			funcVal = funcVal + self.rho_L2 * LA.norm(W)**2
		return funcVal

	def nonsmooth_eval(self, W, rho_1):
		non_smooth_value = 0
		if self.opts.pFlag:
			pass
		else:
			for i in range(self.dimension):
				w = W[i, :]
				non_smooth_value = non_smooth_value+ rho_1 * LA.norm(w, ord=2)
			return non_smooth_value

	def unit_grad_eval(self, w, c, task_idx):
		weight = np.ones((1, self.Y[task_idx].shape[0]))/self.task_num
		weighty = 1/self.task_num * self.Y[task_idx]
		_, n = self.X[task_idx].shape
		z = -self.Y[task_idx]*(np.transpose(self.X[task_idx])@w + np.tile(c, (n, 1)))
		hinge = np.maximum(z, 0)
		funcVal = np.sum((weight @ (np.log(np.exp(-hinge)+np.exp(z-hinge))+hinge)))
		prob = 1./(1+np.exp(z))
		z_prob = -weighty*(1-prob)
		grad_c = np.sum(z_prob)
		grad_w = self.X[task_idx]@z_prob
		return grad_w, grad_c, funcVal

	def unit_funcVal_eval(self, w, c, task_idx):
		weight = np.ones((1, self.Y[task_idx].shape[0]))/self.task_num
		z = -self.Y[task_idx]*(np.transpose(self.X[task_idx])@w + c)
		hinge = np.maximum(z, 0)
		funcVal = np.sum(weight @ (np.log(np.exp(-hinge)+np.exp(z-hinge))+hinge))
		return funcVal

	def get_params(self, deep = False):
		return {'rho1':self.rho1, 'opts':self.opts}

	def _trained_parames(self):
		return self.W, self.funcVal

	def softmax(self, z):
		z -= np.max(z)
		sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
		return sm

	def predict(self, X):
		pred = []
		for i in range(self.task_num):
			pp = np.reshape(X[i], (-1, self.dimension)) @ self.W[:, i]
			probs = self.softmax(pp)
			preds = np.argmax(probs,axis=1)
			if(self.encode_mtx!=None):
				temp = []
				for p in preds:
					temp.append(self.decode_mtx[p])
				preds = np.array(temp)
			pred.append(preds)
		return pred

	def score(self, X, Y):
		pred = self.predict(X)
		correct = 0
		total = 0
		for i, j in zip(Y, pred):
			for k,l in zip(i,j):
				if(k == l):
					correct+=1
				total+=1
		acc = correct/total
		return acc





