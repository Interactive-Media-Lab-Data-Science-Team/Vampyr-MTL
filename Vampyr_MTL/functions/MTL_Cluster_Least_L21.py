import numpy as np
from .init_opts import init_opts
from numpy import linalg as LA
from tqdm import tqdm
from tqdm import trange
import sys
import time
from scipy.sparse import identity
from scipy import linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import isspmatrix

class MTL_Cluster_Least_L21:
    def __init__(self, opts, k, rho1=10, rho2=0.2):
        self.opts = init_opts(opts)
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho_L2 = 0
        self.k = k
        if hasattr(opts, 'rho_L2'):
            rho_L2 = opts.rho_L2
    
    def fit(self, X, Y, **kwargs):
        """
        X: np.array: t x n x d
        Y: np.array t x n x 1
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
        self.eta = self.rho2/self.rho1
        self.c = self.rho1 * self.eta * (1+self.eta)
        funcVal = []
        
        self.XY = [0]* self.task_num
        W0_prep = []
        for t in range(self.task_num):
            self.XY[i] = X[i] @ Y[i]
            W0_prep.append(self.XY[i].reshape((-1,1)))
        W0_prep = np.hstack(W0_prep)
        if hasattr(self.opts,'W0'):
            W0=self.opts.W0
        elif self.opts.init==2:
            W0 = np.zeros((self.dimension, self.task_num))
        elif self.opts.init == 0:
            W0 =W0_prep
        else:
            W0 = np.random.normal(0, 1, (self.dimension, self.task_num))
            
            
        M0 = np.array(identity(self.task_num)) * self.k / self.task_num
        # this flag checks if gradient descent only makes significant step
        
        bFlag=0 
        Wz= W0
        Wz_old = W0
        Mz = M0.toarray()
        Mz_old = M0.toarray()
        
        t = 1
        t_old = 0
        
        it = 0
        gamma = 1.0
        gamma_inc = 2
        
        for it in trange(self.opts.maxIter, file=sys.stdout, desc='outer loop'):
            alpha = (t_old - 1)/t
            Ws = (1 + alpha) * Wz - alpha * Wz_old
            if(isspmatrix(Mz)):
                Mz = Mz.toarray()
            if(isspmatrix(Mz_old)):
                Mz_old = Mz_old.toarray()
            Ms = (1 + alpha) * Mz - alpha * Mz_old
            # compute function value and gradients of the search point
            gWs, gMs, Fs = self.gradVal_eval(Ws, Ms)
            
            in_it = 0
            for in_it in trange(1000,file=sys.stdout, leave=False, unit_scale=True, desc='inner loop'):
                Wzp = Ws - gWs/gamma
                Mzp, Mzp_Pz, Mzp_DiagSigz = self.singular_projection (Ms - gMs/gamma, self.k)
                Fzp = self.funVal_eval(Wzp, Mzp_Pz, Mzp_DiagSigz)
                
                delta_Wzs = Wzp - Ws
                delta_Mzs = Mzp - Ms
                
                r_sum = (LA.norm(delta_Wzs)**2 + LA.norm(delta_Mzs)**2)/2
                Fzp_gamma = Fs + np.sum(delta_Wzs*gWs) + np.sum(delta_Mzs*gMs) + gamma * r_sum
                if (r_sum <=1e-20):
                    bFlag=1 # this shows that, the gradient step makes little improvement
                    break
                if (Fzp <= Fzp_gamma):
                    break
                else:
                    gamma = gamma * gamma_inc
            Wz_old = Wz
            Wz = Wzp
            Mz_old = Mz
            Mz = Mzp
            funcVal.append(Fzp)
            
            if (bFlag):
                print('\n The program terminates as the gradient step changes the solution very small.')
                break
            if (self.opts.tFlag == 0):
                if it >= 2:
                    if (abs(funcVal[-1] - funcVal[-2]) <= self.opts.tol):
                        break
                    
            elif(self.opts.tFlag == 1):
                if it >= 2:
                    if (abs(funcVal[-1] - funcVal[-2]) <= self.opts.tol * funcVal[-2]):
                        break
                    
            elif(self.opts.tFlag == 2):
                if (funcVal[-1] <= self.opts.tol):
                    break
                
            elif(self.opts.tFlag == 3):
                if it >= self.opts.maxIter:
                    break
                
            t_old = t
            t = 0.5 * (1 + (1 + 4 * t ** 2) ** 0.5)
            
        self.W = Wzp
        self.M = Mzp
        self.funcVal = funcVal
    
    def singular_projection (self, Msp, k):
        """[summary]

        Args:
            Msp ([type]): [description]
            k ([type]): [description]

        Returns:
            [type]: [description]
        """
        # l2.1 norm projection.
        EValue, EVector = linalg.eig(Msp) 
        Pz = np.real(EVector)
        diag_EValue = np.real(EValue)
        DiagSigz, _, _ = self.bsa_ihb(diag_EValue, np.ones(diag_EValue.shape), k, np.ones(diag_EValue.shape))
        Mzp = Pz @ np.diag(DiagSigz) @ Pz.T
        Mzp_Pz = Pz
        Mzp_DiagSigz = DiagSigz
        return Mzp, Mzp_Pz, Mzp_DiagSigz

    def bsa_ihb(self, a, b, r, u):
        '''
        Singular Projection
        min 1/2*||x - a||_2^2
        s.t. b'*x = r, 0<= x <= u,  b > 0
        '''
        break_flag = 0
        t_l = a/b
        t_u = (a - u)/b
        T = np.concatenate((t_l, t_u), axis=0)
        t_L = -np.Infinity
        t_U = np.Infinity
        g_tL = 0.
        g_tU = 0.
        
        it = 0
        while(len(T)!=0):
            it +=1
            g_t = 0.
            t_hat = np.median(T)
            
            U = t_hat < t_u
            M = (t_u <= t_hat) & (t_hat <= t_l)
            
            if np.sum(U):
                g_t += np.sum(b[U]*u[U])
            if np.sum(M):
                g_t += np.sum(b[M]*(a[M]-t_hat*b[M]))
            if g_t > r:
                t_L = t_hat
                T = T[T>t_hat]
                g_tL = g_t
            elif g_t <r:
                t_U = t_hat
                T = T[T<t_hat]
                g_tU = g_t
            else:
                t_star = t_hat
                break_flag = 1
                break
        if not break_flag:
            eps = g_tU - g_tL
            t_star = t_L - (g_tL - r) * (t_U - t_L)/(eps)
        est = a-t_star * b
        if(np.isnan(est).any()):
            est[np.isnan(est)] = 0
        x_star = np.minimum(u, np.max(est, 0))
        return x_star, t_star, it
    
    def gradVal_eval(self, W, M):
        IM = self.eta * identity(self.task_num)+M
        # could be sparse matrix to solve
        invEtaMWt = linalg.inv(IM) @ W.T
        if self.opts.pFlag:
            # grad_W = zeros(zeros(W));
            # # parfor i = 1:task_num
            # # grad_W (i, :) = X{i}*(X{i}' * W(:,i)-Y{i})
            pass
        else:
            grad_W = []
            for i in range(self.task_num):
                XWi = self.X[i].T @ W[:,i]
                XTXWi = self.X[i] @ XWi
                grad_W.append((XTXWi - self.XY[i]).reshape(-1,1))
            grad_W = np.hstack(grad_W)
            grad_W = grad_W + 2 * self.c * invEtaMWt.T
            W2 = W.T @ W
            grad_M = - self.c * W2@linalg.inv(IM)@linalg.inv(IM)
            
        funcVal = 0
        if self.opts.pFlag:
            pass
        else:
            for i in range(self.task_num):
                funcVal = funcVal + 0.5 * LA.norm ((self.Y[i] - self.X[i].T @ W[:, i]), ord=2)**2
        funcVal = funcVal + self.c * np.trace( W @ invEtaMWt)
        return grad_W, grad_M, funcVal
			
    def funVal_eval(self, W, M_Pz, M_DiagSigz):
        invIM = M_Pz @ (np.diag(1/(self.eta + np.array(M_DiagSigz)))) @ M_Pz.T
        invEtaMWt = invIM @ W.T
        funcVal = 0
        if self.opts.pFlag:
            # parfor i = 1: task_num
            # #     funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            # # end
            pass
        else:
            for i in range(self.task_num):
                funcVal = funcVal + 0.5 * LA.norm ((self.Y[i] - self.X[i].T @ W[:, i]), ord=2)**2
            funcVal = funcVal + self.c * np.trace(W @ invEtaMWt)
        return funcVal

    def get_params(self, deep = False):
        return {'rho1':self.rho1, 'rho2':self.rho2,'opts':self.opts, 'k':self.k}

    def analyse(self):
        # returns correlation matrix
  
        # kmCMTL_OrderedModel = np.zeros(self.W.shape)
        # for i in range(self.k):
        #     clusModel = self.W[:, i:self.task_num*self.k:self.k]
        #     kmCMTL_OrderedModel[:, (i)*self.task_num: (i+1)* self.task_num] = clusModel
        # return 1-np.corrcoef(kmCMTL_OrderedModel)
        return self.W
     



