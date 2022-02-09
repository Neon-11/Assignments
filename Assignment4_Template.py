import numpy as np
from scipy import optimize
import pandas as pd

class costing:
  def data_clean(self,data):         # 'data' is a pandas dataframe
     y = np.array(data['label'])     # numpy array
     data = data.drop('label',axis = 1)
     df = data.fillna(data.mean())
     f = list(df.head(0))
     for i in range(len(f)):
       if df[f[i]].max() != df[f[i]].min():
         df[f[i]] = (df[f[i]]-df[f[i]].min())/(df[f[i]].max()-df[f[i]].min())
     X = np.array(df)                # Normalized 'X' (numpy array)   
     return X, y

  def sigmoid(self,z):
    g = 1/(1+np.exp(-1*z))
    return g
    
    # Regularized cost function definition
  def costFunctionReg(self,w,X,y,lambda_):
    m = np.shape(X)[0]
    h = self.sigmoid(X@w)
    w[0] = 0
    t1 = y.T@np.log(h) + (1-y).T@np.log(1-h)
    t2 = np.sum(np.square(w))/2
    J = (t2 - t1)/(m)                              # Cost 'J' should be a scalar
    grad = np.array((X.T@(h-y) + lambda_*(w))/m)   # Gradient 'grad' should be a vector
    return J, grad
   
    # Prediction based on trained model
    # Use sigmoid function to calculate probability rounded off to either 0 or 1
  def predictOneVsAll(self,all_w,X,num_labels):
    h = np.round(self.sigmoid(X@all_w.transpose()))
    p = np.zeros(np.shape(h)[0])     # 'p' should be a vector of size equal to that of vector 'y'
    for i in range(np.shape(h)[0]):
      for j in range(num_labels):
        if h[i][j] != 0:
          p[i] = j  
    return p
 
    # Optimization defintion
  def minCostFun(self, train_data): #'train_data' is a pandas dataframe
    lambda_ = 0.1                   # Regularization parameter
    iters = 4000
    X_train,y_train = self.data_clean(train_data)
    X = np.vstack([np.ones(np.shape(X_train)[0]),X_train.T]).T    # Add '1' for bias term
    w = np.array([])
    li = [[] for i in range(10)]
    for k in range(10):
      y = np.array(y_train == k).astype(int)
      wk = np.zeros(785)
      i = 1
      def callbackF(xi):
        global i
        li[k].append(self.costFunctionReg(xi,X,y,lambda_)[0])
      res = optimize.minimize(self.costFunctionReg,wk,(X,y,lambda_),method = 'CG',jac = True,callback = callbackF,options = {'maxiter' :iters})
      if k == 0:
        w = np.vstack([res.x])
      else:
        w = np.vstack([w,res.x])
        
    global all_w
    all_w = w                                   # Optimized weights (size = 10 X 785) rounded off to 3 decimal places
    p = self.predictOneVsAll(all_w,X,10)
    a = np.array(p == y_train).astype(int)
    acrcy = (np.sum(a)/np.shape(a)[0])*100      # Training set accuracy (in %) rounded off to 3 decimal places (Ans ~ 93.2)   
    return all_w, acrcy

    # Calculate testing accuracy
  def TestingAccu(self, test_data): #'test_data' is a pandas dataframe
    Xt,yt = costing().data_clean(test_data)
    Xt = np.vstack([np.ones(np.shape(Xt)[0]),Xt.T]).T    # Add '1' for bias term
    w = all_w
    p = self.predictOneVsAll(w,Xt,10)
    a = np.array(p == yt).astype(int)
    acrcy_test = (np.sum(a)/np.shape(a)[0])*100 # Training set accuracy (in %) rounded off to 3 decimal places (Ans ~ 86.667)
    return acrcy_test
