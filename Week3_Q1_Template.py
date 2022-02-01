import numpy as np
from scipy import optimize
import pandas as pd
from matplotlib import pyplot

class lr:
    # Data cleaning and finding the mean of the column titled "MaxTemp"
    def data_clean(self,data):
        # 'data' is a dataframe imported from '.csv' file using 'pandas'
        # Perform data cleaning steps sequentially as mentioned in assignment
        f = list(data.head(0))
        y = data['RainTomorrow'].replace({'No':0}).replace({'Yes':1})
        y = y.fillna(y.mean()) 
        l = list(enumerate(data.dtypes))
        l2 = []
        for i in range(len(l)):
          if l[i][1] == object:
            l2.append(l[i][0])
        df = data.drop([f[i] for i in l2],axis = 1)
        df = df.fillna(df.mean())
        df = (df-df.min())/(df.max()-df.min())
        X = np.array(df)            # X (Feature matrix) - should be numpy array
        mean = df['MaxTemp'].mean() # Mean of a the normalized "MaxTemp" column rounded off to 3 decimal places
    
        return X, y, mean

class costing:
  # define the function needed to evaluate cost function
  # Input 'z' could be a scalar or a 1D vector
  def sigmoid(self,z):
    g = 1/(1+np.exp(-1*z))
    return g

  # Regularized cost function definition
  def costFunctionReg(self,w,X,y,lambda_):
    p = len(w)
    m = len(X)
    h = costing().sigmoid(X@w)
    w0 = w.copy()
    w0[0] = 0
    t1 = y.transpose()@np.log(h) + (1-y).transpose()@np.log(1-h)
    t2 = (w0.transpose()@w0)/2
    J = (t2 - t1)/m                                         # Cost 'J' should be a scalar
    grad = (1/m)*(X.transpose()@(h-y) + lambda_*w0)         # Gradient 'grad' should be a vector
    return J, grad

    # Prediction based on trained model
    # Use sigmoid function to calculate probability rounded off to either 0 or 1
  def predict(self,w,X):
    h = costing().sigmoid(X@w)
    p = np.zeros(len(h))            # 'p' should be a vector of size equal to that of vector 'y'
    for i in range(len(h)):
      if h[i] < 0.5:
        p[i] = 0
      else:
        p[i] = 1  
    return p
  
    # Optimization defintion
  def minCostFun(self, w_ini, X_train, y_train, iters):
    # iters - Maximum no. of iterations; X_train - Numpy array
    lambda_ = 0.1  # Regularization parameter
    X_train = np.vstack([np.ones(len(X_train)),X_train.transpose()]).transpose()     # Add '1' for bias term

    def f(w):
      return costing().costFunctionReg(w,X_train,y_train,iters)[0]

    res = optimize.minimize(f,w_ini, method = 'TNC',options = {'maxiter' : iters})
        
    w_opt = res.x   # Optimized weights rounded off to 3 decimal places
    p = costing().predict(w_opt,X_train)
    ones = np.ones(len(p))
    e = ones.transpose()@abs(p-y_train)/len(X_train)
    acrcy = (1-e)*100  # Training set accuracy (in %) rounded off to 3 decimal places    
    return w_opt, acrcy
  
    # Calculate testing accuracy
  def TestingAccu(self, w_opt, X_test, y_test):
    w_opt = np.array([-2.40348317,  0.20589886, -0.0065877 , -0.67264944, -1.54570833,
                      -1.88480232,  3.74834075, -0.08465718, -0.89431186,  0.86854556,
                       4.33979133, -1.6378398 , -1.90737076,  0.19323422,  0.92297631,
                       0.06246831, -0.30190665])        
    # Optimum weights calculated using training data
    X_test = np.vstack([np.ones(len(X_test)),X_test.transpose()]).transpose()       # Add '1' for bias term
    p = costing().predict(w_opt,X_test)
    ones = np.ones(len(p))
    e = ones.transpose()@abs(p-y_test)/len(X_test)
    acrcy_test = (1-e)*100   # Testing set accuracy (in %) rounded off to 3 decimal places
        
    return acrcy_test  
