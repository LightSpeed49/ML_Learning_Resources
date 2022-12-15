import numpy as np

class LocallyWeightedRegression():
    def __init__(self,tow=1.0,epochs=5000,lr=0.001):
        self.tow = tow
        self.epochs = epochs
        self.lr = lr
        
    def fit(self,X,y):
        '''
        X : numpy ndarray with shape (n_samples,n_features)
        y : numpy ndarray with shape (n_samples,)
        '''
        X = X.T
        self.X = np.concatenate([X,np.ones(X.shape[1]).reshape(1,-1)],axis=0).T
        self.y = y.reshape(-1,1)
        
    def predict(self,xq):
        '''xq : ndarray shape - (n_features,)'''
        theta = np.random.normal(loc=0, scale=10, size=self.X.shape[1])
        exp_inp = self.X[:,:-1] - xq
        exp_inp = exp_inp**2 
        exp_inp = exp_inp * (-0.5 / self.tow**2)
        wi = np.exp(exp_inp)
        for epoch in range(self.epochs):
            y_h = self.X @ theta.reshape(-1,1)
            dl = self.loss_derivative(self.y,self.X, wi, theta)
            loss = self.loss_fn(self.y,self.X, wi, theta)
            
            theta = theta - self.lr * dl.mean(axis=0)
            
        return theta @ np.pad(xq,[0,1],constant_values=1)
    
    def loss_derivative(self,y_train,x_train,w,theta):
        return -2 * w * ( y_train - (x_train @ theta.reshape(-1,1)) ) * x_train
    def loss_fn(self,y_train,x_train,w,theta):
        return np.sum(w * ((y_train - (x_train @ theta.reshape(-1,1)))**2))