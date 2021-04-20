import scipy
from scipy.stats import norm, lognorm
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import tensorflow_probability as tfp
import math

class Tobit:
    def __init__(self, df, censored, x, y):
        self.df = df
        self.censored = censored
        self.X = x
        self.y = y
    
    def linear_regression(self, censoring):
        regressor = LinearRegression()
        regressor.fit(np.array(self.df[self.X]), self.df[self.y])
        pred = regressor.predict(np.array(self.df[self.X][self.df[self.censored] == censoring]))
        return pred

    def nll(self, vars, upper_threshold):
        sd = vars[-1]
        beta = vars[:-1]
        x_not_censored = np.array(self.df[self.X][self.df[self.censored] == False])
        y_not_censored = np.array(self.df[self.y][self.df[self.censored] == False])
        x_censored = np.array(self.df[self.X][self.df[self.censored] == True])
        y_censored = np.array(self.df[self.y][self.df[self.censored] == True])
        
        
        #print(np.dot(x_censored, beta))
        #print((np.dot(x_censored, beta) - y_censored)/sd)
        ll_censored = scipy.stats.norm.logcdf((np.dot(x_censored, beta) - y_censored)/sd).sum()
        ll_not_censored = (scipy.stats.norm.logpdf(y_not_censored - np.dot(x_not_censored, beta)/sd) - math.log(max(np.finfo('float').resolution, sd))).sum()

        # y_not_censored = self.df[self.y][self.df[self.censored] == False]

        # ll_not_censored = tf.math.reduce_sum(np.log(1/sd) + tfp.distributions.Normal(loc = 0, scale =1).log_prob((y_not_censored - y_not_censored*beta)/sd))
        
        # y_censored = self.df[self.y][self.df[self.censored] == True]
        # ll_censored = tf.math.reduce_sum(tfp.distributions.Normal(loc = 0, scale =1).log_cdf((y_censored*beta - y_censored)/sd))
        
        loglik = float(ll_censored + ll_not_censored)
        return - loglik

    def con(self,sd):
        return sd > 0

    def minimize(self, initial_guess,upper_threshold):
        return minimize(self.nll, initial_guess, method = "Nelder-Mead",args=(upper_threshold), tol=0.01)

    def Lambda(self, sd, prediction):
        return norm.pdf(prediction/sd)/norm.cdf(prediction/sd)

    def predict(self, X, Beta):
        #sd = np.exp(sd)
        #return norm.cdf(prediction / sd) * (prediction + sd * self.Lambda(sd, prediction))
        return np.dot(X,Beta)

if __name__ == "__main__":
    x1 = np.linspace(-5, 10, 200)
    x2 = np.linspace(-5, 10, 200)
    y = x1*2 + x2 + np.random.normal(0,1,200)
    y[y > 15] = 15

    df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
    df['censor'] = y == 15
    t = Tobit(df, 'censor',['x1','x2'], 'y')

    regressor = LinearRegression(fit_intercept=False).fit(df[['x1','x2']],df['y'])
    pred = regressor.predict(df[['x1','x2']])
    beta = regressor.coef_
    print(beta)
    sd_guess = 1.5
    # sds = np.linspace(0.001, 3, 1000)
    # vars = [[beta, i] for i in sds]
    
    
    # nlls = []
    # for var in vars:
    #     nlls.append(t.nll(var,15))
    
    # plt.plot(sds, nlls)
    # plt.show()
    
    vars = np.append(beta, sd_guess)
    #print(vars)
    mini = t.minimize(vars,15)
    #print(mini)
    #print(beta)
    sd = mini.x[-1]
    pred_pred = x1*mini.x[0] + x2*mini.x[1]
    print(mini)
    # pred_t = t.predict(np.exp(sd), pred_pred)

    plt.scatter(x1,y, label="Data", color="black",alpha=0.8)
    plt.plot(x1, t.predict(sd,pred_pred), label="Tobit", color="red", alpha=0.8)
    plt.plot(x1, pred, label="Linear", color="green",alpha=0.8)
    plt.legend()
    plt.show()

    
