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
    
    def nll(self, vars):
        sd = 2
        beta = vars
        x_not_censored = np.array(self.df[self.X][self.df[self.censored] == False])
        y_not_censored = np.array(self.df[self.y][self.df[self.censored] == False])
        x_censored = np.array(self.df[self.X][self.df[self.censored] == True])
        y_censored = np.array(self.df[self.y][self.df[self.censored] == True])
        
        ll_censored = scipy.stats.norm.logcdf((np.dot(x_censored, beta) - y_censored)/sd).sum()
        ll_not_censored = (1/sd*scipy.stats.norm.logpdf((y_not_censored - np.dot(x_not_censored, beta))/sd) - math.log(max(np.finfo('float').resolution, sd))).sum()
        
        loglik = float(ll_censored + ll_not_censored)
        return - loglik

    def minimize(self, initial_guess):
        return minimize(self.nll, initial_guess, method = "BFGS", tol=0.01)

    def predict(self, X, beta):
        return np.dot(X,beta)

if __name__ == "__main__":
    x1 = np.linspace(-5, 10, 200)
    x2 = np.linspace(-5, 10, 200)
    y = x1*2 + x2 + np.random.normal(0,1,200)
    y[y > 15] = 15

    df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
    df['censor'] = y == 15
    t = Tobit(df, 'censor',['x1','x2'], 'y')

    regressor = LinearRegression(fit_intercept=False, positive=True).fit(df[['x1','x2']],df['y'])
    pred = regressor.predict(df[['x1','x2']])
    beta = regressor.coef_
    print(beta)
    sd_guess = 1.5
    
    vars = np.append(beta, sd_guess)

    mini = t.minimize(vars)

    sd = mini.x[-1]
    pred_pred = x1*mini.x[0] + x2*mini.x[1]
    print(mini)

    
    
    pred_t = t.predict(df[["x1","x2"]],mini.x[:-1])
    

    plt.scatter(x1,y, label="Data", color="black",alpha=0.8)
    plt.plot(x1, pred_t, label="Tobit", color="red", alpha=0.8)
    plt.plot(x1, pred, label="Linear", color="blue",alpha=0.8)
    plt.legend()
    plt.show()

    
