from scipy.stats import norm
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score

class Tobit:
    def __init__(self, df, censored, x, y):
        self.df = df
        self.censored = censored
        self.X = x
        self.y = y
    
    def linear_regression(self, censoring):
        regressor = LinearRegression()
        regressor.fit(np.array(self.df[self.X]).reshape(-1,1), self.df[self.y])
        pred = regressor.predict(np.array(self.df[self.X][self.df[self.censored] == censoring]).reshape(-1,1))
        return pred

    def nll(self, sd, upper_threshold):
        y_not_censored = self.df[self.y][self.df[self.censored] == False]
        y_pred_not_censored = self.linear_regression(False)

        ll_not_censored = tf.math.reduce_sum(np.log(1/sd * norm.pdf((y_not_censored - y_pred_not_censored)/sd)))

        y_censored = self.df[self.y][self.df[self.censored] == True]
        #y_pred_censored = self.linear_regression(True)
        ll_censored = tf.math.reduce_sum(np.log(norm.cdf((y_censored - upper_threshold)/sd)))

        return float(- ll_not_censored - ll_censored)

    def minimize(self, initial_guess,upper_threshold):
        return minimize(self.nll, initial_guess, method = "BFGS", args=(upper_threshold))

    def Lambda(self, sd, prediction):
        return norm.pdf(prediction/sd)/norm.cdf(prediction/sd)

    def predict(self, sd, prediction):
        return norm.cdf(prediction / sd) * (prediction + sd * self.Lambda(sd, prediction))


if __name__ == "__main__":
    pass