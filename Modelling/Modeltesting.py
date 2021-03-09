import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import platform
from Modelling.Modelling import modelling
from DataPrep.ImportData import importer
from pathlib import Path
class tests:
    def __init__(self):
        self.df = importer().LagCreation()


    
    def load_model(self, model):
        pmodels = Path("Modelling", "Models", model)
        m = keras.models.load_model(pmodels.absolute())
        
        return m
        
    def predict(self, m):
        modellin = modelling()
        X_train,self.X_test, X_val,y_train,self.y_test, y_val = modellin.ttsplit(self.df)
        
        self.y_pred  = m.predict(self.X_test)


    def ActualvspredPlot(self):
        plt.scatter(self.y_test, self.y_pred)
        plt.show()

        

if __name__ == "__main__":
    t = tests()
    m = t.load_model("NNWithLags.keras")
    t.predict(m)
    t.ActualvspredPlot()