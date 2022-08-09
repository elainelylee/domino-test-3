import pickle
import datetime
import pandas as pd
import numpy as np

with open('model.pkl', 'rb') as f:
    m = pickle.load(f)

def predict(temperature):
    r = m.predict(np.array([temperature]).reshape(-1,1))
    return r[0][0]