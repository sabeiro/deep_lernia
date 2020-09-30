import os, sys, gzip, random, json, datetime, re, io
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import importlib
import lernia.train_reshape as t_r
import deep_lernia as d_l
import deep_lernia.train_deep as t_d
import deep_lernia.train_keras as t_k

#--------------------load---------------------------

tita = pd.read_csv(r'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
bank = pd.read_csv(baseDir + "entr/raw/bank/bank.csv.gz",compression="gzip")

#---------------------transform------------------------

importlib.reload(t_r)
X = bank.drop(columns={"y"})
X = t_r.factorize(X).values
Y = t_r.categorize(bank['y'].values)

#---------------------train---------------------

importlib.reload(t_k)
importlib.reload(t_d)
tp = t_d.predictor(X)
model = tp.logistic()
mod, kpi = tp.train(Y,epochs=100)
tp.plotHistory()
plt.show()
