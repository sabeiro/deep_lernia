"""
train_longShort:
implementation of the long short term memory to forecast time series
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM, RepeatVector, Dropout
from keras.layers import Embedding, GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import lernia.train_score as t_s
from deep_lernia.train_keras import trainKeras


class timeSeries(trainKeras):
    """keras on time series"""
    def __init__(self,X,y=[None]):
        """initiate parent and set Xy"""
        trainKeras.__init__(self,X)

    def longShort(self):
        """define an autoencoder model"""
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim)(inputs)
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)
        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        self.model = encoder

    def textModel(self,max_words=400,max_phrase_len=30):
        """define a model for text"""
        model = Sequential()
        model.add(Embedding(input_dim=max_words,output_dim=256,input_length=max_phrase_len))
        model.add(SpatialDropout1D(0.3))
        model.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(5, activation = 'softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        self.model = model


    def modelXgboost(self):
        """define xgboost model"""
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        except:
            self.newModel()
        
        
    def modelLongShort(self,input_shape=(1,1)):
        """define baseline model"""
        shape = self.X.shape
        input_shape = (batch_size, 1, shape[1])
        model = Sequential()
        model.add(LSTM(50,input_shape=input_shape))
	#model.add(LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]),stateful=True))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
	#model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def lstmBatch(self,batch_size=1, neurons=4, n_in=1, n_out=1):
        """ long short with batch_size"""
        shape = self.X.shape
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, 1, n_in*shape[1]), stateful=True))
        model.add(Dense(n_out*shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.batch_size = batch_size
        self.model = model

    def longShortDeep(self,batch_size=1, neurons=32, n_in=1, n_out=1):
        """ long short with batch_size"""
        shape = self.X.shape
        model = Sequential()
        model.add(LSTM(units=neurons,return_sequences=True,input_shape=(1,n_in*shape[1])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=neurons,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=neurons))
        model.add(Dropout(0.2))
        model.add(Dense(units=n_out*shape[1]))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        self.model = model
        
    def newModel(self):
        """define a new model"""
        print("new long short term memory model")
        self.modelLongShort()
        
    def defModel(self):
        """create or return an existing model"""
        try:
            self.model
        except:
            self.newModel()
    
    def toSupervised(self,X,n_in=1,n_out=1):
        """perform a n_in shift for each of training feature"""
        df = pd.DataFrame(X)
        pre = [df.shift(i) for i in range(1, n_in+1)]
        post = [df.shift(-i) for i in range(0, n_out)]
        post = pd.concat(post, axis=1)
        pre = pd.concat(pre, axis=1)
        if n_out == 1:
            pre = pre[n_in:]
            post = post[n_in:]
        else:
            n_end = n_out - 1
            pre = pre[n_in:-n_end]
            post = post[n_in:-n_end]
        # pre.fillna(0., inplace=True)
        # post.fillna(0., inplace=True)
        return pre.values, post.values

    def shiftColumns(self,X,n_in=1):
        """perform a n_in shift for each of training feature"""
        df = pd.DataFrame(X)
        pre = [df.shift(i) for i in range(0, n_in)]
        pre = pd.concat(pre, axis=1)
        pre.replace(float('nan'),0,inplace=True)
        return pre

    
    def splitSet(self,X1,portion=.75,n_in=1,n_out=1,shuffle=False):
        """overwrite split set preparing Xy"""
        N = X1.shape[0]
        if shuffle:
            shuffleL = np.arange(0,N)
            n = int(np.random.random(1)*N)
            shuffleL = np.roll(shuffleL,n)
        else:
            shuffleL = list(range(N))
        X = np.array(X1)[shuffleL]
        ahead = int(X.shape[0]*portion)
        pre, post = self.toSupervised(X,n_in=n_in,n_out=n_out)
        X_train, X_test = pre[0:ahead], pre[ahead:]
        y_train, y_test = post[0:ahead,:], post[ahead:,:]
        X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
        y_train = y_train.reshape(len(y_train),-1)
        if portion < 1.:
            X_test = X_test.reshape(X_test.shape[0], -1, X_test.shape[1])
            y_test = y_test.reshape(len(y_test),-1)
        if len(X_train) == 0:
            raise Exception('Empty training set')
        return X_train, y_train, X_test, y_test

    def train(self,batch_size=1,nb_epoch=300,shuffle=False,calc_metrics=False,portion=0.75,n_in=1,n_out=1,**args):
        """train a baseline model resetting the states every nb_epochs"""
        X_train, y_train, X_test, y_test = self.splitSet(self.X,shuffle=shuffle,portion=portion,n_in=n_in,n_out=n_out)
        self.defModel()
        for i in range(nb_epoch):
            history = self.model.fit(X_train,y_train
                                     ,epochs=1,batch_size=batch_size
                                     ,validation_data=(X_test,y_test)
                                     ,verbose=2,shuffle=False)
            self.model.reset_states()
            self.history.append(history)
        if not calc_metrics:
            return {}
        self.model.predict(X_train, batch_size=batch_size)
        y_pred = self.predict(X_test,batch_size=batch_size)
        kpi = t_s.calcMetrics(y_pred[:,0], y_test[:,0])
        return kpi

    def evaluate(self,shuffle=True,mode="forecast",portion=0.8,n_in=1,n_out=1):
        """evaluate performances"""
        X_train, y_train, X_test, y_test = self.splitSet(self.X,shuffle=shuffle,portion=portion,n_in=n_in,n_out=n_out)
        if mode == "forecast":
            y_pred = self.forecast(X_test)
        else:
            y_pred = self.predict(X_test)
        kpi = t_s.calcMetrics(y_pred[:,0], y_test[:,0])
        return kpi
    
    def trainCross(self,X_test,batch_size=1,nb_epoch=300,calc_metrics=False,n_in=1,n_out=1):
        """train a baseline model resetting the states every nb_epochs"""
        X_train, y_train, X_train1, y_train1 = self.splitSet(self.X,portion=1.,n_in=n_in,n_out=n_out)
        X_test = np.asarray(self.scaleX(X_test))
        X_test, y_test, X_test1, y_test1 = self.splitSet(X_test,portion=1.)
        self.defModel()
        for i in range(nb_epoch):
            history = self.model.fit(X_train,y_train
                                     ,epochs=1,batch_size=batch_size
                                     ,validation_data=(X_test,y_test)
                                     ,verbose=2,shuffle=False)
            self.model.reset_states()
            self.history.append(history)
        if not calc_metrics:
            return {}
        self.model.predict(X_train, batch_size=batch_size)
        y_pred = self.predict(X_test,batch_size=batch_size)
        kpi = t_s.calcMetrics(y_pred[:,0], y_test[:,0])
        return kpi

    def trainBoost(self,shuffle=False,calc_metrics=False,portion=0.75,n_in=1,n_out=1,**args):
        """train a baseline model resetting the states every nb_epochs"""
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error
        N = self.X.shape[0]
        if shuffle:
            shuffleL = np.arange(0,N)
            n = int(np.random.random(1)*N)
            shuffleL = np.roll(shuffleL,n)
        else:
            shuffleL = list(range(N))
        X = np.array(self.X)[shuffleL]
        ahead = int(X.shape[0]*portion)
        pre, post = self.toSupervised(X,n_in=n_in,n_out=n_out)
        X_train, X_test = pre[0:ahead,:], pre[ahead:,:]
        y_train, y_test = post[0:ahead,-1], post[ahead:,-1]
        y_pred = list()
        history = [x for x in X_train]
        for i in range(len(X_test)):
            testX, testy = X_test[i, :], X_test[i, -1]
            train = np.asarray(history)
            self.model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
            self.model.fit(X_train, y_train)
            testX = testX.reshape( (1,testX.shape[0]) )
            yhat = self.model.predict(testX)[0]
            y_pred.append(yhat)
            history.append(testX)
        t = list(range(ahead,len(self.X)))
        # if not calc_metrics:
        #     return {}
        # error = mean_absolute_error(testX[:, 1], y_pred)
        # kpi = {"error":error}
        return t, y_pred


    def predict(self,X_test,batch_size=1,n_in=1):
        """prediction for the short long term memory"""
        y_pred = []
        if n_in > 1:
            X_test = self.shiftColumns(X_test,n_in=n_in)
        if len(X_test.shape) == 2:
            X_test = self.reshape(X_test)
        for i in range(len(X_test)):
            X1 = X_test[i, :]
            X1 = X1.reshape(1, 1, X1.shape[1])
            pred = self.model.predict(X1, batch_size=batch_size)
            pred = [x for x in pred[0, :]]
            y_pred.append(pred)
        y_pred = np.asarray(y_pred)
        return y_pred
    

    def forecast(self,X_test,batch_size=1,n_in=1):
        """forecast using longshort"""
        if len(X_test.shape) == 2:
            X_test = self.reshape(X_test)
        y_fore = np.zeros((X_test.shape[0],X_test.shape[2]))
        if n_in > 1:
            X_test = self.shiftColumns(X_test,n_in=n_in)
        for i in range(0,n_in):
            y_fore[i] = X_test[0][i]
        for i in range(len(X_test)-n_in):
            X1 = y_fore[i].reshape(1,-1)
            #X1 = y_fore[:i+n_in]
            if n_in > 1:
                X1 = self.shiftColumns(X1,n_in=n_in).values
            X1 = self.reshape(X1)
            pred = self.model.predict(X1, batch_size=batch_size)
            y_fore[i+n_in] = pred[-1]
        return y_fore

    def forecastLongShort(self,y,ahead=28,epoch=50,n_in=2,n_out=1):
        """forecast on ahead time points"""
        portion = 1. - ahead/len(y)
        self.setLimit(y)
        X_train, X_test, y_train, y_test = self.splitSet(self.X,y,portion=portion)
        self.defModel()
        self.history = self.model.fit(X_train,y_train,epochs=epoch,batch_size=72,validation_data=(X_test,y_test),verbose=0,shuffle=False)#,callbacks=[TensorBoard(log_dir='/tmp/lstm')])
        y_scaled = self.model.predict(X_test)
        y_pred = y_scaled[:,0]*(self.yMax+self.yMin) + self.yMin
        y_test1 = y_test*(self.yMax+self.yMin) + self.yMin
        if False:
            plt.plot(y[ahead:],label="test")
            plt.plot(y_pred,label="pred")
            plt.legend()
            plt.show()
        
        kpi = t_s.calcMetrics(y_pred, y_test1)
        return self.model, kpi

    def reshape(self,X,mode=None,batch_size=1):
        """reshape the input dataset"""
        X_test = np.asarray(X)
        X_test = X_test.reshape((X.shape[0], batch_size, X.shape[1]))
        return X_test
    
