"""
train_deep:
implementation of keras library for deep learning predictors and regressors
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM, RepeatVector, Dropout, Activation
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from keras import optimizers
import lernia.train_score as t_s
from deep_lernia.train_keras import trainKeras
import deep_lernia.train_keras as t_k

def getConf():
    """returns a standard configuration dict"""
    return t_k.getConf()

class regressor(trainKeras):
    """keras regressor"""
    def __init__(self,X,y=[None],conf=None):
        trainKeras.__init__(self,X,y,conf)
        self.n_feat = X.shape[1]

    def setRegressor(self,name="baseline",**args):
        """set a regressor"""
        if name == "two_layer":
            self.twoLayerMod(**args)
        elif name == "three_layer":
            self.twoLayerMod(**args)
        else:
            self.baselineMod(**args)
        
    def baselineMod(self,layer=[10]):
        """define a baseline model"""
        n_feat = self.X.shape[1]
        model = Sequential()
        model.add(Dense(layer[0],input_shape=(n_feat,),kernel_initializer='normal',activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model
        
    def twoLayerMod(self,layer=[10,5]):
        """define a two layer regressor"""
        n_feat = self.X.shape[1]
        model = Sequential()
        model.add(Dense(layer[0], input_shape=(n_feat,), activation='relu'))
        model.add(Dense(layer[1], activation='relu'))
        model.add(Dense(1))
        model.compile(loss='binary_crossentropy', optimizer='adam')#, metrics=['accuracy','AUC'])
        self.model = model

    def threeLayerMod(self,layer=[10,5,3]):
        """define a two layer regressor"""
        n_feat = self.X.shape[1]
        model = Sequential()
        model.add(Dense(layer[0], input_shape=(n_feat,), activation='relu'))
        model.add(Dense(layer[1], activation='relu'))
        model.add(Dense(layer[2], activation='relu'))
        model.add(Dense(1))
        model.compile(loss='binary_crossentropy', optimizer='adam')#, metrics=['accuracy','AUC'])
        self.model = model

        
    def baseline(self,y=[None],n_fold=5):
        """run a baseline model"""
        if any(y):
            self.y = y
        X_train, X_test, y_train, y_test = self.splitSet(self.X,self.y)
        seed = 7
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        self.baselineMod()
        estimators.append(('mlp',KerasRegressor(build_fn=self.model,epochs=50,batch_size=5,verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=n_fold,random_state=seed)
        results = cross_val_score(pipeline,self.X,self.y,cv=kfold)
        clf = estimators[1][1]
        clf.fit(X,y)
        res = clf.predict(X_test)
        clf.score(X,y)
        print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
        self.clf = clf
        return clf
        
class predictor(trainKeras):
    """keras regressor"""
    def __init__(self,X):
        """call trainKeras constructor"""
        trainKeras.__init__(self,X)

    def defConv2d(self,X):
        """define a two dimensional convolution neural network"""
        k_size = (3,2) #(5,5)
        convS = [8,16] #[32,64]
        model = Sequential()
        model.add(Conv2D(convS[0],kernel_size=k_size,strides=(1,1),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(0.50))
        model.add(Conv2D(convS[1],k_size,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.50))
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(num_class,activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        self.model = model
        return model

    def categorical(self,n_cat=2):
        """set and compile a logistic model"""
        model = Sequential()
        model.add(Dense(16, input_shape=(self.X.shape[1],)))
        model.add(Dropout(0.30))
        model.add(Activation('sigmoid'))
        model.add(Dense(n_cat))
        model.add(Activation('softmax'))#'sigmoid'
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        self.model = model
        return model
    

    def logistic(self):
        """set and compile a logistic model"""
        return self.categorical(n_cat=2)

    
    def catPictures(self,y,epoch=50):
        """classify pictures by category"""
        N = self.X.shape[0]
        batch_size = 128
        img_x, img_y = X.shape[2], X.shape[1]
        #img_x, img_y = 28, 28
        input_shape = (img_x, img_y, 1)
        num_class = len(set(y))
        N_tr = int(N*.75)
        shuffleL = random.sample(range(N),N)
        x_train = X[shuffleL][:N_tr]
        x_test = X[shuffleL][N_tr:]
        x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
        x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
        y, _ = pd.factorize(y)
        y = keras.utils.to_categorical(y, num_class)
        y_train = y[shuffleL][:N_tr]
        y_test = y[shuffleL][N_tr:]
        self.defConv2d()
        self.history = self.model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch,verbose=1,validation_data=(x_test, y_test))#,callbacks=[TensorBoard(log_dir='/tmp/categorical')])
        score = model.evaluate(x_test, y_test, verbose=0)
        self.model = model
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    
# def modKeras(x_train,y_train):
#     """simple deep learning training"""
#     Nneu = x_train.shape[1]
#     Nent = y_train.shape[0]
#     try:
#         Ncat = y_train.shape[1]
#     except:
#         Ncat = 1
#     model = Sequential()
#     model.add(Dense(input_dim=Nneu,output_dim=Nneu,activation='relu'))#,init="uniform"))
#     keras.layers.core.Dropout(rate=0.15)
#     model.add(Dense(input_dim=Nneu,output_dim=Nneu,activation='relu'))#,init="uniform"))
#     keras.layers.core.Dropout(rate=0.15)
#     model.add(Dense(input_dim=Nneu,output_dim=Ncat,activation='sigmoid'))#,init="uniform"))
#     #model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
#     sgd = keras.optimizers.SGD(lr=0.001,decay=1e-7,momentum=.9)
#     adam = keras.optimizers.adam(lr=0.01,decay=1e-5)
#     #model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
#     model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
#     model.get_params = model.get_config()
#     return model

    # def tuneKeras():
    #     """tune deep learning"""
    #     seed = 7
    #     numpy.random.seed(seed)
    #     model = KerasClassifier(build_fn=create_model, verbose=0)
    #     param_grid ={"batch_size":[10, 20, 40, 60, 80, 100],"epochs":[10, 50, 100]}
    #     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    #     grid_result = grid.fit(X, Y)
    #     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #     means = grid_result.cv_results_['mean_test_score']
    #     stds = grid_result.cv_results_['std_test_score']
    #     params = grid_result.cv_results_['params']
    #     for mean, stdev, param in zip(means, stds, params):
    #         print("%f (%f) with: %r" % (mean, stdev, param))

