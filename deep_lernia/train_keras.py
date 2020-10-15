"""
train_keras:
parent class for keras learning  
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
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, Callback
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from keras import optimizers
import lernia.train_score as t_s
import lernia.train_viz as t_v
# session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
# tensorflow.set_random_seed(1)
# sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
# keras.backend.set_session(sess)

def getConf():
    """returns a standard configuration dict"""
    conf = {"optimizer":"adam","model_type":"convNet"
            ,"normalize":False
            ,"epochs":200,"dump_step":100
            ,"model_save":"model_dump","model_load":"model_dump","plot_history":"plot_history"
            ,"version":"0.5.0 - including train_gan"}
    return conf

class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    
    def on_batch_end(self,batch,logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print('\n{0}/{1} - Batch Loss: {2}'.format(self.seen,self.params['nb_sample'],
                                                self.params['metrics'][0]) )
            
class trainKeras:
    """train_keras basics utilities"""
    def __init__(self,X,y=[None],conf=None):
        """save data set matrix"""
        if conf == None:
            conf = getConf()
        self.conf = conf
        self.setX(X)
        if any(y): self.setY(y)
        self.history = []

    def setBoundary(self,X):
        """Save the normalization boundaries"""
        self.minX, self.maxX = 0.,1.
        self.minY, self.maxY = 0.,1.
        if isinstance(X,pd.DataFrame):
            self.minX = [np.min(X[x]) for x in X.columns]
            self.maxX = [np.max(X[x]) for x in X.columns]
        else:
            self.minX = [np.min(X[:,x]) for x in range(X.shape[1])]
            self.maxX = [np.max(X[:,x]) for x in range(X.shape[1])]

    def setLimit(self,y):
        """Save the normalization boundaries"""
        self.yMin = np.min(y)
        self.yMax = np.max(y)

    def setOptimizer(self,name=None,**args):
        """set optimizer for the model"""
        if name == None:
            name = self.conf['optimizer']
        if name == "sgd":
            opt = optimizers.SGD(**args)
        elif name == "rmsdrop":
            opt = optimizers.RMSprop(**args)
        elif name == "adagrad":
            opt = optimizers.Adagrad(**args)
        elif name == "adadelta":
            opt = optimizers.Adadelta(**args)
        elif name == "adam":
            opt = optimizers.Adam(**args)
        elif name == "adamax":
            opt = optimizers.Adamax(**args)
        else: 
            opt = optimizers.Adadelta(**args)
        self.opt = opt

    def setModel(self,model):
        """set an external created model"""
        self.model = model
        
    def setX(self,X):
        """reset data set"""
        if len(X.shape) == 1:
            X = np.array(X).reshape(-1,1)
        if isinstance(X,pd.DataFrame):
            self.feat = X.columns
        else:
            self.feat = list(range(X.shape[1]))
        self.setBoundary(X)
        self.X = X

    def getX(self):
        """return data set"""
        return self.X

    def setY(self,y):
        """reset data set"""
        if len(y.shape) == 1:
            if isinstance(y,pd.Series):
                y = y.values.reshape(-1,1)
            else:
                y = y.reshape(-1,1)
        self.y = y

    def getY(self):
        """return data set"""
        return self.y

    def getEpochs(self):
        """return the total number of epochs"""
        history = self.history
        loss = np.concatenate([x for x in history.history['loss']])
        return len(loss)

    def printModel(self):
        """print current model"""
        print(self.model.summary())

    def reshape(self,X,mode=None):
        """reshape the input dataset"""
        return X
    
    def plotHistory(self,ax=None):
        """plot history of training performace"""
        if not self.history:
            raise Exception("train the model first")
        if ax == None:
            fig, ax = plt.subplots(1,1)
        history = self.history
        loss = np.concatenate([x.history['loss'] for x in history])
        val_loss = np.concatenate([x.history['val_loss'] for x in history])
        ax.plot(loss,label='train')
        ax.plot(val_loss,label='test')
        ax.set_xlabel("number of epochs")
        ax.set_ylabel("realtive error")
        ax.legend()

    def plotPrediction(self,X=[None],y=[None],t=[None],ax=None,color=None,reshape=None):
        """plot history of training performace"""
        if not self.model:
            raise Exception("train the model first")
        if ax == None:
            fig, ax = plt.subplots(1,1)
        if color == None:
            color = np.random.choice(t_v.colorL)
        if not any(X):
            X = self.X
        if not any(y):
            y = self.y
        if not any(t):
            t = list(range(len(y)))
        X = self.reshape(np.asarray(X),mode=reshape)
        y_pred = self.predict(X)
        ax.plot(t,y[:,0],label='real',linestyle="solid",color=color,linewidth=2,alpha=0.5)
        ax.plot(t,y_pred[:,0],label='predicted',linestyle="dashed",color=color,linewidth=1,alpha=1.)
        ax.legend()


    def saveModel(self,fName):
        """save last trained model"""
        if not self.model:
            raise Exception("train the model first")
        fName = fName.split(".")[0]
        model_json = json.loads(self.model.to_json())
        json.dump(model_json,open(fName+".json","w"))
        # with open(fName+".json","w") as json_file:
        #     json_file.write(self.model.to_json())
        self.model.save_weights(fName+".h5")

    def loadModel(self,fName):
        """load trained model"""
        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import model_from_json
        fName = fName.split(".")[0]
        json_file = open(fName+'.json','r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(fName+".h5")
        json_file.close()
        self.model = model

    def getModel(self):
        """return the encoder"""
        if not self.model:
            raise Exception("train the model first")
        return self.model

    def delModel(self):
        """delete current model"""
        try:
            del self.model
        except:
            pass
        
    def scale(self,feature_range=(0,1)):
        """scaler function"""
        X = self.X.astype('float32')
        if not hasattr(self,"scaler"):
            self.scaler = MinMaxScaler(feature_range=feature_range).fit(X)
        X = self.scaler.transform(X)
        self.X = pd.DataFrame(X,columns=self.feat)
        if hasattr(self,"y"):
            self.y = self.y.astype('float32')
            if not hasattr(self,"scaler_y"):
                self.scaler_y = MinMaxScaler(feature_range=feature_range).fit(self.y)
            self.y = self.scaler_y.transform(self.y)

    def scaleX(self,X=[],feature_range=(0,1)):
        """scaler function"""
        if len(X) == 0:
            X = self.X
        X = X.astype('float32')
        if not hasattr(self,"scaler"):
            self.scaler = MinMaxScaler(feature_range=feature_range).fit(X)
        X = self.scaler.transform(X)
        return pd.DataFrame(X,columns=self.feat)

    def descaleX(self,X=[],feature_range=(0,1)):
        """scaler function"""
        if len(X) == 0:
            X = self.X
        X = X.astype('float32')
        if not hasattr(self,"scaler"):
            print('no scaler defined')
            return X
        X = self.scaler.inverse_transform(X)
        return pd.DataFrame(X,columns=self.feat)

    def scaleY(self,y=[None],feature_range=(0,1)):
        """scale the y"""
        if any(y):
            if len(y.shape) == 1:
                if isinstance(y,pd.Series): y = y.values.reshape(-1,1)
                else: y = y.reshape(-1,1)
        else:
            y = self.y
        if not hasattr(self,"scaler_y"):
            self.scaler_y = MinMaxScaler(feature_range=feature_range).fit(y)
        y = self.scaler_y.transform(y)
        return y

    def delScale(self):
        """remove scaler from class"""
        if not hasattr(self,"scaler"):
            del(self.scaler)
        if not hasattr(self,"scaler_y"):
            del(self.scaler_y)
            
    def invert_scale(self, value):
        """invert scaled values"""
        new_row = [x for x in self.X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        return inverted[0, -1]
    
    def defConv2D(self):
        """define a convolutional neural network for small images"""
        k_size = (3,2) #(5,5)
        convS = [8,16] #[32,64]
        input_shape = self.X.shape
        model = Sequential()
        model.add(Conv2D(convS[0],kernel_size=k_size,strides=(1,1),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(convS[1],k_size,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(num_class,activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        return model

    def newModel(self):
        """pointer to the model"""
        self.model = self.defConv2D()
        
    def defModel(self):
        """return a new model if not previously defined"""
        if hasattr(self,"model"):
            return self.model
        else:
            self.model = self.newModel()
            return self.model

    def splitSet(self,X1,y,shuffle=True,portion=.75):
        """split data set"""
        X = np.array(X1)
        N = X.shape[0]
        N_tr = int(N*portion)
        if shuffle:
            shuffleL = random.sample(range(N),N)
        else:
            #shuffleL = np.arange(0,N)
            n = int(np.random.random(1)*N)
            shuffleL = np.roll(shuffleL,n)
        X_train = X[shuffleL][:N_tr]
        X_test = X[shuffleL][N_tr:]
        y_train = y[shuffleL][:N_tr]
        y_test = y[shuffleL][N_tr:]
        return X_train, X_test, y_train, y_test

    def evaluate(self,shuffle=True,mode="predict",n_in=1,n_out=1):
        """evaluate performances"""
        X_train, X_test, y_train, y_test = self.splitSet(self.X,self.y,shuffle=shuffle,n_in=n_in,n_out=n_out)
        y_pred = self.predict(X_test)
        kpi = t_s.calcMetrics(y_pred[:,0], y_test[:,0])
        return kpi
        
    def predict(self,X_test):
        """predict a trained model"""
        return self.model.predict(X_test)
    
    def train(self,y=[None],shuffle=True,calc_metrics=False,**args):
        """train the current model, calculate scores"""
        if any(y): self.setY(y)
        X_train, X_test, y_train, y_test = self.splitSet(self.X,self.y,shuffle=shuffle)
        self.defModel()
        out_batch = NBatchLogger(display=10)
        history = self.model.fit(X_train,y_train
                                 ,validation_data=(X_test,y_test)
                                 ,**args
                                 ,verbose=2
                                 #,callbacks=[out_batch]
                                 #,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        )
        if not calc_metrics:
            return {}
        score = self.model.evaluate(X_test, y_test, batch_size=16, verbose=0)
        print(score)
        self.history.append(history)
        y_pred = self.predict(X_test)
        kpi = t_s.calcMetrics(y_pred[:,0], y_test[:,0])
        return kpi
    
    def featKnockOut(self,X=[None],shuffle=True,portion=0.8,mode="predict"):
        """feature knock out, recursively remove one feature at time and calculate performances"""
        if any(X): self.setX(X)
        X = self.X
        tL = X.columns
        x_val = 16
        perfL = []
        print('all')
        for j in range(x_val):
            kpi = self.evaluate(shuffle=shuffle,mode=mode)
            kpi['feature'] = "all"
            perfL.append(kpi)
        for i in tL:
            print(i)
            X1 = X.copy()
            X1.loc[:,i] = np.random.random(X1.shape[0])
            self.setX(X1)
            for j in range(x_val):
                kpi = self.evaluate(shuffle=shuffle,mode=mode)
                kpi['feature'] = "- " + i
                perfL.append(kpi)
        perfL = pd.DataFrame(perfL)
        self.perfL = perfL
        return perfL
    
    def plotFeatImportance(self,perfL,ax=[None]):
        """boxplot of scores per feature"""
        if ax[0] == None:
            fig, ax = plt.subplots(1,2)
        perfL.boxplot(by="feature",column="rel_err",ax=ax[0])
        perfL.boxplot(by="feature",column="cor",ax=ax[1])
        for a in ax:
            for tick in a.get_xticklabels():
                tick.set_rotation(15)
        return ax

