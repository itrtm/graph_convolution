# Graph convolution layer test
# requires Tensorflow 1.14.0, Keras 2.2.4
#
import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import Layer, Dense, Activation, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

class gconv_lstm(Layer):
    def __init__(self, dim1, dim2, units, Adj, batch_size,**kwargs):
        super(gconv_lstm, self).__init__(**kwargs)
        self.dim1 = dim1
        self.dim2 = dim2
        self.units = units
        self.batch_size = batch_size
        self.Adj = K.variable(Adj)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.units,self.dim2),initializer='uniform', trainable=True, name='weight')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='bias')
        self.trainable_weights = [self.W,self.b]
        super(gconv_lstm, self).build(input_shape)
    
    def compute_output_shape(self,input_shape):
        return (self.batch_size,self.dim1,self.dim2)

    def call(self, x):
        tensor1=[]
        for i in range(self.units):
            w0 = K.tf.multiply(self.Adj,self.W[i])
            w0 = K.transpose(w0)
            tensor1.append(K.dot(x,w0)+self.b[i])
        tensor1=K.stack(tensor1,axis=3)
        tensor1 = K.tf.reduce_mean(tensor1, axis=3)
        return tensor1    

#load input data
X_train=np.load('inp_test.npy')
y_train=np.load('out_test.npy')

# load neighbourhood matrix
matA0 = pd.read_csv('neighbourhood.csv', header=None)
matAdj = matA0.values

#build model
model_GL=Sequential()
model_GL.add(BatchNormalization())
model_GL.add(gconv_lstm(16,35,20,matAdj,10))  # timesteps,input sources,units,adjoint matrix, batch size
model_GL.add(Activation("relu"))
model_GL.add(LSTM(5, activation='relu', input_shape=(16, 35)))
model_GL.add(Dense(16))
model_GL.add(Activation("relu"))
model_GL.add(Dense(3))
model_GL.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# training:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
EPOCHS = 100
BS = 10
H = model_GL.fit(X_train[0:500], y_train[0:500], validation_data=(X_train[501:], y_train[501:]), batch_size=BS, epochs=EPOCHS, verbose=1)
print("Minimum Validation Loss:",min(H.history['val_loss']))
K.clear_session()

