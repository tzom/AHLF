import tensorflow as tf 
import numpy as np

import sys,os
#sys.path.append('../TCN/')
from TCN import TCN


class network(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, padding, dropout):
        super(network, self).__init__()
        #init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)#, dtype=tf.float32)
        init = tf.keras.initializers.he_normal()
        glorot_init = tf.keras.initializers.glorot_normal()
        #init = None
        contstraint = None#tf.keras.constraints.MaxNorm(1.0)

        self.sig = tf.keras.layers.Conv1D(filters=128,kernel_size=1,strides=1,kernel_initializer=init,padding=padding,activation=None)
        self.ac0 = tf.keras.layers.Activation('relu')
        
        tcn_dropout  =.0
        dense_dropout= dropout
        
        self.tcn = TCN(num_channels, kernel_size=kernel_size, padding=padding, dropout_rate=tcn_dropout)
        self.pooling = tf.keras.layers.MaxPooling1D(20) 
        self.flatten = tf.keras.layers.Flatten()
        self.drop0 = tf.keras.layers.Dropout(dense_dropout)
        self.dense1 = tf.keras.layers.Dense(512, kernel_initializer=init,kernel_constraint=contstraint,activation=None)
        #self.bn1 = tf.keras.layers.BatchNormalization(scale=False)
        self.drop1 = tf.keras.layers.Dropout(dense_dropout)
        self.ac1 = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(512, kernel_initializer=init,kernel_constraint=contstraint,activation=None)
        #self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
        self.drop2 = tf.keras.layers.Dropout(dense_dropout)
        self.ac2= tf.keras.layers.Activation('relu')
        self.dense3 = tf.keras.layers.Dense(512, kernel_initializer=init,kernel_constraint=contstraint,activation=None)
        #self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
        #self.drop3 = tf.keras.layers.Dropout(dense_dropout)
        self.ac3= tf.keras.layers.Activation('relu')
        self.dense = tf.keras.layers.Dense(1, kernel_initializer=glorot_init,activation='sigmoid')

    def call(self, x, training=tf.keras.backend.learning_phase()):
        y = self.sig(x)
        y = self.ac0(y)

        y = self.tcn(y, training=training)
        #yf = y[:,-1,:]
        y = self.pooling(y)
        y = self.flatten(y)
        #y = tf.keras.layers.Concatenate()([y,yf])
        y = self.drop0(y)# if training else y

        y = self.dense1(y)
        #y = self.bn1(y)
        y = self.ac1(y)
        y = self.drop1(y)# if training else y 

        y = self.dense2(y)  
        #y = self.bn2(y) 
        y = self.ac2(y)
        y = self.drop2(y)# if training else y  

        y = self.dense3(y)
        #y = self.bn3(y)   
        y = self.ac3(y)
        #y = self.drop3(y) if training else y   

        y = self.dense(y)
        return y