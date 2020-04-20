import tensorflow as tf 
import numpy as np
import sys,os
#import simplejson
AUTOTUNE = tf.data.experimental.AUTOTUNE

from dataset import get_dataset
from network import network

train = True
saving = False

def binary_accuracy(y_true, y_pred):
    #y_pred = tf.math.sigmoid(y_pred)
    return tf.reduce_mean(tf.cast(tf.math.equal(y_true, tf.math.round(y_pred)),tf.float32))

callbacks = []

ch = 64
net = network([ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch],kernel_size=2,padding='same',dropout=.2) 

inp = tf.keras.layers.Input((3600,2))
sigm = net(inp)
model = tf.keras.Model(inputs=inp,outputs=sigm)
bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)

learning_rate=5.0e-6
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0),loss=bce,val_loss=bce,metrics=['binary_accuracy','Recall','Precision'])

batch_size=64

steps=1
maximum_steps=batch_size*steps

train_data = get_dataset(dataset=['./training'],maximum_steps=maximum_steps,batch_size=batch_size,mode='training').prefetch(buffer_size=AUTOTUNE)

if train:
    model.fit(train_data,steps_per_epoch=1,epochs=1,callbacks=callbacks)

if saving:
    model.save_weights('model_weights.hdf5')
