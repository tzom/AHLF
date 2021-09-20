import tensorflow as tf 
import numpy as np
import sys,os

import argparse

parser = argparse.ArgumentParser(description='Finetune AHLF and store finetuned weights.')
parser.add_argument('model_weights', type=str, help='[loaded:] trained model weights')
parser.add_argument('finetuned_weights', type=str, help='[saved:] finetuned model weights')
parser.add_argument('mgf_files', type=str, help='directory with mgf-files with suffixes: [.phos.mgf/.other.mgf]')

args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE
#tf.compat.v1.disable_eager_execution()
from dataset import get_dataset
from network import network

N_TRAINABLE_DENSE_LAYERS = 3
BATCH_SIZE=64
LEARNING_RATE=1.0e-4
DROPOUT_RATE=0.5
EPOCHS=1
VIRTUAL_EPOCH_SIZE=100

train = True
saving = True

def binary_accuracy(y_true, y_pred):
    #y_pred = tf.math.sigmoid(y_pred)
    return tf.reduce_mean(tf.cast(tf.math.equal(y_true, tf.math.round(y_pred)),tf.float32))

callbacks = []

ch = 64
net = network([ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch],kernel_size=2,padding='same',dropout=DROPOUT_RATE) 

inp = tf.keras.layers.Input((3600,2))
sigm = net(inp)
model = tf.keras.Model(inputs=inp,outputs=sigm)
bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.load_weights(args.model_weights)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,clipnorm=1.0),loss=bce,metrics=['binary_accuracy','Recall','Precision'])

# Freeze the model
for l in model.layers[-1].layers:    
    l.trainable = False
    if type(l)==tf.keras.layers.Dropout:
        l.rate=DROPOUT_RATE
        print('RATE:',l.rate)

# Strategy - tune N terminal dense layers: 
i = 0
for l in model.layers[-1].layers[::-1]:
    if type(l)==tf.keras.layers.Dense:
        l.trainable = True
        i+=1
    if i>=N_TRAINABLE_DENSE_LAYERS:
        break

model.layers[-1].summary()

train_data = get_dataset(dataset=[args.mgf_files],maximum_steps=None,batch_size=BATCH_SIZE,mode='training').prefetch(buffer_size=AUTOTUNE)

if train:
    model.fit(train_data,steps_per_epoch=VIRTUAL_EPOCH_SIZE,epochs=EPOCHS,callbacks=callbacks)

if saving:
    model.save_weights(args.finetuned_weights)
