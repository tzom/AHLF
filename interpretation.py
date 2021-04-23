import tensorflow as tf 
import numpy as np
import sys,os

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.compat.v1.disable_eager_execution()
from dataset import get_dataset_inference
from network import network

ch = 64
net = network([ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch],kernel_size=2,padding='same',dropout=.2) 
inp = tf.keras.layers.Input((3600,2))
sigm = net(inp)
model = tf.keras.Model(inputs=inp,outputs=sigm)
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))

#####################################################################
#####################################################################
#####################################################################

batch_size=64

model.load_weights("model/alpha_model_weights.hdf5")

from get_spectrum_as_numpy import get_spectrum
import shap 

spectrum = get_spectrum('example/example.mgf')
background_spectra = np.zeros((1,3600,2))
e = shap.DeepExplainer(model, background_spectra)

take_specific_spectrum=0
spectrum=spectrum[take_specific_spectrum,:,:]
spectrum=np.expand_dims(spectrum,0)

interpretation = np.squeeze(e.shap_values(spectrum))
spectrum = np.squeeze(spectrum)

import matplotlib.pyplot as plt

plt.title("Mirrorplot: demonstrating how SHAP values are used to interpret AHLF.")
plt.stem(spectrum[:,0]/np.max(spectrum[:,0]),linefmt='C0-',markerfmt=' ',basefmt=' ',use_line_collection=True,label='acquired peak')
plt.stem(- np.abs(interpretation[:,0])/np.max(np.abs(interpretation[:,0])),linefmt='C1-',markerfmt=' ',basefmt=' ',use_line_collection=True,label='abs. SHAP value')
plt.xlabel('feature')
plt.ylabel('Norm. abundance [a.u.]')
plt.legend()
plt.savefig('interpretation.png')

