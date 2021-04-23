import tensorflow as tf 
import numpy as np
import sys,os
import argparse

parser = argparse.ArgumentParser(description='Read spectra from an mgf-file and output a prediction score.')
parser.add_argument('model_weights', type=str, help='trained model weights')
parser.add_argument('mgf_file', type=str, help='input filename: mgf-file containing ms/ms spectra.')
parser.add_argument('out_file', type=str, help='output filename')
parser.add_argument('--tsv', dest='tsv', action='store_const',
                    const=True, default=False, help='write a tsv-file')

args = parser.parse_args()

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

model.load_weights(args.model_weights)

mgf_file = args.mgf_file


ds = get_dataset_inference(mgf_file=mgf_file,batch_size=batch_size)#.prefetch(buffer_size=AUTOTUNE)
print('predicting: ',mgf_file)

some_big_number=10**5
pred_score = model.predict(ds,steps=some_big_number)#dirty-hack to repair: keras crying about unspecified steps.

score = np.atleast_1d(np.squeeze(pred_score))
pred = np.round(score)

print('Writing results to file: %s'%args.out_file)

if not args.tsv:
    np.savetxt(args.out_file,score)
###

if args.tsv:
    from pyteomics import mgf
    import pandas as pd
    def get_scans(entry):
        title = str(entry['params']['title'])
        scans = int(entry['params']['scans'])
        return title,scans      
    title_,scan_ = zip(*list(map(get_scans,mgf.read(mgf_file))))
    
    ###
    df = pd.DataFrame({'title':title_,'scan':scan_,'score':score,'pred':pred})
    df.to_csv(args.out_file,sep='\t')

print('Done.')