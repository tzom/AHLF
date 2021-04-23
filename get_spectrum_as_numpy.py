import tensorflow as tf
import numpy as np
import sys,os,glob
ahlf_dir = "./AHLF"
sys.path.append(ahlf_dir)
from dataset import get_dataset_inference

def tf_dataset_to_numpy(dataset):
    if tf.executing_eagerly():
        data = []
        for i,x in enumerate(dataset):
            data.append(x)
            #if i > n:
            #    break
    else:
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            data=[]
            while True:
                try:
                    data.append(sess.run(next_element))
                except:
                    break
    x,y=zip(*data) 
    x,y=np.array(x),np.array(y)
    x,y=np.squeeze(x),np.squeeze(y)
    return x,y

def _f(path):
    return get_dataset_inference(mgf_file=path,batch_size=1)

def get_spectrum(path):
    spectrum, _ = tf_dataset_to_numpy(_f(path))
    return spectrum