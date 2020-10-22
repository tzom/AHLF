import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from dataset import get_dataset_inference
import matplotlib.pyplot as plt
import numpy as np

repo = "./example"
raw_file = "example.mgf"
mgf_file="%s/%s"%(repo,raw_file) 

ds = get_dataset_inference(mgf_file=mgf_file,batch_size=1)
two_vec,scan=next(iter(ds))
intensities,mz = np.squeeze(two_vec)[:,0],np.squeeze(two_vec)[:,1]

plt.figure(figsize=(5,4))
plt.locator_params(axis='y', nbins=4)
plt.locator_params(axis='x', nbins=6)

plt.subplot(2,1,1)
plt.title('dataset: %s; raw-file: %s; scan number: %s'%(repo,raw_file,int(scan)),fontsize=8)
_,stemlines,_=plt.stem(np.arange(1,len(intensities)+1),intensities/np.max(intensities),basefmt=' ',markerfmt=' ',linefmt='black',use_line_collection=True)
plt.setp(stemlines, 'linewidth', 0.3)
plt.ylabel('norm. intensity [a.u.]')

plt.subplot(2,1,2)
_,stemlines,_=plt.stem(np.arange(1,len(intensities)+1),mz/np.max(mz),basefmt=' ',markerfmt=' ',linefmt='black',use_line_collection=True)
plt.setp(stemlines, 'linewidth', 0.3)
plt.xlabel('feature')
plt.ylabel('mz-remainder [a.u.]')
plt.tight_layout()
plt.savefig('two_vec_spectrum.png',dpi=300)
plt.savefig('two_vec_spectrum.svg')
        
        
