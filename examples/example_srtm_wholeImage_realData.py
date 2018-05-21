
# coding: utf-8

# In[1]:

import os
import numpy as np
import nibabel as nib
from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996

# chage this to your local gitrepo installation path
local_install_path = 'H:\gitrepos\kineticmodel'
data_path = local_install_path + '/data/nru/'
# loading real volumetric data 
inputTac_filename = os.path.join(data_path, 'input.mni305.2mm.sm6.nii.gz')
cerebMask_filename = os.path.join(data_path, 'cereb.mni305.2mm.nii.gz')
tim_filename = os.path.join(data_path, 'info_tim.csv')

inputTac = nib.load(inputTac_filename)
cerebMask = nib.load(cerebMask_filename)

# using Zhou's algorithm to run it
# Note: The srtm Lammertsmaa implementation is very slow in the volume and hence not shown here
results_img = SRTM_Zhou2003.volume_wrapper(timeSeriesImgFile=inputTac_filename,frameTimingCsvFile=tim_filename,refRegionMaskFile=cerebMask_filename,time_unit='s',startActivity='flat',fwhm=(2*np.sqrt(2*np.log(2))) * 5)


# In[2]:

results_img.keys()
BP_img = results_img['BP']

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.imshow(BP_img[:,:,20])


# In[3]:

plt.imshow(BP_img[:,:,5])


# In[ ]:



