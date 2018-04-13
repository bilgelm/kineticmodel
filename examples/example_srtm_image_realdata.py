import os
import numpy as np
import nibabel as nib
from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996
import cprofile

data_path = '/data1/Ganz/gitrepos/kineticmodelling/data/nru/' #'H:/gitrepos/kineticmodel/data/nru/'
inputTac_filename = os.path.join(data_path, 'input.mni305.2mm.sm6.nii.gz')
cerebMask_filename = os.path.join(data_path, 'cereb.mni305.2mm.nii.gz')
tim_filename = os.path.join(data_path, 'info_tim.csv')

inputTac = nib.load(inputTac_filename)
cerebMask = nib.load(cerebMask_filename)

#results_img = SRTM_Lammertsma1996.volume_wrapper(timeSeriesImgFile=inputTac_filename,frameTimingCsvFile=tim_filename,refRegionMaskFile=cerebMask_filename,time_unit='s',startActivity='flat')
cProfile.run('results_img = SRTM_Zhou2003.surface_wrapper(timeSeriesImgFile=inputTac2_filename,frameTimingCsvFile=tim_filename,refRegionMaskFile=cerebMask_filename,time_unit='s',startActivity='flat',fwhm=(2*np.sqrt(2*np.log(2))) * 5)')
BP_img = results_img['BP']

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(BP_img[:,:,20])
plt.imshow(BP_img[:,:,5])

inputTac2_filename = os.path.join(data_path, 'tac.surf.lh.nii.gz')
inputTac2 = nib.load(inputTac2_filename)
