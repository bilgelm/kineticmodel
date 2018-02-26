import os
import numpy as np
import nibabel as nib
from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996

data_path = 'H:/gitrepos/kineticmodel/data/nru/'
inputTac_filename = os.path.join(data_path, 'input.mni305.2mm.sm6.nii.gz')
cerebMask_filename = os.path.join(data_path, 'cereb.mni305.2mm.nii.gz')
tim_filename = os.path.join(data_path, 'info_tim.csv')

inputTac = nib.load(inputTac_filename)
cerebMask = nib.load(cerebTac_filename)

data = inputTac.get_data()
data.shape
type(data)

data2 = cerebMask.get_data()
data2.shape
type(data2)

#mdl_lammertsma = SRTM_Lammertsma1996(t, dt, TAC_matrix, refTAC, time_unit='min')

results_img = SRTM_Lammertsma1996.volume_wrapper(timeSeriesImgFile=inputTac_filename,frameTimingCsvFile=tim_filename,refRegionMaskFile=cerebTac_filename,time_unit='s',startActivity='flat')
results_img.shape
