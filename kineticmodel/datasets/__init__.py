import os

pet4D_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'input.mni305.2mm.sm6.nii.gz')
refRegionMask_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'cereb.mni305.2mm.nii.gz')
timing_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'info_tim.csv')

print('datasets init defined pet4D_file:' + pet4D_file)
