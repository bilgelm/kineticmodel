from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996
from temporalimage import load as ti_load
import nibabel as nib
import os
from unittest import TestCase
from kineticmodel.datasets import pet4D_file, timing_file, refRegionMask_file

class TestKineticModelVoxelwise(TestCase):
    def setUp(self):
        # read in image from examples/data
        #pet4D_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                          os.pardir,'examples/data/input.mni305.2mm.sm6.nii.gz')
        #timing_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                           os.pardir,'examples/data/info_tim.csv')
        print('test_voxelwise pet4D_file:' + pet4D_file)
        print(os.path.isfile(pet4D_file))
        print('test_voxelwise timing_file:' + timing_file)
        print(os.path.isfile(timing_file))
        print(os.listdir(os.pardir))
        print(os.listdir(os.path.dirname(__file__)))
        print(os.listdir(os.path.join(os.path.dirname(__file__),'kineticmodel')))
        print(os.listdir(os.path.join(os.path.dirname(__file__),'kineticmodel','datasets')))

        # check if files exist

        self.ti = ti_load(pet4D_file, timing_file)
        #self.refRegionMaskFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                                      os.pardir,'examples/data/cereb.mni305.2mm.nii.gz')
        self.refRegionMaskFile = refRegionMask_file
        self.time_unit = 'min'
        self.startActivity = 'flat'
        self.weights = 'frameduration'
        self.fwhm = 4

    def test_srtm_zhou2003_voxelwise(self):
        results_img = SRTM_Zhou2003.volume_wrapper(ti=self.ti,
                                    refRegionMaskFile=self.refRegionMaskFile,
                                    time_unit=self.time_unit,
                                    startActivity=self.startActivity,
                                    weights=self.weights,
                                    fwhm=self.fwhm)

        '''
        # save BP, R1 images
        BP_img = nib.Nifti1Image(results_img['BP'], affine=self.ti.affine,
                                 header=self.ti.header, extra=self.ti.extra)
        BP_file = os.path.abspath('tests/test_outputs/BP.mni305.2mm.sm6.nii.gz')
        nib.save(BP_img, BP_file)

        R1_img = nib.Nifti1Image(results_img['R1'], affine=self.ti.affine,
                                 header=self.ti.header, extra=self.ti.extra)
        R1_file = os.path.abspath('tests/test_outputs/R1.mni305.2mm.sm6.nii.gz')
        nib.save(R1_img, R1_file)
        '''
