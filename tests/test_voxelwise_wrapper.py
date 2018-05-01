from kineticmodel import SRTM_Zhou2003, SRTM_Gunn1997
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

    def test_srtm_gunn1997_voxelwise(self):
        results_img = SRTM_Gunn1997.volume_wrapper(ti=self.ti,
                                    refRegionMaskFile=self.refRegionMaskFile,
                                    time_unit=self.time_unit,
                                    startActivity=self.startActivity,
                                    weights=self.weights)

        '''
        # save BP, R1 images
        BP_img = nib.Nifti1Image(results_img['BP'], affine=self.ti.affine,
                                 header=self.ti.header, extra=self.ti.extra)
        BP_file = os.path.abspath('tests/test_outputs/BPgunn.mni305.2mm.sm6.nii.gz')
        nib.save(BP_img, BP_file)

        R1_img = nib.Nifti1Image(results_img['R1'], affine=self.ti.affine,
                                 header=self.ti.header, extra=self.ti.extra)
        R1_file = os.path.abspath('tests/test_outputs/R1gunn.mni305.2mm.sm6.nii.gz')
        nib.save(R1_img, R1_file)
        '''
