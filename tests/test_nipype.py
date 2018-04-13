# THIS NEEDS TO BE REVISITED AFTER IMPLEMENTING VOXELWISE WRAPPER TEST

try:
    import os
    import unittest

    from kineticmodel.nipype import KineticModel
    from kineticmodel.datasets import pet4D_file, timing_file, refRegionMask_file
    from nipype.pipeline.engine import Node

    class TestKineticModelNipype(unittest.TestCase):
        def setUp(self):
            # read in image from examples/data
            #self.pet4D_file = os.path.abspath('examples/data/input.mni305.2mm.sm6.nii.gz')
            #self.pet4D_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            #                               os.pardir,'examples/data/input.mni305.2mm.sm6.nii.gz')
            #self.timing_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            #                                os.pardir,'examples/data/info_tim.csv')
            #self.refRegionMaskFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            #                                      os.pardir,'examples/data/cereb.mni305.2mm.nii.gz')
            self.pet4D_file = pet4D_file
            self.timing_file = timing_file
            self.refRegionMaskFile = refRegionMask_file
            
            self.time_unit = 'min'
            self.startActivity = 'flat'
            self.weights = 'frameduration'
            self.fwhm = 4

        def test_nipype_srtm_zhou2003(self):
            km = Node(interface=KineticModel(model='SRTM_Zhou2003',
                                             timeSeriesImgFile=self.pet4D_file,
                                             frameTimingCsvFile=self.timing_file,
                                             refRegionMaskFile=self.refRegionMaskFile,
                                             time_unit=self.time_unit,
                                             startActivity=self.startActivity,
                                             weights=self.weights,
                                             fwhm=self.fwhm), name="km")
            km.run()

except ImportError:
    print('Cannot perform kineticmodel.nipype tests. \
           To carry out these tests, install kineticmodel using nipype option.')
