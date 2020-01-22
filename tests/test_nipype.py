try:
    import os
    from shutil import rmtree
    from uuid import uuid4
    import unittest

    from kineticmodel.nipype_wrapper import KineticModel
    from kineticmodel.datasets import pet4D_file, timing_file, refRegionMask_file
    from nipype.pipeline.engine import Node, Workflow
    from nipype.interfaces.utility import IdentityInterface

    class TestKineticModelNipype(unittest.TestCase):
        def setUp(self):
            self.pet4D_file = pet4D_file
            self.timing_file = timing_file
            self.refRegionMaskFile = refRegionMask_file

            self.startActivity = 'flat'
            self.weights = 'frameduration'
            self.fwhm = 4

            # make a temporary directory in which to save the temporary nipype image files
            self.tmpdirname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           'tests_output_'+uuid4().hex)

            if not os.path.isdir(self.tmpdirname):
                os.makedirs(self.tmpdirname)

        def tearDown(self):
            # remove the tests_output directory
            rmtree(self.tmpdirname)

        def test_nipype_srtm_zhou2003(self):
            infosource = Node(IdentityInterface(fields=['in_file']),
                              name="infosource")
            infosource.iterables = ('in_file', [self.pet4D_file])

            km = Node(KineticModel(model='SRTM_Zhou2003',
                                   #timeSeriesImgFile=self.pet4D_file,
                                   frameTimingFile=self.timing_file,
                                   refRegionMaskFile=self.refRegionMaskFile,
                                   startActivity=self.startActivity,
                                   weights=self.weights,
                                   fwhm=self.fwhm), name="km")

            km_workflow = Workflow(name="km_workflow",
                                   base_dir=self.tmpdirname)
            km_workflow.connect([
                                 (infosource, km, [('in_file', 'timeSeriesImgFile')])
                                ])

            km_workflow.run()

except ImportError:
    print('Cannot perform kineticmodel nipype tests. \
           To carry out these tests, install kineticmodel using nipype option.')
