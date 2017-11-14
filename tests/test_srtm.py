from kineticmodel import SRTM_Zhou2003
from .generate_test_data import generate_fakeTAC_SRTM
import unittest
import numpy as np

class TestSRTM(unittest.TestCase):
    def setUp(self):
        self.BP = 0.2
        self.R1 = 1.0
        t, dt, TAC, refTAC = generate_fakeTAC_SRTM(self.BP, self.R1)

        startActivity = 'flat'
        # 'flat' performs slightly better than 'increasing' in this example
        # 'zero' is the worst, but values are not much different across these

        self.model = SRTM_Zhou2003(t, dt, TAC, refTAC, startActivity)

    def test_srtm_zhou2003_fit(self):
        self.model.fit()

        self.assertAlmostEqual(self.BP, self.model.BP, delta=1e-4)
        self.assertAlmostEqual(self.R1, self.model.R1, delta=1e-2)

class TestSRTM_ROI_matrix(unittest.TestCase):
    def setUp(self):
        numROIs = 10

        BP = 0.2
        R1 = 1.0
        t, dt, TAC, refTAC = generate_fakeTAC_SRTM(BP, R1)

        startActivity = 'flat'

        TAC_matrix = np.tile(TAC,(numROIs,1))
        self.BP = np.repeat(BP, numROIs)
        self.R1 = np.repeat(R1, numROIs)

        self.model = SRTM_Zhou2003(t, dt, TAC_matrix, refTAC, startActivity)

    def test_srtm_zhou2003_fit_many(self):
        self.model.fit(smoothTAC=self.model.TAC)

        print('\nBP estimates:')
        print(self.model.BP)
        print('\nR1 estimates:')
        print(self.model.R1)

        self.assertAlmostEqual(np.amax(np.absolute(self.BP - self.model.BP)), 0, delta=1e-4)
        self.assertAlmostEqual(np.amax(np.absolute(self.R1 - self.model.R1)), 0, delta=1e-2)
