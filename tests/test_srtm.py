from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996
from .generate_test_data import generate_fakeTAC_SRTM
import unittest

class TestSRTM(unittest.TestCase):
    def setUp(self):
        self.BP = 0.2
        self.R1 = 1.0
        self.t, self.dt, self.TAC, self.refTAC = generate_fakeTAC_SRTM(self.BP, self.R1)

        self.startActivity = 'flat'
        # 'flat' performs slightly better than 'increasing' in this example
        # 'zero' is the worst, but values are not much different across these

    def test_srtm_zhou2003_fit(self):
        self.model = SRTM_Zhou2003(self.t, self.dt, self.TAC, self.refTAC, self.startActivity)

        print('\nFitting SRTM_Zhou2003 with %s start activity' % self.startActivity)
        self.model.fit()

        print('True BP = %.6f; estimated BP = %.6f' % (self.BP, self.model.BP))
        print('True R1 = %.6f; estimated R1 = %.6f' % (self.R1, self.model.R1))
        self.assertAlmostEqual(self.BP, self.model.BP, delta=1e-4)
        self.assertAlmostEqual(self.R1, self.model.R1, delta=1e-2)

    def test_srtm_lammertsma(self):
        self.model = SRTM_Lammertsma1996(self.t, self.dt, self.TAC, self.refTAC, self.startActivity)

        print('\nFitting SRTM_Lammertsma1996 with %s start activity' % self.startActivity)
        self.model.fit()

        print('True BP = %.6f; estimated BP = %.6f' % (self.BP, self.model.BP))
        print('True R1 = %.6f; estimated R1 = %.6f' % (self.R1, self.model.R1))
        self.assertAlmostEqual(self.BP, self.model.BP, delta=1e-3)
        self.assertAlmostEqual(self.R1, self.model.R1, delta=1e-1)
