from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996, SRTM_Gunn1997
from .generate_test_data import generate_fakeTAC_SRTM
import unittest
from ddt import ddt, data, unpack

@ddt
class TestSRTM(unittest.TestCase):
    iters = [(BP, R1, startActivity) for BP in [0.2, 0.5, 0.8] \
                                     for R1 in [1.0, 0.7, 1.3] \
                                     for startActivity in ['flat','increasing']]

    @unpack
    @data(*iters)
    def test_srtm_zhou2003_fit(self, BP, R1, startActivity):
        self.t, self.dt, self.TAC, self.refTAC = generate_fakeTAC_SRTM(BP, R1)
        self.model = SRTM_Zhou2003(self.t, self.dt, self.TAC, self.refTAC,
                                   time_unit='min', startActivity=startActivity)

        print('\nFitting SRTM_Zhou2003 with %s start activity' % startActivity)
        self.model.fit()

        print('True BP = %.6f; estimated BP = %.6f; percent error = %.1E' % (BP, self.model.results['BP'], 100*abs(BP-self.model.results['BP'])/BP))
        print('True R1 = %.6f; estimated R1 = %.6f; percent error = %.1E' % (R1, self.model.results['R1'], 100*abs(R1-self.model.results['R1'])/R1))
        self.assertAlmostEqual(BP, self.model.results['BP'], delta=5e-3)
        self.assertAlmostEqual(R1, self.model.results['R1'], delta=5e-2)

    @unpack
    @data(*iters)
    def test_srtm_lammertsma(self, BP, R1, startActivity):
        self.t, self.dt, self.TAC, self.refTAC = generate_fakeTAC_SRTM(BP, R1)
        self.model = SRTM_Lammertsma1996(self.t, self.dt, self.TAC, self.refTAC,
                                         time_unit='min', startActivity=startActivity)

        print('\nFitting SRTM_Lammertsma1996 with %s start activity' % startActivity)
        self.model.fit()

        print('True BP = %.6f; estimated BP = %.6f; percent error = %.1E' % (BP, self.model.results['BP'], 100*abs(BP-self.model.results['BP'])/BP))
        print('True R1 = %.6f; estimated R1 = %.6f; percent error = %.1E' % (R1, self.model.results['R1'], 100*abs(R1-self.model.results['R1'])/R1))
        self.assertAlmostEqual(BP, self.model.results['BP'], delta=5e-2)
        self.assertAlmostEqual(R1, self.model.results['R1'], delta=5e-1)

    @unpack
    @data(*iters)
    def test_srtm_gunn(self, BP, R1, startActivity):
        self.t, self.dt, self.TAC, self.refTAC = generate_fakeTAC_SRTM(BP, R1)
        self.model = SRTM_Gunn1997(self.t, self.dt, self.TAC, self.refTAC,
                                   time_unit='min', startActivity=startActivity)

        print('\nFitting SRTM_Gunn1997 with %s start activity' % startActivity)
        self.model.fit()

        print('True BP = %.6f; estimated BP = %.6f; percent error = %.1E' % (BP, self.model.results['BP'], 100*abs(BP-self.model.results['BP'])/BP))
        print('True R1 = %.6f; estimated R1 = %.6f; percent error = %.1E' % (R1, self.model.results['R1'], 100*abs(R1-self.model.results['R1'])/R1))
        self.assertAlmostEqual(BP, self.model.results['BP'], delta=5e-1)
        self.assertAlmostEqual(R1, self.model.results['R1'], delta=7e-1)
