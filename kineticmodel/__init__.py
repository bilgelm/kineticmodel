# import main class
from .kineticmodel import KineticModel, integrate
from .srtm import SRTM_Zhou2003, SRTM_Lammertsma1996, SRTM_Gunn1997

try:
    import kineticmodel.nipype_wrapper
except ImportError:
    print('Install kineticmodel using nipype option.')
