from abc import ABCMeta, abstractmethod
from scipy import integrate as sp_integrate

class KineticModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, t, dt, TAC, refTAC,
                 startActivity='increasing'):
        '''
        Abstract method for initializing a kinetic model.
        Defines required inputs for all kinetic models.

        Args
        ----
            t : np.array
                time corresponding to each point of the time activity curve (TAC)
            dt : np.array
                duration of each time frame
            TAC : np.array
                time activity curve of the region/voxel/vertex of interest
            refTAC : np.array
                time activity curve of the reference region
                NOTE: maybe this should not be a required input for all
                      KineticModels -- if there is arterial sampling, then
                      refTAC is not needed.
            startActivity : one of 'flat', 'increasing', or 'zero'
                defines the method for determining the value of the initial
                integral \int_0^{t_0} TAC(t) dt (default: 'increasing')
                if 'flat', TAC(t)=TAC(t_0) for 0≤t<t_0, which results in this
                    integral evaluating to t_0 * TAC(t_0)
                if 'increasing', TAC(t)=TAC(t_0) / t_0 * t for 0≤t<t_0,
                    which results in this integral evaluating to t_0 * TAC(t_0) / 2
                if 'zero', TAC(t)=0 for 0≤t<t_0, which results in this integral
                    evaluating to 0
        '''

        # basic input checks
        if not len(t)==len(dt)==len(TAC)==len(refTAC):
            raise ValueError('KineticModel inputs t, dt, TAC, refTAC must have same length')
        if not (t[0]>=0):
            raise ValueError('Time of initial frame must be >=0')
        if not _strictly_increasing(t):
            raise ValueError('Time values must be monotonically increasing')
        if not all(dt>0):
            raise ValueError('Time frame durations must be >0')

        self.t = t
        self.dt = dt
        self.TAC = TAC
        self.refTAC = refTAC
        self.startActivity = startActivity

        self.params = {}
        self.modelfit = {}

        for param_name in self.__class__.param_names:
            self.params[param_name] = None

        for modelfit_name in self.__class__.modelfit_names:
            self.modelfit[modelfit_name] = None

    @abstractmethod
    def fit(self):
        # update self.params
        # update self.modelfit
        return self

def _strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def integrate(TAC, t, startActivity, axis=-1):
    '''
    Static method to perform time activity curve integration.
    '''
    if startActivity=='flat':
        # assume TAC(t)=TAC(t_0) for 0≤t<t_0
        initialIntegralValue = t[0]*TAC[0]
    elif startActivity=='increasing':
        # assume TAC(t)=TAC(t_0) / t_0 * t for 0≤t<t_0
        initialIntegralValue = t[0]*TAC[0]/2
    elif startActivity=='zero':
        # assume TAC(t)=0 for 0≤t<t_0
        initialIntegralValue = 0

    # Numerical integration of TAC
    intTAC = sp_integrate.cumtrapz(TAC,t,initial=initialIntegralValue,axis=axis)

    return intTAC
