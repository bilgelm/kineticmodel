from abc import ABCMeta, abstractmethod
from scipy import integrate as sp_integrate
import numpy as np

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
                time activity curve of the region/voxel/vertex of interest OR
                two-dimensional array where each row corresponds to
                a region/voxel/vertex of interest and each column
                corresponds to a time point
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
        if not len(t)==len(dt)==len(refTAC):
            raise ValueError('KineticModel inputs t, dt, refTAC must have same length')

        if TAC.ndim==1:
            if not len(TAC)==len(t):
                raise ValueError('KineticModel inputs TAC and t must have same length')
        elif TAC.ndim==2:
            if not TAC.shape[1]==len(t):
                raise ValueError('Number of columns of TAC must be the same \
                                  as length of t')
        else:
            raise ValueError('TAC must be 1- or 2-dimensional')

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

        self.BP = None
        self.R1 = None

    def fit(self, **kwargs):
        # do all assignments to self here.
        # _fit_one should be a static method.
        if self.TAC.ndim==1:
            (self.BP, self.R1) = self._fit_one(self.TAC, **kwargs)
            return self
        elif self.TAC.ndim==2:
            return self._fit_many(**kwargs)
        else:
            raise ValueError('TAC must be 1- or 2-dimensional')

    @abstractmethod
    def _fit_one(self, TAC, **kwargs):
        '''
        Abstract method for fitting a kinetic model given a time activity curve (TAC)
        for the region/voxel/vertex of interest.
        '''
        pass

    def _fit_many(self, **kwargs):
        '''
        Method for fitting a kinetic model given multiple time activity curves
        (TACs). Subclasses should overwrite this method if necessary.
        '''

        est = np.apply_along_axis(self._fit_one, axis=1, arr=self.TAC, **kwargs)

        self.BP = est[:,0].flatten() # order='F' #?
        self.R1 = est[:,1].flatten()

        return self

def _strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def integrate(TAC, t, startActivity):
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
    intTAC = sp_integrate.cumtrapz(TAC,t,initial=initialIntegralValue)

    return intTAC
