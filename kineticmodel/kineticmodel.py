from abc import ABCMeta, abstractmethod
from scipy import integrate as sp_integrate
import numpy as np

class KineticModel(metaclass=ABCMeta):
    # possible values for startActivity
    startActivity_values = ('flat','increasing','zero')

    def __init__(self, t, dt, TAC, refTAC,
                 startActivity='flat'):
        '''
        Method for initializing a kinetic model.
        Defines required inputs for all kinetic models.

        Args
        ----
            t : np.array
                time corresponding to each point of the time activity curve (TAC)in [seconds]
            dt : np.array
                duration of each time frame in [seconds]
            TAC : np.array 1- or 2-D
                each row is the time activity curve of a region/voxel/vertex of interest
                if 1-D, can be a column or row vector
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
        if not t.ndim==dt.ndim==refTAC.ndim==1:
            raise ValueError('KineticModel inputs t, dt, refTAC must be 1-D')
        if not len(t)==len(dt)==len(refTAC):
            raise ValueError('KineticModel inputs t, dt, refTAC must have same length')

        if TAC.ndim==1:
            if not len(TAC)==len(t):
                raise ValueError('KineticModel inputs TAC and t must have same length')
            # make TAC into a row vector
            TAC = TAC[np.newaxis,:]
        elif TAC.ndim==2:
            if not TAC.shape[1]==len(t):
                raise ValueError('Number of columns of TAC must be the same \
                                  as length of t')
        else:
            raise ValueError('TAC must be 1- or 2-dimensional')

        if not (t[0]>=0):
            raise ValueError('Time of initial frame must be >=0')
        if not strictly_increasing(t):
            raise ValueError('Time values must be monotonically increasing')
        if not all(dt>0):
            raise ValueError('Time frame durations must be >0')

        if not (startActivity in KineticModel.startActivity_values):
            raise ValueError('startActivity must be one of: ' + str(KineticModel.startActivity_values))

        self.t = t
        self.dt = dt
        self.TAC = TAC
        self.refTAC = refTAC
        self.startActivity = startActivity

        self.results = {}

        for result_name in self.__class__.result_names:
            self.results[result_name] = np.empty(self.TAC.shape[0])
            self.results[result_name].fill(np.nan)

    @abstractmethod
    def fit(self, **kwargs):
        # update self.results
        return self

    def save_result(self, result_name):
        if not (result_name in self.__class__.result_names):
            raise ValueError(result_name + ' must be one of ' + self.__class__.result_names)

        result = self.results[result_name]

        # write result to csv file
        raise NotImplementedError()

def strictly_increasing(L):
    '''
    Check if L is a monotonically increasing vector
    '''
    return all(x<y for x, y in zip(L, L[1:]))

def integrate(TAC, t, startActivity='flat'):
    '''
    Static method to perform time activity curve integration.
    '''
    if not TAC.ndim==t.ndim==1:
        raise ValueError('TAC must be 1-dimensional')
    if not len(t)==len(TAC):
        raise ValueError('TAC and t must have same length')
    if not (startActivity in KineticModel.startActivity_values):
        raise ValueError('startActivity must be one of: ' + str(KineticModel.startActivity_values))

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
