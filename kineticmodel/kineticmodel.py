from abc import ABCMeta, abstractmethod

class KineticModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, t, dt, TAC, refTAC):
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
        '''

        self.t = t
        self.dt = dt
        self.TAC = TAC
        self.refTAC = refTAC

    @abstractmethod
    def fit(self):
        pass
