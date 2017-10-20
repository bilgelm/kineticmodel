import numpy.matlib as mat
from scipy import integrate, linalg
import math

class SRTM(KineticModel):
    def __init__(self, t, dt, TAC, refTAC,
                 TAC_smoothed=None):
        super().__init__(t, dt, TAC, refTAC)
        self.TAC_smoothed = TAC_smoothed

    def fit():
        n = len(t)
        m = 3

        W = mat.diag(dt)

        # Numerical integration of reference region TAC
        intrefTAC = integrate.cumtrapz(self.refTAC,self.t,initial=0)
        # Numerical integration of TAC
        intTAC = integrate.cumtrapz(self.TAC,self.t,axis=3,initial=0)

        # Get DVR, BP
        if TAC_smoothed is None:
            X = np.mat(np.column_stack((intrefTAC, self.refTAC, self.TAC)))
        else:
            X = np.mat(np.column_stack((intrefTAC, self.refTAC, self.TAC_smoothed)))
        y = np.mat(intTAC).T
        b = linalg.solve(X.T * W * X, X.T * W * y)
        residual = y - X * b
        var_b = residual.T * W * residual / (n-m)

        dvr = b[:,0]
        bp = dvr - 1

        # Get R1
        X = np.mat(np.column_stack((self.refTAC,intrefTAC,-intTAC)))
        y = np.mat(self.TAC).T
        b = linalg.solve(X.T * W * X, X.T * W * y)
        residual = y - X * b
        var_b = residual.T * W * residual / (n-m)

        r1 = b[:,0]

        return (bp, r1)
