import numpy as np
import numpy.matlib as mat
from scipy import linalg
import math
from kineticmodel import KineticModel
from kineticmodel import integrate as km_integrate

class SRTM_Zhou2003(KineticModel):
    '''
    Compute binding potential (BP) and relative delivery (R1) kinetic parameters
    from dynamic PET data based on simplified reference tissue model (SRTM).
    The nonlinear SRTM equations are linearized using integrals of time
    activity curves (TACs) of the reference and target tissues. Kinetic
    parameters are then estimated by weighted linear regression (WLR).
    If provided, the spatially smoothed TAC of the target region is used in
    the computation of BP as part of the linear regression with spatial
    constraint (LRSC) approach.

    To obtain the R1 estimate that incorporates spatial smoothness based on
    LRSC, run refine_R1() after running fit().

    Reference:
    Zhou Y, Endres CJ, Brašić JR, Huang S-C, Wong DF.
    Linear regression with spatial constraint to generate parametric images of
    ligand-receptor dynamic PET studies with a simplified reference tissue model.
    Neuroimage. 2003;18:975–989.
    '''

    '''
    Args
    ----
        smoothTAC : np.array
            spatially smoothed time activity curve of the region/voxel/vertex
            of interest (optional - if not provided, [unsmoothed] TAC is used)
    '''
    def __init__(self, t, dt, TAC, refTAC, startActivity):
        super().__init__(t, dt, TAC, refTAC, startActivity)

        # Perform tasks that are independent of the ROI TAC and store the results

        # diagonal matrix with diagonal elements corresponding to the duration
        # of each time frame
        self.W = mat.diag(self.dt)

        # Numerical integration of reference TAC
        self.intrefTAC = km_integrate(self.refTAC,self.t,self.startActivity)

    def _fit_one(self, TAC, smoothTAC=None):
        n = len(self.t)
        m = 3

        # Numerical integration of target TAC
        intTAC = km_integrate(TAC,self.t,self.startActivity)

        # ----- Get DVR, BP -----
        # Set up the weighted linear regression model
        # based on Eq. 9 in Zhou et al.
        # Per the recommendation in first paragraph on p. 979 of Zhou et al.,
        # smoothed TAC is used in the design matrix, if provided.
        if smoothTAC is None:
            smoothTAC = TAC

        X = np.mat(np.column_stack((self.intrefTAC, self.refTAC, smoothTAC)))
        y = np.mat(intTAC).T
        b = linalg.solve(X.T * self.W * X, X.T * self.W * y)
        residual = y - X * b
        var_b = residual.T * self.W * residual / (n-m)

        DVR = b[0]
        BP = DVR - 1

        # ----- Get R1 -----
        # Set up the weighted linear regression model
        # based on Eq. 8 in Zhou et al.
        X = np.mat(np.column_stack((self.refTAC,self.intrefTAC,-intTAC)))
        y = np.mat(TAC).T
        b = linalg.solve(X.T * self.W * X, X.T * self.W * y)
        residual = y - X * b
        var_b = residual.T * self.W * residual / (n-m)

        R1 = b[0]

        return (BP, R1)

    def _fit_many(self, smoothTAC=None):
        if smoothTAC is None:
            # call parent class' implementation of _fit_many
            return super(SRTM_Zhou2003, self)._fit_many()
        else:
            numROIs = self.TAC.shape[0]
            self.BP = np.zeros(numROIs)
            self.R1 = np.zeros(numROIs)

            for k in range(numROIs):
                est = self._fit_one(self.TAC[k,:],smoothTAC=smoothTAC[k,:])
                self.BP[k], self.R1[k] = est

            return self

    def refine_R1(self, smoothb):
        # to be implemented
        raise NotImplementedError()
