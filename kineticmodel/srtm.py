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
    def __init__(self, t, dt, TAC, refTAC, startActivity,
                 smoothTAC=None):
        super().__init__(t, dt, TAC, refTAC, startActivity)
        self.smoothTAC = smoothTAC

    def fit(self):
        n = len(self.t)
        m = 3

        # diagonal matrix with diagonal elements corresponding to the duration
        # of each time frame
        W = mat.diag(self.dt)

        # Numerical integration of target TAC
        intTAC = km_integrate(self.TAC,self.t,self.startActivity)
        # Numerical integration of reference TAC
        intrefTAC = km_integrate(self.refTAC,self.t,self.startActivity)

        # ----- Get DVR, BP -----
        # Set up the weighted linear regression model
        # based on Eq. 9 in Zhou et al.
        # Per the recommendation in first paragraph on p. 979 of Zhou et al.,
        # smoothed TAC is used in the design matrix, if provided.
        if self.smoothTAC is None:
            X = np.mat(np.column_stack((intrefTAC, self.refTAC, self.TAC)))
        else:
            X = np.mat(np.column_stack((intrefTAC, self.refTAC, self.smoothTAC)))
        y = np.mat(intTAC).T
        b = linalg.solve(X.T * W * X, X.T * W * y)
        residual = y - X * b
        var_b = residual.T * W * residual / (n-m)
        
        DVR = b[0]
        BP = DVR - 1

        # ----- Get R1 -----
        # Set up the weighted linear regression model
        # based on Eq. 8 in Zhou et al.
        X = np.mat(np.column_stack((self.refTAC,intrefTAC,-intTAC)))
        y = np.mat(self.TAC).T
        b = linalg.solve(X.T * W * X, X.T * W * y)
        residual = y - X * b
        var_b = residual.T * W * residual / (n-m)

        R1 = b[0]

        self.BP = BP
        self.R1 = R1

        # return self #???
        return (BP, R1)

    def refine_R1(smoothb):
        # to be implemented
        pass
