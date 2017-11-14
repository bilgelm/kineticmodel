import numpy as np
import numpy.matlib as mat
from scipy import linalg
from scipy.optimize import curve_fit
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

    # This class will estimate the following parameters:
    param_names = ['BP','R1']

    # This class will provide the following model fit indicators:
    modelfit_names = []

    def __init__(self, t, dt, TAC, refTAC, startActivity,
                 smoothTAC=None):
        '''
        Initialize Zhou 2003 SRTM model.

        Args
        ----
            smoothTAC : np.array
                spatially smoothed time activity curve of the region/voxel/vertex
                of interest (optional - if not provided, [unsmoothed] TAC is used)
        '''
        if smoothTAC is not None:
            if smoothTAC.ndim==1:
                if not len(smoothTAC)==len(t):
                    raise ValueError('smoothTAC and t must have same length')
                # make TAC into a row vector
                smoothTAC = smoothTAC[np.newaxis,:]
            elif smoothTAC.ndim==2:
                if not smoothTAC.shape[1]==len(t):
                    raise ValueError('Number of columns of smoothTAC must be the same \
                                      as length of t')
            else:
                raise ValueError('smoothTAC must be 1- or 2-dimensional')

        super().__init__(t, dt, TAC, refTAC, startActivity)
        self.smoothTAC = smoothTAC

    def fit(self):
        n = len(self.t)
        m = 3

        # diagonal matrix with diagonal elements corresponding to the duration
        # of each time frame
        W = mat.diag(self.dt)

        # Numerical integration of reference TAC
        intrefTAC = km_integrate(self.refTAC,self.t,self.startActivity)
        # Numerical integration of target TAC

        for k, TAC in enumerate(self.TAC):
            intTAC = km_integrate(TAC,self.t,self.startActivity)

            # ----- Get DVR, BP -----
            # Set up the weighted linear regression model
            # based on Eq. 9 in Zhou et al.
            # Per the recommendation in first paragraph on p. 979 of Zhou et al.,
            # smoothed TAC is used in the design matrix, if provided.
            if self.smoothTAC is None:
                X = np.mat(np.column_stack((intrefTAC, self.refTAC, TAC)))
            else:
                X = np.mat(np.column_stack((intrefTAC, self.refTAC, smoothTAC[k,:].flatten())))
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
            y = np.mat(TAC).T
            b = linalg.solve(X.T * W * X, X.T * W * y)
            residual = y - X * b
            var_b = residual.T * W * residual / (n-m)

            R1 = b[0]

            self.params['BP'][k] = BP
            self.params['R1'][k] = R1

        return self

    def refine_R1(smoothb):
        # to be implemented
        raise NotImplementedError()

class SRTM_Lammertsma1996(KineticModel):
    '''
    Compute binding potential (BP) and relative delivery (R1) kinetic parameters
    from dynamic PET data based on simplified reference tissue model (SRTM).
    Reference:
    Simplified reference tissue model for PET receptor studies.Lammertsma AA1,
    Hume SP. Neuroimage. 1996 Dec;4(3 Pt 1):153-8.
    '''

    # This class will estimate the following parameters:
    param_names = ['BP','R1','k2']

    # This class will provide the following model fit indicators:
    modelfit_names = ['err','mse','fpe','logl','akaike']

    def fit(self):
        n = len(self.t)
        m = 4 # 3 model parameters + noise variance

        def make_srtm_est(startActivity):
            '''
            Wrapper to construct the SRTM TAC estimation function with a given
            startActivity.
            Args
            ----
                startActivity : determines initial condition for integration.
                                See integrate in kineticmodel.py
            '''

            def srtm_est(X, BPnd, R1, k2):
                '''
                Compute fitted TAC given t, refTAC, BP, R1, k2.

                Args
                ----
                    X : tuple where first element is t, and second element is intrefTAC
                    BPnd : binding potential
                    R1 : R1
                    k2 : k2
                '''
                t, refTAC = X

                k2a=k2/(BPnd+1)
                # Convolution of reference TAC and exp(-k2a) = exp(-k2a) * Numerical integration of
                # refTAC(t)*exp(k2at).
                integrant = refTAC * np.exp(k2a*t)
                conv = np.exp(-k2a*t) * km_integrate(integrant,t,startActivity)
                TAC_est = R1*refTAC + (k2-R1*k2a)*conv
                return TAC_est
            return srtm_est

        X = (self.t, self.refTAC)
        # upper bounds for kinetic parameters in optimization
        BP_upper, R1_upper, k2_upper = (20.,10.,8.)

        srtm_fun = make_srtm_est(self.startActivity)

        for k, TAC in enumerate(self.TAC):
            popt, pcov = curve_fit(srtm_fun, X, TAC,
                                   bounds=(0,[BP_upper, R1_upper, k2_upper]))
            y_est = srtm_fun(X, *popt)

            sos=np.sum(np.power(TAC-y_est,2))
            err = np.sqrt(sos)/n
            mse =  sos / (n-m) # 3 par + std err
            fpe =  sos * (n+m) / (n-m)

            SigmaSqr = np.var( TAC-y_est )
            logl = -0.5*n* math.log( 2* math.pi * SigmaSqr) - 0.5*sos/SigmaSqr
            akaike = -2*logl + 2*m # 4 parameters: 3 model parameters + noise variance

            self.params['BP'][k], self.params['R1'][k], self.params['k2'][k] = popt

            self.modelfit['err'][k] = err
            self.modelfit['mse'][k] = mse
            self.modelfit['fpe'][k] = fpe
            self.modelfit['logl'][k]= logl
            self.modelfit['akaike'][k] = akaike

        return self
