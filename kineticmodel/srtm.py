import math
import numpy as np
import numpy.matlib as mat
from scipy.linalg import solve
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
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

    # This class will compute the following results:
    result_names = [ # estimated parameters
                    'BP',
                    'R1','k2','k2a',
                    'R1_lrsc','k2_lrsc','k2a_lrsc',
                    # model fit indicators
                    'noiseVar_eqDVR','noiseVar_eqR1']

    def fit(self, smoothTAC=None):
        if smoothTAC is not None:
            if smoothTAC.ndim==1:
                if not len(smoothTAC)==len(self.t):
                    raise ValueError('smoothTAC and t must have same length')
                # make smoothTAC into a row vector
                smoothTAC = smoothTAC[np.newaxis,:]
            elif smoothTAC.ndim==2:
                if not smoothTAC.shape==self.TAC.shape:
                    raise ValueError('smoothTAC and TAC must have same shape')
            else:
                raise ValueError('smoothTAC must be 1- or 2-dimensional')

        n = len(self.t)
        m = 3

        # diagonal matrix with diagonal elements corresponding to the duration
        # of each time frame
        W = mat.diag(self.dt)

        # Numerical integration of reference TAC
        intrefTAC = km_integrate(self.refTAC,self.t,self.startActivity)

        # Compute BP/DVR, R1, k2, k2a
        for k, TAC in enumerate(self.TAC):
            # Numerical integration of target TAC
            intTAC = km_integrate(TAC,self.t,self.startActivity)

            # ----- Get DVR, BP -----
            # Set up the weighted linear regression model
            # based on Eq. 9 in Zhou et al.
            # Per the recommendation in first paragraph on p. 979 of Zhou et al.,
            # smoothed TAC is used in the design matrix, if provided.
            if smoothTAC is None:
                X = np.mat(np.column_stack((intrefTAC, self.refTAC, -TAC)))
            else:
                X = np.mat(np.column_stack((intrefTAC, self.refTAC, -smoothTAC[k,:].flatten())))

            y = np.mat(intTAC).T
            b = solve(X.T * W * X, X.T * W * y)
            residual = y - X * b
            noiseVar_eqDVR = residual.T * W * residual / (n-m) # unbiased estimator of noise variance

            DVR = b[0]
            #R1 = b[1] / b[2]
            #k2 = b[0] / b[2]
            BP = DVR - 1

            # ----- Get R1 -----
            # Set up the weighted linear regression model
            # based on Eq. 8 in Zhou et al.
            X = np.mat(np.column_stack((self.refTAC,intrefTAC,-intTAC)))
            y = np.mat(TAC).T
            b = solve(X.T * W * X, X.T * W * y)
            residual = y - X * b
            noiseVar_eqR1 = residual.T * W * residual / (n-m) # unbiased estimator of noise variance

            R1 = b[0]
            k2 = b[1]
            k2a = b[2]

            self.results['BP'][k] = BP # distinguish between BP estimated using smoothed v. unsmoothed TAC?
            self.results['R1'][k] = R1
            self.results['k2'][k] = k2
            self.results['k2a'][k] = k2a

            self.results['noiseVar_eqDVR'][k] = noiseVar_eqDVR
            self.results['noiseVar_eqR1'][k] = noiseVar_eqR1

        return self

    def refine_R1(self, smoothR1, smoothk2, smoothk2a, h):
        # Ridge regression to get better R1, k2, k2a estimates
        #
        # (smoothR1, smoothk2, smoothk2a) are the values to drive the estimates toward
        # h is the diagonal elements of the matrix used to compute the weighted norm

        if not smoothR1.ndim==smoothk2.ndim==smoothk2a.ndim==1:
            raise ValueError('smoothR1, smoothk2, smoothk2a must be 1-D')
        if not len(smoothR1)==len(smoothk2)==len(smoothk2a)==self.TAC.shape[0]:
            raise ValueError('Length of smoothR1, smoothk2, smoothk2a must be \
                             equal to the number of rows of TAC')
        if not h.ndim==2:
            raise ValueError('h must be 2-D')
        if not h.shape==(self.TAC.shape[0], 3):
            raise ValueError('Number of rows of h must equal the number of rows of TAC, \
                             and the number of columns of h must be 3')

        # diagonal matrix with diagonal elements corresponding to the duration
        # of each time frame
        W = mat.diag(self.dt)

        # Numerical integration of reference TAC
        intrefTAC = km_integrate(self.refTAC,self.t,self.startActivity)
        for k, TAC in enumerate(self.TAC):
            # Numerical integration of target TAC
            intTAC = km_integrate(TAC,self.t,self.startActivity)

            # ----- Get R1 incorporating spatial constraint -----
            # Set up the ridge regression model
            # based on Eq. 11 in Zhou et al.
            X = np.mat(np.column_stack((self.refTAC,intrefTAC,-intTAC)))
            y = np.mat(TAC).T
            H = mat.diag(h[k,:])
            b_sc = np.mat( (smoothR1[k],smoothk2[k],smoothk2a[k]) ).T
            b = solve(X.T * W * X + H, X.T * W * y + H * b_sc)

            R1_lrsc = b[0]
            k2_lrsc = b[1]
            k2a_lrsc = b[2]

            self.results['R1_lrsc'][k] = R1_lrsc
            self.results['k2_lrsc'][k] = k2_lrsc
            self.results['k2a_lrsc'][k] = k2a_lrsc

        return self

class SRTM_Lammertsma1996(KineticModel):
    '''
    Compute binding potential (BP) and relative delivery (R1) kinetic parameters
    from dynamic PET data based on simplified reference tissue model (SRTM).
    Reference:
    Simplified reference tissue model for PET receptor studies.Lammertsma AA1,
    Hume SP. Neuroimage. 1996 Dec;4(3 Pt 1):153-8.
    '''

    # This class will compute the following results:
    result_names = [ # estimated parameters
                    'BP','R1','k2',
                    # model fit indicators
                    'err','mse','fpe','logl','akaike']

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
                                   bounds=(0,[BP_upper, R1_upper, k2_upper]),
                                   sigma=1/np.sqrt(self.dt), absolute_sigma=False)
            y_est = srtm_fun(X, *popt)

            sos=np.sum(np.power(TAC-y_est,2))
            err = np.sqrt(sos)/n
            mse =  sos / (n-m) # 3 par + std err
            fpe =  sos * (n+m) / (n-m)

            SigmaSqr = np.var( TAC-y_est )
            logl = -0.5*n* math.log( 2* math.pi * SigmaSqr) - 0.5*sos/SigmaSqr
            akaike = -2*logl + 2*m # 4 parameters: 3 model parameters + noise variance

            self.results['BP'][k], self.results['R1'][k], self.results['k2'][k] = popt

            self.results['err'][k] = err
            self.results['mse'][k] = mse
            self.results['fpe'][k] = fpe
            self.results['logl'][k]= logl
            self.results['akaike'][k] = akaike

        return self
