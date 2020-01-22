def generate_fakeTAC_SRTM(BP,R1):
    # generate fake TAC using SRTM ODE model

    import numpy as np
    from scipy.integrate import odeint
    from temporalimage import Quantity

    frameStart = Quantity(np.array([0,
                                    0.25, 0.5, 0.75, 1.0,
                                    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                                    6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    17, 20,
                                    25, 30, 35, 40, 45, 50, 55, 60, 65, 70]),
                           'minute')
    frameEnd = Quantity(np.array([0.25, 0.5, 0.75, 1.0,
                                  1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                                  6, 7, 8, 9, 10, 11, 12, 13, 14,
                                  17, 20,
                                  25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]),
                        'minute')
    frameDuration = frameEnd - frameStart
    frameMidPoint = 0.5*(frameStart+frameEnd)

    def model(y,t,BP,R1):
        '''
        ODE describing SRTM
        '''
        plasma_rate = -0.03

        Kref1 = 1.0
        K1 = R1 * Kref1

        DVR = 1+BP
        kref2 = 0.3
        k2a = kref2 * R1 / DVR

        Cp = y[0]
        Ct = y[1]
        Cr = y[2]

        dCpdt = plasma_rate * Cp
        dCtdt = K1*Cp - k2a*Ct
        dCrdt = Kref1*Cp - kref2*Cr

        dydt = [dCpdt,dCtdt,dCrdt]

        return dydt

    # initial condition
    y0 = [200, 0, 0]

    # number of time points
    numODEpts = 500

    t_ODE = np.linspace(frameStart[0].magnitude, frameEnd[-1].magnitude,
                        numODEpts)
    y = odeint(model,y0,t_ODE,args=(BP,R1))

    # "Digitize" this curve
    Cref = np.zeros(len(frameMidPoint))
    Ct = np.zeros(len(frameMidPoint))
    for ti in range(len(frameMidPoint)):
        idx = (t_ODE>=frameStart[ti]) & (t_ODE<frameEnd[ti])
        Ct[ti] = y[idx,1].mean()
        Cref[ti] = y[idx,2].mean()

    return (frameMidPoint, frameDuration, Ct, Cref)
