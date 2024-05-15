# satellite.py

import numpy as np
from numba import jit
from manager import manager
from utils import Constants, Rotation, DTypes, dot_vec_vec, dot_mat_vec


class _Helpers:
    """
    Helper class for the Numba compiled functions from satellite module.

    NOTE: this class depends on ephSv which is originally a DataFrame. It is converted to NumPy and
    as an array and the positions corresponding to every label are taken.\n
    This is needed because Numba does not support Pandas. 
    The indexes for every DataFrame column can be easily seen with,
    for example: `np.column_stack([ephSv.columns.to_numpy(), np.arange(ephSv.columns.size)])`
    in the process() function before converting ephSv to numpy.
    """
    
    @jit(nopython = True)
    def calc_tx_time(t_sow, pseudo, ephSv, speedOfLight):
        """
        Numba compiled function to calculate satellite time offset.
        
        :param t_sow: second of week
        :param pseudo: pseudorange
        :param ephSv: satellite ephemerides struct
        :param speedOfLight: speed of light constant, since Constants class is not accessible from Numba.

        :return timeTxFromSv: the time offset from the SV.
        :return deltaTimeRel: relativistic delta time, which estimation is enhanced in calc_pos_and_clkcorr.
        :return deltaTimeSv: total delta time as stated in the ICD documentation.
        """

        # Calculate t_sv in Eq. 1 of ICD.
        timeTxFromSv = t_sow - pseudo/speedOfLight
        # Calculate t_sv - t_oc of Eq. 2 of ICD. This is t - t_oc, but t is approximated to t_sv as indicated in ICD, so do t_sv - t_oc
        deltaTimeSoW = timeTxFromSv - ephSv[11] # Toe = 11
        #deltaTimeSoW = 0

        # Adjust deltaTimeSoW calculation depending on boundaries
        #if deltaTimeSoW >= secondsInHalfWeek:
        #    deltaTimeSoW -= secondsInWeek
        #elif deltaTimeSoW < -secondsInHalfWeek:
        #    deltaTimeSoW += secondsInWeek

        deltaTimeRel = 0 # Relativistic time is not substracted here, instead it is iteratively calculated in next step,
                         # the reason is that we don't yet have Ek, which is computed during SV position and clock algorithm
                         # following Newthon's method. Therefore, it is included there.
        
        # Compute correction term based on clock polynomial
        timesArray = np.array([ephSv[0], # SVClockBias = 0
                                ephSv[1], # SVclockDrift = 1
                                ephSv[2]  # SVclockDriftRate = 2
                                ])
        delta_times_array = np.array([1, deltaTimeSoW, deltaTimeSoW**2])
        
        deltaTimeSv = dot_vec_vec(timesArray, delta_times_array) # TODO: cambiar por @

        deltaTimeSv = deltaTimeSv + deltaTimeRel
        
        # Stated in IS-GPS-200-M, on section "20.3.3.3.3.1 User Algorithm for SV Clock Correction":
        # The user can compute 1st and 2nd derivatives of deltaTimeSv, however this is not done here.
        
        # Return t_sv, delta_t and delta_rel (this last unused). Eq. 1: t = t_sv - delta_t, but not computed in next step, after delta_rel (which is part of delta_t) is calculated.
        return timeTxFromSv, deltaTimeRel, deltaTimeSv
    
    @jit(forceobj=True)
    def calc_pos_and_clkcorr(t_sow,ephSv,timeSv,deltaTimeSv,deltaTimeRel,
                           muGravConstant, forceF, omegaEDot,
                           secondsInHalfWeek, secondsInWeek, epsilonErrMag10):
        """
        Numba compiled function to calculate satellite position in ECEF frame and clock error.
        
        :param t_sow: second of week
        :param ephSv: satellite ephemerides struct
        :param timeSv: first estimation of satellite time offset as calculated in calc_tx_time.
        :param deltaTimeSv: first estimation of satellite delta time offset as calculated in calc_tx_time.
        :param deltaTimeRel: first estimation of satellite relativistic time offset as calculated in calc_tx_time.
        :param muGravConstant: Earth gravitational constant, since Constants class is not accessible from Numba.
        :param forceF: force constant, since Constants class is not accessible from Numba.
        :param forceF: force constant, since Constants class is not accessible from Numba.
        :param omegaEDot: Earth spin rate, since Constants class is not accessible from Numba.
        :param secondsInHalfWeek: seconds in half week, since Constants class is not accessible from Numba.
        :param secondsInWeek: seconds in week, since Constants class is not accessible from Numba.
        :param epsilonErrMag10: constant to determine when the loop estimation of Ek is enough, since Constants class is not accessible from Numba.

        :return  xyzECEF: SV position in ECEF frame.
        :return svClkCorr: total SV clock correction.
        """
                
        # Compute semi-major axis A and mean motion n0
        A = ephSv[10]**2
        sqrtA = ephSv[10]
        ecc = ephSv[8]
        n0 = np.sqrt(muGravConstant / A**3)

        # Here we will compute the relativistic time to minimize
        while (True):
            # SV time at time of TX
            timeSvCorrected = timeSv - deltaTimeRel
            # Time from Ephemeris Reference
            tk = timeSvCorrected - ephSv[11]
            # Adjust tk calculation depending on boundaries
            if tk >= secondsInHalfWeek:
                tk -= secondsInWeek
            elif tk < -secondsInHalfWeek:
                tk += secondsInWeek
            # Correct Mean Motion
            n0Corrected = n0 + ephSv[5]
            # Correct Mean Anomaly
            Mk = ephSv[6] + n0Corrected * tk
            # Compute Initial Eccentric anomaly
            E0 = Mk
            Ek = 0
            # Correct Ek from Kepler's Equation via Newthon Raphson
            while (np.abs(Mk - (Ek - ecc * np.sin(Ek))) > epsilonErrMag10):
                Ek = E0 + (Mk - E0 + ecc * np.sin(E0)) / (1 - ecc * np.cos(E0))
                E0 = Ek
            # Compute updated relativistic time delta
            deltaTimeRelUpdated = forceF * ecc * sqrtA * np.sin(Ek)
            #svClkCorr = deltaTimeSv - deltaTimeRelUpdated
            if (np.abs(deltaTimeRel - deltaTimeRelUpdated) > epsilonErrMag10):
                deltaTimeRel = deltaTimeRelUpdated
            else:
                break
        
        # Run one more with the rinest deltaTimeRelUpdated:
        # SV time at time of TX
        timeSvCorrected = timeSv - deltaTimeRelUpdated
        # Time from Ephemeris Reference
        tk = timeSvCorrected - ephSv[11]
        # Adjust tk calculation depending on boundaries
        if tk >= secondsInHalfWeek:
            tk -= secondsInWeek
        elif tk < -secondsInHalfWeek:
            tk += secondsInWeek
        # Correct Mean Motion
        n0Corrected = n0 + ephSv[5]
        # Correct Mean Anomaly
        Mk = ephSv[6] + n0Corrected * tk
        # Compute Initial Eccentric anomaly
        E0 = Mk
        Ek = 0
        # Correct Ek from Kepler's Equation via Newthon Raphson
        while (np.abs(Mk - (Ek - ecc * np.sin(Ek))) > epsilonErrMag10):
            Ek = E0 + (Mk - E0 + ecc * np.sin(E0)) / (1 - ecc * np.cos(E0))
            E0 = Ek

        # Compute updated relativistic time delta
        #deltaTimeRelUpdated = forceF * ecc * sqrtA * np.sin(Ek)
        #svClkCorr = deltaTimeSv - deltaTimeRelUpdated
        #if (np.abs(deltaTimeRel - deltaTimeRelUpdated) > epsilonErrMag10):
        #    deltaTimeRel = deltaTimeRelUpdated
        #else:
        #    break
            
        # Calculate True Anomaly vk, update for Ek, and argument of Latitude thetak
        vk = 2 * np.arctan(np.sqrt((1 + ecc)/(1 - ecc)) * np.tan(Ek / 2) )
        thetak = vk + ephSv[17]

        # Calculate corrections for second harmonic perturbations:
        # Argument of Latitude and correction
        deltaUk = ephSv[9] * np.sin(2 * thetak) + ephSv[7] * np.cos(2 * thetak)
        uk = thetak + deltaUk
        # Radius and correction
        deltaRk = ephSv[4] * np.sin(2 * thetak) + ephSv[16] * np.cos(2 * thetak)
        rk = A * (1 - ecc * np.cos(Ek)) + deltaRk
        # Inclination and Correction
        deltaIk = ephSv[14] * np.sin(2 * thetak) + ephSv[12] * np.cos(2 * thetak)
        ik = ephSv[15] + deltaIk + ephSv[19] * tk

        # Calculate SV Position in orbital plane
        # Since calculations are in orbital fixed plane, later have to be rotated to ECEF plane,
        # as well as account for the time difference between ECEF at Tx time vs ECEF at Rx time.
        # This is because, the ECEF axes rotate with time, and there are some ms of difference between the
        # transmission and reception time due to the signal propagation time.
        xk0 = rk * np.cos(uk)
        yk0 = rk * np.sin(uk)
        # Calculate corrected Longitude of Ascending Node, to be utilized to plane rotation from orbital plane to Earth fix plane.
        OmegaK = ephSv[13] + tk * (ephSv[18] - omegaEDot) - omegaEDot * ephSv[11]
        # Rotation matrix from Orbital to Fix plane. Basically, the equations in Table IV for xk', yk' and zk' in matricial form.
        R_EarthFix = np.array([[np.cos(OmegaK), - np.cos(ik) * np.sin(OmegaK), 0], \
                               [np.sin(OmegaK),   np.cos(ik) * np.cos(OmegaK), 0], \
                               [0,                np.sin(ik),                  0]])

        # Now interpret satellite position frame as ECI = ECEF at Tx time.
        # This means that previous rotation matrix will rotate from Orbital to ECEF, but to ECEF at Tx time.
        x0_vec = np.array([xk0, yk0, 0])
        xyzECI = dot_mat_vec(R_EarthFix, x0_vec) #TODO: cambiar por @
        # We want to account for how much the Earth (and consequently the ECEF plane) has rotated from the Tx time to the Rx time, and finally calculate
        # the SV in the ECEF plane at Rx time. Therefore, below we generate an ECI2ECEF matrix accouning for the Earth rotation during the Tx-Rx travel time.
        
        # Calculate SV position in ECEF at Rx time following 20.3.3.4.3.3.2 Earth-Centered, Inertial (ECI) Coordinate System.
        # Note that in the IS-GPS-200M document, the ECEF-to-ECI rotation is explained, and we want to do the opposite.
        # This is as generating the rotation matrix as stated in the document, and transpose it (rotation matrix property).
        omegaTravelTime = omegaEDot * (t_sow - timeSvCorrected)
        R_ECI2ECEF = np.array([[np.cos(omegaTravelTime), - np.sin(omegaTravelTime), 0], \
                               [np.sin(omegaTravelTime),   np.cos(omegaTravelTime), 0], \
                               [0,                         0,                       1]]).T # Inverse = Transpose
        xyzECEF = dot_mat_vec(R_ECI2ECEF, xyzECI) # TODO: cambiar por @

        ## Calculate SV Velocity
        # TBD

        # Calculate total SV clock correction (recall that in Eq. 2 the relativity delta time is subtracting).
        svClkCorr = deltaTimeSv + deltaTimeRelUpdated

        # Return and Store in general matrix
        return xyzECEF, svClkCorr
    
    @jit(nopython = True)
    def calc_sv_elev_and_az(xSv, xRx, Rot):
        """
        Numba compiled function to calculate satellite elevation and azimuth angles.
        
        :param xSv: SV position in ECEF frame.
        :param xRx: Receiver position in ECEF frame.
        :param Rot: Rotation matrix depending on receiver latitude, longitude and height.

        :return elevation: SV elevation angle in radians.
        :return azimuth: SV azimuth angle in radians.

        """
        # Direction vector Rx - Sv
        xDiff_norm = xSv - xRx
        # Calculate direction cosine
        xDiff_norm /= np.sqrt(dot_vec_vec(xDiff_norm, xDiff_norm))
        # Get vector coordinates in local plane
        xDiff_enu = dot_mat_vec(Rot, xDiff_norm)
        # Calculate Elevation and Azimuth
        elevation = np.arcsin(xDiff_enu[2])
        azimuth = np.arctan2(xDiff_enu[0], xDiff_enu[1])

        return elevation, azimuth

class SvPVTCalculation:
    """
    Main class to handle SV PVT calculation: position in ECEF frame and time offset.
    """
    def __init__(self, sys_):
        self.svPosClkCorr = [] # xyzPos, timeSv, deltaTimeSv : #PRNs x 5
        self.sys = sys_
        self.timeSv = 0
        self.deltaTimeSv = 0
        self.svResultHandler = manager.results.svResultHandler

    def calc_tx_time(self, t_sow, ephSv, pseudo, sigTypeFreq):
        """
        Calculate Tx time and clock correction except for relativistic effect.\n
        "20.3.3.3.3.1 User Algorithm for SV Clock Correction" on IS-GPS-200M\n
        NOTE: This functions is a wrapper to Numba (C-compiled) function to accelerate processing.

        :param t_sow: second of week
        :param ephSv: satellite ephemerides struct
        :param pseudo: pseudorange
        :param sigTypeFreq: signal type frequency, currently only GPS supported.
        """
        timeTxFromSv, self.deltaTimeRel, self.deltaTimeSv = _Helpers.calc_tx_time(t_sow, pseudo, ephSv, Constants.speedOfLight)
        # Apply group delay based on L1/L2
        if sigTypeFreq == Constants.GPS_L1_FREQ_HZ:
            self.deltaTimeSv -= ephSv[25]
        else:
            self.deltaTimeSv -= Constants.GPS_GAMMA_L1L2 * ephSv[25]

        self.timeSv = timeTxFromSv - self.deltaTimeSv
        return


    def calc_pos_and_clkcorr(self,t_sow,ephSv): 
        """
        Estimation of satellite position in ECEF and clock correction.\n
        "20.3.3.4.3 Algorithms for Ephemeris Determination" on IS-GPS-200M \n
        Basically, Table IV is followed. The Keplerian paramaters are utilized to determine the SV position and clock correction.
        This is done after the Tx time estimation and initial clock correction (which did not account for relativistic effect).\n\n
        In fact, relativistic effect depends on Eccentric Anomaly Ek, which is determined during this algorithm previous to the SV position estimation.\n\n
        Note that, the term 't' in equation 'tk = t - toe' in Table IV on ICD document, is taken as 'timeSvCorrected' in this algorithm,
        and corresponds to the initial Tx time estimation done in function calcTxTime(), which is then corrected by the relativistic term in every loop iteration in calc_pos_and_clkcorr()
        since the relativistic term is estimated together with Ek.\n\n
        This way of handling the relativistic term is also commented on function calcTxTime().
        NOTE: This functions is a wrapper to Numba (C-compiled) function to accelerate processing.

        :param t_sow: second of week
        :param ephSv: satellite ephemerides struct

        :return: concatenated SV postion and time error
        """
        xyz, dt = _Helpers.calc_pos_and_clkcorr(t_sow,ephSv,self.timeSv,self.deltaTimeSv,self.deltaTimeRel,
                                                                           Constants.muGravConstant, Constants.forceF, Constants.omegaEDot,
                                                                           Constants.secondsInHalfWeek, Constants.secondsInWeek, Constants.epsilonErrMag10)
        
        return np.hstack([xyz, dt])
        
        
    
    def process(self):
        """
        Main function to process satellite raw pseudoranges, extimate satellite clock offset and position in ECEF plane.
        """
        meas = manager.processing.meas
        ephs = manager.processing.ephs
        
        measList = []
        posList = []
        t_ix = manager.currentTimeEpoch.t_ix
        t_sow = manager.currentTimeEpoch.t_sow

        calc_el_az_from_ref = manager.config.calc_el_az_from_ref
        last_pvt_valid_idx = manager.monitor.last_pvt_valid_idx
        if last_pvt_valid_idx != -1:
            last_pvt_ecef = np.array(manager.results.rxResultHandler.pvt[0].iloc[last_pvt_valid_idx][['x', 'y', 'z']], dtype=DTypes.FLOAT)
            last_pvt_llh = np.array(manager.results.rxResultHandler.pvt[0].iloc[last_pvt_valid_idx][['lat', 'lon', 'hei']], dtype=DTypes.FLOAT)

        # Loop over all GNSS systems ['G', 'E']. Currently only GPS supported.
        for sysIndicator in self.sys:
            # Get data from system
            dataSys = meas.get_data(sysIndicator)
            # Loop over all SVs per system
            for prnIx, is_valid in enumerate(dataSys.visible[t_ix,:]):
                # Check if visible
                if is_valid:
                    prn = meas.sv[meas.sv[:,1] == sysIndicator,0][prnIx].astype('uint8')
                    prnIxEphem = prn - 1 # prnIx for ephemerides won't work since is not of dimension 32
                    # Get pseudorange to be used later as a Tx time approximation
                    pseudo  = dataSys.pseudorange[t_ix,prnIx]
                    # Get SV ephemeris for given PRN and receiver time
                    t = meas.t[t_ix]
                    valid, ephSv = ephs.check_eph_validity(t, prnIxEphem)
                    if valid:
                        # Calculate coarse SV Tx time
                        self.calc_tx_time(t_sow, ephSv, pseudo, Constants.GPS_L1_FREQ_HZ)
                        # Calculate SV position (ECEF), and clock correction
                        svPosClkCorr = self.calc_pos_and_clkcorr(t_sow, ephSv)
                        # Calculate elevation and azimuth, we need to ensure that a fix is already gotten, since elevation and azimuth calculation depend on it.

                        if last_pvt_valid_idx > -1:
                            if calc_el_az_from_ref:
                                ref_ecef = manager.processing.meas.pvt_reference
                                ref_llh = Rotation.conversion_ecef_2_wgs84(ref_ecef)
                                el, az = _Helpers.calc_sv_elev_and_az(svPosClkCorr[0:3], ref_ecef, Rotation.matrix_wgs84_2_enu(ref_llh))
                            else:
                                el, az = _Helpers.calc_sv_elev_and_az(svPosClkCorr[0:3], last_pvt_ecef, Rotation.matrix_wgs84_2_enu(last_pvt_llh))
                        else:
                            el = np.nan
                            az = np.nan

                        # Add elevation and azimuth to dataManager database
                        dataSys.elevation[t_ix, prnIx] = el
                        dataSys.azimuth[t_ix, prnIx] = az
                        
                        measList.append(prnIx)
                        posList.append(svPosClkCorr)
                    else:
                         dataSys.visible[t_ix,prnIx] = False


        # Store current epoch results
        try:
            self.svResultHandler.pos = np.array(posList)
            self.svResultHandler.measIdx = np.array(measList)
            # Filter based on validity (check no NAN in position and clock computation)
            self.svResultHandler.isSvValid = ~ np.isnan(self.svResultHandler.pos).any(axis=1)
            self.svResultHandler.isSvValid  = self.svResultHandler.isSvValid & (np.sign(dataSys.elevation[t_ix,self.svResultHandler.measIdx]) != -1)
            self.svResultHandler.validSVsIx = np.where(self.svResultHandler.isSvValid == True)[0]
            self.svResultHandler.numMeas = self.svResultHandler.isSvValid.astype('uint8').sum()
        except:
            self.svResultHandler.pos = np.array([])
            self.svResultHandler.measIdx = np.array([])
            # Filter based on validity (check no NAN in position and clock computation)
            self.svResultHandler.isSvValid = []
            self.svResultHandler.validSVsIx = []
            self.svResultHandler.numMeas = 0
        return
    