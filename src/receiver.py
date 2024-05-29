# receiver.py

import numpy as np

from manager import manager, HelpersDataframe, SvResultHandler
from utils import Constants, DTypes, Rotation

from wls import *

class Troposphere:
    """
    Class to handle Troposphere delay following [Collins, 1999]
    https://gssc.esa.int/navipedia/index.php/Tropospheric_Delay
    "Example of Tropospheric model for Standard Point Positioning"
    """
    k1 = 77.604
    k2 = 382000
    Rd = 287.054
    gm = 9.784
    g  = 9.80665

    #                South  North
    TROPO_DOY_MIN = [211,   28]
    MET_TROPO_LATS  = np.array([15, 30, 45, 60, 75])

    #                             P0       T0      e0     B0       la0        Latitude (absolute value)
    MET_TROPO_AVG    = np.array([[1013.25, 299.65, 26.31, 6.30e-3, 2.77],   # 15 or less
                                 [1017.25, 294.15, 21.79, 6.05e-3, 3.15],   # 30
                                 [1015.75, 283.15, 11.66, 5.58e-3, 2.57],   # 45
                                 [1011.75, 272.15,  6.78, 5.39e-3, 1.81],   # 60
                                 [1013.00, 263.65,  4.11, 4.53e-3, 1.55]])  # 75 or greater
    
    #                   Deltas:   P0      T0    e0    B0       la0      Latitude (absolute value)
    MET_TROPO_SEASON = np.array([[ 0.00,  0.00, 0.00, 0.00e-3, 0.00],   # 15 or less
                                 [-3.75,  7.00, 8.85, 0.25e-3, 0.33],   # 30
                                 [-2.25, 11.00, 7.24, 0.32e-3, 0.46],   # 45
                                 [-1.75, 15.00, 5.36, 0.81e-3, 0.74],   # 60
                                 [-0.50, 14.50, 3.39, 0.62e-3, 0.30]])  # 75 or greater
    
    def tropo_select_params(lat):
        if (lat <= Troposphere.MET_TROPO_LATS[0]):
            return Troposphere.MET_TROPO_AVG[0,:], Troposphere.MET_TROPO_SEASON[0,:]
        elif (lat >= Troposphere.MET_TROPO_LATS[-1]):
            return Troposphere.MET_TROPO_AVG[4,:], Troposphere.MET_TROPO_SEASON[4,:]
        else:
            idx0 = np.argmin(np.abs(Troposphere.MET_TROPO_LATS - lat))
            coeff0 = lat/Troposphere.MET_TROPO_LATS[idx0]
            coeff1 = lat/Troposphere.MET_TROPO_LATS[np.minimum(idx0 + 1,4)]
            sum_coeffs = coeff0 + coeff1
            coeff0 = coeff0 / sum_coeffs
            coeff1 = coeff1 / sum_coeffs
            return coeff0 * Troposphere.MET_TROPO_AVG[idx0,:] + coeff1 * Troposphere.MET_TROPO_AVG[np.minimum(idx0 + 1,4),:], \
                   coeff0 * Troposphere.MET_TROPO_SEASON[idx0,:] + coeff1 * Troposphere.MET_TROPO_SEASON[np.minimum(idx0 + 1,4),:]
    
    def tropo_calc_params(lat, doy):
        AVG, SEASON= Troposphere.tropo_select_params(lat)
        N = Troposphere.MET_TROPO_LATS.size
        params = np.zeros(N)
        doy_min = Troposphere.TROPO_DOY_MIN[lat>0] # If lat is positive, it selects the north DOY_MIN

        for ii in range(N):
            params[ii] = AVG[ii] - SEASON[ii] * np.cos(2*np.pi * (doy - doy_min) / 365.25)
        
        return params
    
    def calc_tropo_delay(t_sow, lat, hei, elev, doy, numMeas):
        """
        Main function for Tropospheric Delay calculation.
        :param t_sow: second of week
        :param lat: receiver latitude
        :param hei: receiver height
        :param elev: array with SVs elevation angles
        :param doy: day of year
        :param numMeas: number of measurements present

        :return tropoDelay: array with each SV tropospheric delay in meters
        """
        tropoDelay = np.zeros(numMeas)
        # Get parameters
        P,T,e,B,la = Troposphere.tropo_calc_params(lat * 180/np.pi, doy)

        for ii in range(numMeas):
            el = np.mod(elev[ii], np.pi)
            # Calculate obliquity factor
            M_E = 1.001 / np.sqrt(0.002001 + np.sin(el)**2)
            # Calculate zero-altitude vertical delay for dry and wet components
            T_Z0_DRY = 1e-6 * Troposphere.k1 * Troposphere.Rd * P / Troposphere.gm
            T_Z0_WET = 1e-6 * Troposphere.k2 * Troposphere.Rd / ((la + 1) * Troposphere.gm - B * Troposphere.Rd) * e / T
            # Calculate vertical delay at receiver height for dry and wet components
            T_Z_DRY = (1 - B*hei/T)**(Troposphere.g / (Troposphere.Rd * B)) * T_Z0_DRY
            T_Z_WET = (1 - B*hei/T)**((la + 1) * Troposphere.g / (Troposphere.Rd * B) - 1) * T_Z0_WET
            # Calculate combined dry and wet delay
            tropoDelay[ii] = (T_Z_DRY + T_Z_WET) * M_E

        return tropoDelay

class Ionosphere:
    """
    Class to handle Ionosphere delay calculation following Klobuchar Ionospheric Model
    https://gssc.esa.int/navipedia/index.php?title=Klobuchar_Ionospheric_Model
    """
    def calc_iono_delay(t_sow, alphan, betan, lat, lon, elev, azim, numMeas):
        """
        Main function to calculate the ionospheric delay
        
        :param t_sow: second of week
        :param alphan: alpha parameter for ionospheric model
        :param betan: beta parameter for ionospheric model
        :param lat: receivr latitude
        :param lon: receiver longitude
        :param elev: array with SVs elevation angles
        :param azim: array with SVs azimuth angles
        :param numMeas: number of available satellites

        :return ionoDelay: ionospheric delay in meters
        """
        ionoDelay = np.zeros(numMeas)
        phiu = np.mod(lat, np.pi)
        lambdau = np.mod(lon,np.pi) / np.pi
        ionoDelayAmplitude = 0
        ionoDelayPeriod    = 0
        secondsInDay = 86400
        speedOfLight = 299792458

        for ii in range(numMeas):
            el = np.mod(elev[ii], np.pi) / np.pi
            az = np.mod(azim[ii], np.pi) / np.pi
            # Earth-centred angle (elevation in semicircles)
            psi = np.mod(0.0137 / (el + 0.11) - 0.022, np.pi) / np.pi
            # Latitude of the Ionospheric Pierce Point (IPP)
            phii = phiu + psi * np.cos(az)
            phii = np.sign(phii) * np.minimum(np.abs(phii), 0.416)
            phii = np.mod(phii, np.pi) / np.pi
            # Longitude of IPP
            lambdai = lambdau + psi * np.sin(az) / np.cos(phii)
            lambdai = np.mod(lambdai, np.pi) / np.pi
            # Geomagnetic latitude of IPP
            phim = phii + 0.064 * np.cos(lambdai - 1.617)
            phim = np.mod(phim, np.pi) / np.pi
            # Local time at IPP
            tIPP = 43200 * lambdai + t_sow
            tIPP = np.mod(tIPP, secondsInDay)
            
            # Amplitude of ionospheric delay
            ionoDelayAmplitude = np.sum(alphan * phim ** np.arange(4))
            ionoDelayAmplitude = np.maximum(0, ionoDelayAmplitude)
            # Period of ionospheric delay
            ionoDelayPeriod = np.sum(betan * phim ** np.arange(4))
            ionoDelayPeriod = np.maximum(72000, ionoDelayPeriod)
            # Phase of ionospheric delay
            ionoDelayPhase = 2 * np.pi * (tIPP - 50400) / ionoDelayPeriod
            # Slant factor of ionospheric delay (elevation in semicircles)
            slantFactor = 1 + 16 * (0.53 - el)**3
            # Compute ionospheric delay
            if np.abs(ionoDelayPhase) < 1.57:
                delay = slantFactor * (5e-9 + ionoDelayAmplitude * (1 - 1/2 * ionoDelayPhase**2 + 1/24 * ionoDelayPhase**4))
            else:
                delay = slantFactor * 5e-9
            ionoDelay[ii] = (delay * speedOfLight)
            
        return ionoDelay


class _Helpers:
    """
    Helper class for the receiver module
    """
# 
#  Class for helper functions for Rx PVT processing
#

    def check_valid_sv_masks(dataSys, mode):
        """
        Check the SV validity depending on the chosen mode.

        Remember satellite module gives all available measurements, and then receiver module choses what mode to use.

        Such modes can vary, for example, elevation and cn0 masks.

        :param dataSys: measurements data from specific system, i.e. GPS
        :param mode: chosen mode following Modes class specified in config module by the user.

        :return svResultHandler: handler with satellites which are available following the chosen mode parameters.
        """

        # Filter SV based on modes' mask
        mgrSvResultHandler = manager.results.svResultHandler
        svResultHandler = SvResultHandler()
        
        isSvValid = mgrSvResultHandler.isSvValid
        isSvValid &= dataSys.cn0[manager.currentTimeEpoch.t_ix, mgrSvResultHandler.measIdx] >= mode.cn0_mask
        # Also check the elevation mask, but this will be checked only if elevations are non NAN
        # therefore, at least one PVT solution should have been calculated, since elevation (and azimuth)
        # computation depends on the estimated receiver location.
        if manager.monitor.last_pvt_valid_idx > -1:
            el = dataSys.elevation[manager.currentTimeEpoch.t_ix, mgrSvResultHandler.measIdx]
            isSvValid &= np.rad2deg(el) > mode.elev_mask

        # Recalculate SVs and meas indexes, as well as ovewrite position matrix and number of measurements.
        svResultHandler.isSvValid = isSvValid
        svResultHandler.validSVsIx = np.where(svResultHandler.isSvValid == True)[0]
        svResultHandler.pos = mgrSvResultHandler.pos[svResultHandler.isSvValid,:]
        svResultHandler.measIdx = mgrSvResultHandler.measIdx[svResultHandler.isSvValid]
        svResultHandler.numMeas = isSvValid.sum()
        return svResultHandler

class RxPVTCalculation:
    """
    Main class for Rx PVT processing
    """
    
    def __init__(self, sys_):
        self.sys = sys_
        self.nIters = 0
        self.svResultHandler = SvResultHandler()
        self.svResultHandlerWindow = SvResultHandler()
        self.xRx = np.zeros(4, dtype=DTypes.FLOAT)
        self.xRxDelta = np.zeros(4, dtype=DTypes.FLOAT) + Constants.epsilonErrMag3
        self.jacobian = np.empty([],dtype=DTypes.FLOAT)
        self.S = np.eye(4, dtype=DTypes.FLOAT) * Constants.epsilonErrMag3 
        self.weightsMatrix = np.empty([],dtype=DTypes.FLOAT)
        self.covMatrix = np.empty([], dtype=DTypes.FLOAT)
        self.resnorm = 0


        if manager.config.use_pvt_ref_as_origin:
            self.x_pvt_reference = manager.processing.meas.pvt_reference
        else:
            self.x_pvt_reference = []

    def _calc_atmospheric_corrections(self, numMeas, enable_iono_corr, enable_tropo_corr):
        """
        Private function to handle Atmospheric Corrections: both Ionospheric and Tropospheric

        :param numMeas: number of available measurements
        :paramm enable_iono_corr: boolean from user-specified mode indicating if ionospheric correction should be done.
        :param enable_tropo_corr: boolean from user-specified mode indicating if tropospheric correction should be done.

        :return iono: array with ionospheric delays in meters for each SV.
        :return tropo: array with tropospheric delays in meters for each SV.
        """

        t_ix = manager.currentTimeEpoch.t_ix
        t_sow = manager.currentTimeEpoch.t_sow
        doy = manager.currentTimeEpoch.doy
        iono = np.repeat(0, numMeas)
        tropo = np.repeat(0, numMeas)

        # Get Iono and Tropo delays
        if manager.monitor.last_pvt_valid_idx != -1:
            # Retrieve data from system
            dataSys = manager.processing.meas.get_data(self.sys)

            el, az = dataSys.elevation[t_ix, self.svResultHandler.measIdx], dataSys.azimuth[t_ix, self.svResultHandler.measIdx]
            lat, lon, hei = manager.results.rxResultHandler.pvt[0].loc[manager.monitor.last_pvt_valid_idx][['lat','lon','hei']]

            if enable_iono_corr:
                alphan = manager.processing.ephs.ionoAlpha
                betan  = manager.processing.ephs.ionoBeta

                iono = Ionosphere.calc_iono_delay(t_sow, alphan, betan, lat, lon, el, az, numMeas)
                if np.abs(np.min(iono.min())) > 1000:
                    iono = np.repeat(0, numMeas)

            if enable_tropo_corr:
                tropo = Troposphere.calc_tropo_delay(t_sow, lat, hei, el, doy, numMeas)
                if np.abs(np.min(tropo.min())) > 1000:
                    tropo = np.repeat(0, numMeas)

        return iono, tropo
    

    def _process_pseudo(self, pseudos):
        """
        Process pseudorange, calculate pseudorange estimation and residuals
        
        p = r - t_sv + t_rx + iono + tropo + multipath + noise
        * geometric range r is calculated as the euclidean distance between estiamted Rx and Tx positions.
        * time delays are represented in meters
        * multipath and noise assumed as generic error
        * iono and tropo corrected after the first position estimation, meanwhile they are 0
        * t_rx and xRx are estimations, thus they change in the iterative algorithm until convergence.

        :param pseudos: array containing available satellites' pseudoranges
        
        :return pseudorangeResiduals: pseudorange residuals as the difference between estimated and meausred pseudoranges.
        """

        # Separate xRx and tRx
        timeRxMeters = np.sum(self.xRx[3:])
        # Array with SV time offsets in meters
        dtSv = self.svResultHandlerWindow.pos[:,3]
        xSv = self.svResultHandlerWindow.pos[:,0:3]
        timeSvMetersN = dtSv * Constants.speedOfLight
        # Array with the geometric ranges
        rangeGeometricN = np.linalg.norm((self.xRx[0:3] - xSv).astype(float), 2, axis = 1)

        # Pseudorange Estimation
        pseudorangeEst = rangeGeometricN - timeSvMetersN + np.repeat(timeRxMeters, self.svResultHandlerWindow.numMeas) + self.svResultHandlerWindow.iono + self.svResultHandlerWindow.tropo
        # Calculate Pseudorange Residuals
        pseudorangeResiduals = pseudorangeEst - pseudos
        return pseudorangeResiduals
    

    def _process_jacobian(self):
        """
        Compute Jacobian, i.e. geometry matrix.

        :return jacobian: jacobian Nx4 matrix, where N is the number of measurements.
        
        """
        jacobian = np.zeros((self.svResultHandlerWindow.numMeas, 4), dtype=DTypes.FLOAT)
        rangeGeometricN = np.linalg.norm((self.svResultHandlerWindow.pos[:,:3] - self.xRx[:3]).astype(float), 2, axis = 1)
        jacobian[:,0:3] = (self.svResultHandlerWindow.pos[:,:3] - self.xRx[0:3]) / np.kron(rangeGeometricN, np.ones((3,1))).T
        jacobian[:,3] = -np.ones(self.svResultHandlerWindow.numMeas)
        return jacobian
    

    def _prepare_weights(self, ls_weight_algorithm, dataSys, pRes):
        """
        Prepare information to be used by weight functions for WLS, and calculate weights.

        :params ls_weight_algorithm: function in modes defined by the user stating what weight function in wls module use.
        :params dataSys: system data, i.e. GPS observables.
        :params pRes: pseudorange residuals

        :return weights: array containing the weights for each measurement.
        """
        
        # Until no position is calculated, the weights are identity matrix
        # This is because all other weights depend on measurements that can be calculated
        # after the first fix is obtained, for example, the elevation angle depends on the position fix.
        if manager.monitor.last_pvt_valid_idx == -1:
            return np.ones(self.svResultHandlerWindow.numMeas)
        
        cn0 = dataSys.cn0[manager.currentTimeEpoch.t_ix, self.svResultHandlerWindow.measIdx]
        el = dataSys.elevation[manager.currentTimeEpoch.t_ix, self.svResultHandlerWindow.measIdx]
        az = dataSys.azimuth[manager.currentTimeEpoch.t_ix, self.svResultHandlerWindow.measIdx]
        params = WeightAlgorithmsParameters(self.nIters, self.svResultHandlerWindow.numMeas, cn0, el, az, pRes)
        weights = ls_weight_algorithm(params)
        return weights
    

    def _check_singular(self, jacobian):
        """
        Check geometry to avoid linear dependent columns among the Jacobian rows, thus avoid triggering singular matrix error.
        
        :param jacobian: Nx4 jacobiann matrix

        :return validGeometry: boolean stating if the jeometry is valid (matrix is not singular, i.e. determinant is nonzero).
        """
        validGeometry = True
        if np.linalg.det(jacobian.T @ jacobian) < Constants.epsilonErrMag10:
            validGeometry = False

        return validGeometry


    def _calculate_ls(self, jacobian, weights, residuals):
        """
        LS or WLS fitting calculation

        :param jacobian: Nx4 jacobian matrix.
        :param weights: weights to apply.
        :param residuals: pseudorange residuals.
        """
        self.covMatrix = np.linalg.inv(jacobian.T @ np.diag(weights) @ jacobian) # stored since will be reused for DOP calculation
        self.xRxDelta = self.covMatrix @ (jacobian.T @ np.diag(weights)) @ residuals
        self.xRx += self.xRxDelta
        return 
    
    
    def _process_ls(self, dataSys, mode):
        """
        Main fuction for processing linearized LS loops
        
        :param dataSys: system data, i.e. GPS, containing system observables.
        :param mode: user defined mode to run which applies to LS.

        :return last_iter_ok_mode: boolean stating if the iteration is valid (successful position calculation) or not (diverging calculation, singular matrix).
        """

        # LS processing
        self.nIters = 0
        # Initialize error
        err = 0.1
        # Loop processing
        while(err > Constants.epsilonErrMag3):
            if self.nIters > mode.ls_max_iterations:
                break

            # Compute Pseudorange based on current estimates, apply corrections and obtain residuals to be used in LS
            # Get pseudoranges
            pseudos = dataSys.pseudorange[manager.currentTimeEpoch.t_ix, self.svResultHandlerWindow.measIdx]
            pseudorangeResiduals = self._process_pseudo(pseudos)
            
            # Compute Jacobian
            jacobian = self._process_jacobian()
            
            # Prepare the weights for current mode
            weights = self._prepare_weights(mode.ls_weight_algorithm, dataSys, pseudorangeResiduals)

            # Check geometry to avoid algorithm divergence due to linear dependent rows in the Jacobian
            if not self._check_singular(jacobian):
                is_last_iter_ok_mode = False
                return is_last_iter_ok_mode
            
            # Calculate PVT based on algorithm if geometry is valid
            self._calculate_ls(jacobian, weights, pseudorangeResiduals)

            err = np.linalg.norm(self.xRxDelta, 2)
            self.nIters = self.nIters + 1
        
        self.resnorm = np.linalg.norm(pseudorangeResiduals,2)
        is_last_iter_ok_mode = ~np.isnan(self.xRx).any()
        return is_last_iter_ok_mode

    def _process_mmse(self, dataSys, mode):
        """
        Main fuction for processing linearized MMSE
        
        :param dataSys: system data, i.e. GPS, containing system observables.
        :param mode: user defined mode to run which applies to LS.

        :return last_iter_ok_mode: boolean stating if the iteration is valid (successful position calculation) or not (diverging calculation, singular matrix).
        """

        # Compute Pseudorange based on current estimates, apply corrections and obtain residuals to be used in LS
        # Get pseudoranges
        pseudos = dataSys.pseudorange[manager.currentTimeEpoch.t_ix, self.svResultHandlerWindow.measIdx]
        y = self._process_pseudo(pseudos)

        # Prior PSD
        rngGeom = np.linalg.norm(self.svResultHandlerWindow.pos[:,:3]- self.xRx[:3],axis=1)
        fi = np.array([yy * (2*ss - self.xRx[:3]) / rr**3 + (ss - self.xRx[:3])**2 / rr**2 \
                       for yy,rr,ss in zip(y,rngGeom,self.svResultHandlerWindow.pos[:,:3])]).sum(axis = 0)
        fi = np.hstack([fi, 2 * self.svResultHandlerWindow.numMeas])
        fi = fi / y.var()
        Px = np.diag(1/fi) * mode.mmse_px_mpy

        # Compute Jacobian
        H = self._process_jacobian()

        # Prepare the weights for current mode and calculate measurement covariance matrix
        weights = self._prepare_weights(mode.ls_weight_algorithm, dataSys, y)
        R = np.diag(1/weights**2) * mode.mmse_r_mpy + np.eye(y.size)*Constants.epsilonErrMag6

        # Check geometry to avoid algorithm divergence due to linear dependent rows in the Jacobian
        if not self._check_singular(H):
            is_last_iter_ok_mode = False
            return is_last_iter_ok_mode
        
        # MMSE Estimation covariance matrix
        self.S = np.linalg.inv(np.linalg.inv(Px) +  H.T @ np.linalg.inv(R) @ H)
        self.covMatrix = self.S
        self.xRxDelta = (self.S @ H.T @ np.linalg.inv(R)) @ y
        
        self.xRx += self.xRxDelta
        self.resnorm = np.linalg.norm(y,2)
        is_last_iter_ok_mode = ~np.isnan(self.xRx).any()
        return is_last_iter_ok_mode


    def _run_mode(self, mode):
        """
        Processing for every mode
        
        :param mode: user defined mode to run which applies to LS.

        :return is_last_iter_ok_mode: bool stating if the calculatioon is valid or not coming from _process_ls or _process_mmse .
        """

        # Retrieve data from system
        dataSys = manager.processing.meas.get_data(self.sys)

        # Prepare window or full block processing
        if mode.ls_window_diversity:
            #windowSize = int(4 + 0.5 * (self.svResultHandler.numMeas - 4)) # minimum (4) + 1/2 of the remaining satellites
            #validIdx = np.where(self.svResultHandler.isSvValid == True)[0]
            #ranges = [np.arange(0 + ss, windowSize + ss) for ss in range(self.svResultHandler.numMeas - windowSize + 1)]
            ranges = [np.setdiff1d(np.arange(0,self.svResultHandler.numMeas),ii) for ii in range(0,self.svResultHandler.numMeas)]
            #rangesIx = [np.arange(0 + ss, windowSize + ss) for ss in range(self.svResultHandler.numMeas - windowSize + 1)]
        else:
            windowSize = self.svResultHandler.numMeas
            ranges = [np.arange(0, windowSize)]
            #rangesIx = ranges
        
        # Process LS
        windowOutputs = []
        dops_params = []
        resnorm = []
        # Loopw through windows. Only one loop if no windowing is done.
        for window in ranges:
            self.svResultHandlerWindow = SvResultHandler(
                                            _isSvValid  = self.svResultHandler.isSvValid[np.where(self.svResultHandler.isSvValid==True)[0][window]],
                                            _validSVsIx = self.svResultHandler.validSVsIx[window],
                                            _pos        = self.svResultHandler.pos[window,:],
                                            _measIdx    = self.svResultHandler.measIdx[window],
                                            _iono       = self.svResultHandler.iono[window],
                                            _tropo      = self.svResultHandler.tropo[window], 
                                            _numMeas    = len(window))
            
            # Initialize xRx (position solution) and xRxDelta (LS output)
            if mode.use_ls:
                self.xRxDelta = np.zeros(4, dtype=DTypes.FLOAT)
                if mode.ls_use_last_fix and manager.monitor.last_pvt_valid_bool: # Initialize to last calculated solution
                    self.xRx = np.array(manager.results.rxResultHandler.pvt[mode.idx][['x','y','z','dt']], dtype=DTypes.FLOAT)[manager.monitor.last_pvt_valid_idx]
                else:  # Every iteration is independent from the previous, so re-initialize arrays
                    self.xRx = np.zeros(4, dtype=DTypes.FLOAT)
            else:
                self.xRxDelta = np.zeros(4, dtype=DTypes.FLOAT)
                if manager.monitor.last_pvt_valid_bool: # Initialize to last calculated solution
                    self.xRx = np.array(manager.results.rxResultHandler.pvt[mode.idx][['x','y','z','dt']], dtype=DTypes.FLOAT)[manager.monitor.last_pvt_valid_idx]
                else:  # Every iteration is independent from the previous, so re-initialize arrays
                    self.xRx = np.zeros(4, dtype=DTypes.FLOAT)
            # Run LS
            if mode.use_ls:
                is_last_iter_ok_mode = self._process_ls(dataSys, mode)
            else:
                if not manager.monitor.last_pvt_valid_bool:
                    # Run LS
                    is_last_iter_ok_mode = self._process_ls(dataSys, mode)
                else:
                    # Run MMSE
                    is_last_iter_ok_mode = self._process_mmse(dataSys, mode)
            resnorm = np.hstack([resnorm, self.resnorm])

            # Return in case that calculation for this mode is not OK.
            # Remember that the tool allows multiple modes, and this checking is done to have fair comparison.
            if not is_last_iter_ok_mode:
                return is_last_iter_ok_mode
            
            # Get LLH, and compute DOP parameters
            # In line with https://gssc.esa.int/navipedia/index.php/Positioning_Error#Predicted_Accuracy:_Dilution_of_Precision
            # Note that for Formal Accuracy in the same URL, the multiplication with LLH rotation matrix is needed as
            llh = Rotation.conversion_ecef_2_wgs84(self.xRx[:3])
            gdop = np.sqrt(self.covMatrix.trace())
            
            # Append DOP and position results to average in case that windowing is done.
            dops_params.append(np.diag(Rotation.matrix_wgs84_2_enu(llh).T @ self.covMatrix[:-1,:-1] @ Rotation.matrix_wgs84_2_enu(llh)))
            windowOutputs.append(np.hstack([gdop, self.xRx]))

        # Calculate final position solution and DOP. Notice that gdopWeights = 1 if windowing is not done.
        windowOutputs = np.array(windowOutputs)
        
        # Rather than weighting amonv all solutions, keep the one with smalles DoP.
        #gdopWeights = np.exp(-(windowOutputs[:,0]/np.min(windowOutputs[:,0]))**2)
        #gdopWeights = gdopWeights/gdopWeights.sum()
        self.xRx = windowOutputs[np.argmin(windowOutputs[:,0]),1:] #gdopWeights @ windowOutputs[:,1:]
        self.resnorm = resnorm[np.argmin(windowOutputs[:,0])] #gdopWeights @ resnorm

        dops_params = dops_params[np.argmin(windowOutputs[:,0])] #gdopWeights @ np.array(dops_params)
        hdop = np.sqrt(dops_params[:2].sum())
        vdop = np.sqrt(dops_params[-1])

        # Compute LLH, ENU and DOP
        llh = Rotation.conversion_ecef_2_wgs84(self.xRx[:3])
        
        # Set XYZ reference as the 1st valid position if is not set
        if len(self.x_pvt_reference) == 0:
            self.x_pvt_reference = self.xRx[:3]
            manager.processing.meas.pvt_reference = self.x_pvt_reference

        enu = Rotation.matrix_wgs84_2_enu(llh) @ (self.xRx[:3] - self.x_pvt_reference)
        # Store results in dataframe
        HelpersDataframe.df_add(manager.results.rxResultHandler.pvt[mode.idx],
                                np.hstack([manager.currentTimeEpoch.t_sow, self.nIters, self.xRx, llh, enu, hdop, vdop, self.resnorm]), 
                                ['tsow','it', 'x','y','z','dt','lat','lon','hei','e','n','u','hdop','vdop','resnorm'], 
                                -1)
    
        return is_last_iter_ok_mode


    def process(self):
        """
        Main interface function to call for process Receiver PVT.
        All modes will be executed here, and the bool parameter is_last_iter_ok_mode coming from _run_modes is evaluated to ensure that all modes are valid and can be fairly comparted. 
        """

        # Add new storage line for receiver results
        for mode in manager.config.modes:
            HelpersDataframe.df_add(manager.results.rxResultHandler.pvt[mode.idx],
            np.hstack([manager.currentTimeEpoch.t_sow, np.repeat(np.nan, manager.results.rxResultHandler.pvt[mode.idx].columns.size - 1)]),
            manager.results.rxResultHandler.pvt[mode.idx].columns,
            -1,
            True)
                
        # If num meas is less than minimum required, then output NAN results
        if manager.results.svResultHandler.numMeas < 4:
            manager.monitor.last_pvt_valid_bool = False
            return
        
        # Since several modes are supported in parallel for comparison
        # a control variable is initialized to check if all modes provide a valid solution.
        # This is done for fair comparison of the results.
        is_last_iter_ok_mode = np.full(len(manager.config.modes), True)

        # Retrieve data from system
        dataSys = manager.processing.meas.get_data(self.sys)

        # Processing: loop through modes
        for mode in manager.config.modes:
            # Re validate SVs based on CN0 and Elevation masks. This could have been done in satellites.py, 
            # but doing it here allows for customization for running different modes in parallel.
            self.svResultHandler = _Helpers.check_valid_sv_masks(dataSys, mode)

            # Check again that numMeas is still meeting the minimum requirement based on unknowns.
            if self.svResultHandler.numMeas < 4:
                is_last_iter_ok_mode[mode.idx] = False
                return
                    
            # Calculate iono and tropo corrections
            self.svResultHandler.iono, self.svResultHandler.tropo = self._calc_atmospheric_corrections(self.svResultHandler.numMeas, mode.enable_iono_corr, mode.enable_tropo_corr)

            # Processing of current mode
            is_last_iter_ok_mode[mode.idx] = self._run_mode(mode)

            # Set current epoch as invalid, same as all SVs in current epoch
            if not is_last_iter_ok_mode[mode.idx]:
                break
        
        # Set last tIx with a valid solution in all modes, and indicate 1st valid solution (if not yet indicated).
        if (~ is_last_iter_ok_mode).any():
            manager.monitor.last_pvt_valid_bool = False
            # Set SVs as not valid and discard results in current epoch
            self.svResultHandler.isSvValid[:] = False
            for mode in manager.config.modes:
                HelpersDataframe.df_add(manager.results.rxResultHandler.pvt[mode.idx],
                            np.hstack([manager.currentTimeEpoch.t_sow, np.repeat(np.nan, manager.results.rxResultHandler.pvt[mode.idx].columns.size - 1)]),
                            manager.results.rxResultHandler.pvt[mode.idx].columns,
                            -1)
        else:
            manager.monitor.last_pvt_valid_idx = manager.currentTimeEpoch.t_ix
            manager.monitor.last_pvt_valid_bool = True
        return


