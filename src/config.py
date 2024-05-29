# config.py
import os
import numpy as np

from wls import *
from utils import Rotation

class Mode():
    """
    Main class defining the user defined modes.

    :param _name: Default ''. Defines the mode name. Plots will take this name as label.
    :param _use_ls: Default True. Bool to determine if using LS (=True) or MMSE (=False).
    :param _ls_use_last_fix: Default False. Bool to determine if pick LS from last iteration solution (=True), or start iterations from scratch (=False).
    :param _ls_weight_algorithm: Default WeightAlgorithms().algo_linear. This is the function from class WeightAlgorithms defining the weights. LS, is the corresponding WLS, and in MMSE, these weights are applied in the measurement noise matrix R.  
    :param _ls_max_iterations: Default 15. Max number of iterations in LS.
    :param _ls_exclusion: Default Fasle. If True, in LS it will run the LS process N times, each time using N-1 measurements (excluding one), and finally pick the solution with smallest GDOP.
    :param _elev_mask: Default 5. Elevation mask so that only SVs with higher elevation will be considered as valid. Unit degrees.
    :param _cn0_mask: Default is 10. CN0 mask so that only SVs with higher CN0 will be considered as valid. Units dB/Hz.
    :param _enable_iono_corr: Default True. Boolean to enbale (=True) or disable (=False) ionospheric correction.
    :param _enable_tropo_corr: Default True. Boolean to enbale (=True) or disable (=False) tropospheric correction.
    :param _mmse_px_mpy: Default 1. If MMSE (_use_ls = False) then the prior covariance matrix Px is multiplied by this factor to ensure more lr less trustness on prior.
    :param _mmse_r_mpy: Default 1. If MMSE (_use_ls = False) then the measurement covariance matrix R is multiplied by this factor to ensure more lr less trustness on mesaurements.
    """

    def __init__(self,
                 _name = '',
                 _use_ls = True,
                 _ls_use_last_fix = False, 
                 _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                 _ls_max_iterations = 15,
                 _ls_exclusion = False,
                 _elev_mask = 5,
                 _cn0_mask = 10,
                 _enable_iono_corr = True,
                 _enable_tropo_corr = True,
                 _mmse_px_mpy = 1,
                 _mmse_r_mpy = 1,
                 ):
        self.name = _name
        self.use_ls = _use_ls
        self.ls_use_last_fix = _ls_use_last_fix
        self.ls_weight_algorithm = _ls_weight_algorithm
        self.ls_max_iterations = _ls_max_iterations
        self.ls_window_diversity = _ls_exclusion
        self.elev_mask = _elev_mask
        self.cn0_mask = _cn0_mask
        self.enable_iono_corr = _enable_iono_corr
        self.enable_tropo_corr = _enable_tropo_corr
        self.mmse_px_mpy = _mmse_px_mpy,
        self.mmse_r_mpy = _mmse_r_mpy,
        self.idx = [] # Do not set, this is set automatically in manager.py
        return

MODES_ATMOSPHERIC_COMPARISON = [
                            # Normal LS
                            Mode('No Tropo/Iono corr', 
                                _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                                _enable_iono_corr = False,
                                _enable_tropo_corr = False),

                            # Normal LS
                            Mode('Iono corr only', 
                                _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                                _enable_iono_corr = True,
                                _enable_tropo_corr = False),

                            # Normal LS
                            Mode('Iono + Tropo corr', 
                                _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                                _enable_iono_corr = True,
                                _enable_tropo_corr = True)
                                ]


MODES_ELEVATION_FILTERING = [
                    # Normal LS
                    Mode('No filtering', 
                        _ls_use_last_fix = True,
                        _ls_weight_algorithm = WeightAlgorithms().algo_linear),

                    # Normal LS filtering low elevation SVs
                    Mode('Elevation filtering', 
                        _ls_use_last_fix = True,
                        _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                        _elev_mask = 20),

                    # Normal LS filtering low elevation SVs and low CN0
                    Mode('Elevation and CN0 filtering', 
                        _ls_use_last_fix = True,
                        _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                        _elev_mask = 20,
                        _cn0_mask = 30),
                        ]

MODES_LSE_VS_FE     = [
                    # LS
                    Mode('LS',
                         _ls_use_last_fix = True,
                         _elev_mask = 20,
                         _cn0_mask = 20),
                    
                    # LS + Exclusion
                    Mode('LS FE',
                         _ls_use_last_fix = True,
                         _ls_exclusion = True,
                         _elev_mask = 20,
                         _cn0_mask = 20),
                        ]

MODES_LSE_VS_WLS     = [
                    # LS
                    Mode('LS',
                         _ls_use_last_fix = True,
                         _elev_mask = 20,
                         _cn0_mask = 20),
                    
                    # WLS ELEVATION + CN0
                    Mode('WLS ELEVATION',
                         _ls_use_last_fix = True,
                         _elev_mask = 20,
                        _cn0_mask = 20, 
                        _ls_weight_algorithm = WeightAlgorithms().algo_cn0_elev),

                    # WLS TUNED ELEVATION + CN0
                    Mode('WLS TUNED ELEVATION',
                         _ls_use_last_fix = True,
                         _elev_mask = 20,
                        _cn0_mask = 20, 
                        _ls_weight_algorithm = WeightAlgorithms().algo_cn0_elev_tuned),
                        ]

MODES_LSE_VS_MMSE     = [
                    # LS
                    Mode('LS',
                         _ls_use_last_fix = False,
                         _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                         _elev_mask = 20,
                         _cn0_mask = 20),

                    # MMSE Linear
                    Mode('MMSE Linear',
                         _elev_mask = 20,
                        _cn0_mask = 20, 
                        _use_ls = False,
                        _ls_weight_algorithm = WeightAlgorithms().algo_linear,
                        _mmse_px_mpy = 1,
                        _mmse_r_mpy = 0.5)
                        ]

class Config:
    """
    Main class for Config.

    :param modes: defined Modes
    :param imgsdir: name of directory to create within imgs/ where result plot images and CSV with statistics will be stored. The directory will be created, and if it alrady exists an error message will be triggered to either remove or rename it to avoid mistakenly ovewriting previous reaults.
    :param fileObs: RINEX observation file in /data/obs folder
    :param fileNav: RINEX navigation file in /data/nav folder. If empty, then the program will download the corresponding navigation file for the day in the observation file from the CDDIS server.
    :param use_pvt_ref_as_origin: Bool. If True, it will use the user-defined PVT reference or RINEX XYZ reference for ENU calculation. If False it uses the 1st PVT estimated.
    :param pvt_reference: user-defined PVT reference in ECEF frame. If Latitude, Longitude and Height are used, they can be converted with Rotation.conversion_wgs84_2_ecef()
    :param calc_el_az_from_ref: Bool. If True, decides to calculate satellite elevation and azimuth angles from PVT reference, if False it will use the last PVT estimation.
    :param plotsExcludeDOP: Bool. If True then the result plots and statistics will exclude all fixes with HDOP bigger than the HDOP's mean + half sigma for each mode.
    :param maxEpochsPercentage: Number from 0 - 100. Is the percentage of total epochs that will be processed. Example: In a static 30s daily observation file with a total of 2880 epochs, we can decide to process only the first 10% by setting the value to 10.

    
    **NOTE** on **fileNav**: when downloaded from server it is stored in /data/nav. If when reading the downloaded file it triggers an error saying that it cannot open the file, try to manually uncompress it and write the uncompressed filename in fileNav.

    **NOTE** on **use_pvt_ref_as_origin**: if True, it will use the user-defined PVT reference, **only if** the RINEX XYZ reference is not present.
    If the RINEX XYZ reference is present, it will use that one instead of the user-sdefined.
    Program assumes that reference in RINEX, if present, is correct. If False, it will use the 1st calculated PVT as reference.
    This corresponds to the 1st calculated mode: if LS and MMSE are run in that order, the reference will be the LS result.

    **NOTE** on **calc_el_az_from_ref**: On static locations, we can calculate the satellite elevation and azimuth from reference rather than from receiver estimates.
    """
    
    def __init__(self):
        self.modes                 = MODES_LSE_VS_MMSE
        self.imgsdir               = 'MODES_LSE_VS_MMSE_TEST'
        self.fileObs               = 'madr1110.23o'
        self.fileNav               = ''
        self.use_pvt_ref_as_origin = True
                                     #Rotation.conversion_wgs84_2_ecef(np.array([LAT_RADIANS,LON_RADIANS,HEI_METERS]))
        self.pvt_reference         = np.array([  4849202.3940,  -360328.9929,  4114913.1862]) # Recall that this takes effect only if use_pvt_ref_as_origin = True, and observation RINEX does not have a reference XYZ.
        self.calc_el_az_from_ref   = True
        self.plotsExcludeDOP       = False
        self.maxEpochsPercentage   = 20

