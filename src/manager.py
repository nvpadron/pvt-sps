# manager.py

import numpy as np
import pandas as pd
from numba import jit
import os

from utils import Constants, SysInfo, TimeConverters
from readRinex import ReadRinex
from config import Config

class _Helpers:
    """
    Class to handle the JIT accelerated processing for Ephemerides module and other helper functions
    """
    
    def get_gr_meas_label(label):
        if len(label) == 2:
            return ''
        else:
            return 'C'
        
    @jit(nopython = True)
    def check_eph_validity_jit(tdiff, prnIx, ephemeridesDim, ephemeridesTuple):
        validTimesIdx = np.where(tdiff[tdiff >= 0] < 4 * 3600)[0]
        data = ephemeridesTuple[0][validTimesIdx,prnIx]
        validEphsIdx = ~ np.isnan(data)
        validEphsIdx = np.where(validEphsIdx == True)[0]
        if validEphsIdx.size == 0:
            validEphFound = False
        else:
            validEphFound = True
            validTimeIdx = validTimesIdx[validEphsIdx[-1]]

        ephSv = np.zeros(ephemeridesDim)
        for ii in np.arange(ephemeridesDim):
            ephSv[ii] = ephemeridesTuple[ii][validTimeIdx,prnIx]
            
        return validEphFound, ephSv, validTimeIdx
    

class SvResultHandler:
    """
    Class for handling the satellite module measurements under usage
    """

    def __init__(self, _isSvValid = [], _validSVsIx = [], _pos = [], _measIdx = [], _iono = [], _tropo = [], _numMeas = []):
        self.isSvValid  = _isSvValid
        self.validSVsIx = _validSVsIx
        self.pos        = _pos
        self.measIdx    = _measIdx
        self.iono       = _iono
        self.tropo      = _tropo
        self.numMeas    = _numMeas

    def reset(self):
        self = self.__init__()  


class RxResultHandler:
    """
    Class for handling the receiver module results
    """

    def __init__(self, modes):
        # RX PVT Manager Template
        self.pvtTemplate = pd.DataFrame(columns = ['tsow','it','x','y','z','dt','lat','lon','hei','e','n','u','hdop','vdop','resnorm']) # xyz, llh, enu
        # RX PVT Manager
        self.pvt = []
        # Result Stats Manager
        self.stats = pd.DataFrame(columns = ['std_e', 'std_n', 'std_u', 'rms_e','rms_n','rms_u','rms_2d','rms_3d', 'CEP50','hdop','vdop'])
        self.pvt = [pd.DataFrame(self.pvtTemplate) for _ in range(0,len(modes))] 

class ResultHandlers:
    """
    Class containing the results for satellite and receiver modules.
    """

    def __init__(self, modes):
        # Sv handler
        self.svResultHandler = SvResultHandler()
        # RX PVT Manager
        self.rxResultHandler = RxResultHandler(modes)

class _EphHandlerModule:
    """
    Private class to handle ephemerides, check the validity and pick the closest one in past time.
    """

    def __init__(self, dataNav):
        self.timesEpochs = dataNav.time
        self.ephemerides = dataNav
        toc = self.ephemerides['TransTime']
        for ii in range(toc.shape[1]):
            toc[toc[:,ii] > Constants.secondsInWeek,ii] = toc[toc[:,ii] > Constants.secondsInWeek,ii] - Constants.secondsInWeek
        self.ephemerides['Toc'] = toc
        self.ephemeridesLabels = [label for label in self.ephemerides.data_vars]
        # Tuple of ephemerides
        self.ephemeridesTuple = tuple(np.array(self.ephemerides[label]) for label in self.ephemeridesLabels)

        self.ionoAlpha = dataNav.ionospheric_corr_GPS[:4]
        self.ionoBeta = dataNav.ionospheric_corr_GPS[4:]
        return    

    def check_eph_validity(self, t, prnIx):
        tdiff = np.array(t - self.timesEpochs).astype('timedelta64[s]').astype(int)
        validEphFound, ephSv, self.validTimeIdx = _Helpers.check_eph_validity_jit(tdiff, prnIx, len(self.ephemeridesLabels), self.ephemeridesTuple)
        
        return validEphFound, ephSv
    
    def get_ephs_and_check(self, t, prnIx, label): # Unused
        validEphsIdx = np.array(t - self.timesEpochs).astype('timedelta64[s]').astype(int)
        validEphsIdx = np.where(validEphsIdx[validEphsIdx > 0] < 4 * 3600)[0]
        values = np.array(self.ephemerides.data_vars.get(label)[validEphsIdx,prnIx])
        value = values[~ np.isnan(values)][-1]
        status = value.size > 0
        return value, status
    
    def get_ephs(self, prnIx):
        data = [np.array(self.ephemerides.data_vars.get(label)[self.validTimeIdx,prnIx]) for label in self.ephemeridesLabels[:-1]]
        df = pd.DataFrame(columns = self.ephemeridesLabels[:-1])
        df.loc[0] = data
        toc = df['TransTime'][0]
        if toc > 604800:
            toc -= 604800
        df.insert(int(df.size), 'Toc', toc)
        dfLabels = dict(zip(df.columns, np.arange(df.columns.size)))
        return df, dfLabels
    
class _MeasSystemBand:
    """
    Private class to handle measurements per band
    """

    def __init__(self, idx, dataObs):
       # Obs data
       labels = np.array([label for label in dataObs.data_vars])
       labelStyle = _Helpers.get_gr_meas_label(labels[0])
       lab = [ll[:2] for ll in labels]
       if lab.count('C1'):
           self.pseudorange = np.array(dataObs.data_vars.get('C1' + labelStyle)[:,idx])
       elif lab.count('P1'):
           self.pseudorange = np.array(dataObs.data_vars.get('P1' + labelStyle)[:,idx])
       else:
           print('ERROR: no pseudorange found')
           manager.monitor.status = False
           return
       
       measDims = self.pseudorange.shape

       if lab.count('S1'):
           self.cn0 = np.array(dataObs.data_vars.get('S1' + labelStyle)[:,idx])
       else:
           self.cn0 = np.full(measDims, 0)
       
       if lab.count('D1'):
           self.doppler = np.array(dataObs.data_vars.get('D1' + labelStyle)[:,idx])
       else:
           self.doppler = -np.ones(measDims)
       self.visible = ~np.isnan(self.cn0) * ~np.isnan(self.pseudorange)
       self.elevation = np.full(measDims, np.nan)
       self.azimuth  = np.full(measDims, np.nan)
       self.pseudorange_smoothed = np.zeros(measDims)

       return
        

class _ObsHandlerModule:
    """
    Private class to handle measurmmenets from observation file. Currently only GPS L1 supported.
    """

    def __init__(self, dataObs):
       # Rx control
       self.t = dataObs.time
       self.interval = dataObs.interval
       self.numEpochs = self.t.size
       self.sv = np.array([[sv[1:], sv[0]] for sv in np.array(dataObs.sv)])
       # Obs data
       if dataObs.attrs.get('position'):
           self.pvt_reference = np.array(dataObs.position)
       else:
           self.pvt_reference = []
       self._dataGPS = _MeasSystemBand(self.sv[:,1] == SysInfo.GPS_INDICATOR, dataObs)

       self._dataSys = {SysInfo.GPS_INDICATOR : self._dataGPS}
       return
    
    def get_data(self, sysIndicator):
        return self._dataSys[sysIndicator]

class Interface:
    """
    Interface class to handle progress bar
    """
    def __init__(self):
        self.progress_bar = []
        self.numEpochs = []

    def show_progress(self, currentTime):
        bin_percentage = 0.05
        per = 0.0

        if np.mod(currentTime.t_ix, np.floor(bin_percentage * self.numEpochs)) == 0 or (currentTime.t_ix + 1 == self.numEpochs):
            if currentTime.t_ix == 0:
                self.progress_bar = '['
            elif currentTime.t_ix + 1 < self.numEpochs:
                self.progress_bar += ' ='
            elif currentTime.t_ix + 1 >= self.numEpochs:
                self.progress_bar += ' ]'

            os.system('cls')
            per = (100 * (currentTime.t_ix + 1) / self.numEpochs)
            print('Processing t_sow = %(t_sow).2f [%(per).2f%%]' % {'t_sow': currentTime.t_sow, 'per' : per})
            print(self.progress_bar)
        return
        

class Monitor:
    """
    Monitor class to store sanity checks.
    """
    def __init__(self):
         # Monitor
        self.status = True
        self.last_pvt_valid_idx = -1
        self.last_pvt_valid_bool = False

class Time:
    """
    Class for to map observation file second of week, day of year and corresponding epoch index.
    """
    def __init__(self, _t, _t_ix, interval = 0):
        self.t_sow = TimeConverters.sec_of_week(_t, _t_ix, interval)
        self.doy   = TimeConverters.day_of_year(_t)[0]
        self.t_ix  = _t_ix
        return

class Processing:
    """
    Class read and store the data to be used in processing: observation and ephemerides modules read from their respective RINEX files.
    """
    def __init__(self):

        # Observations
        self.meas = []
        self.ephs = []

    def _read_and_parse(self, config):
        # Read
        readRinex = ReadRinex(config.fileObs, config.fileNav)
        if readRinex.read_rinex_file() == False:
            print('Error reading input files.')
            manager.monitor.status = False
            return

        dataObs, dataNav = readRinex.get_rinex_data()
        # Parse
        self.meas = _ObsHandlerModule(dataObs)
        self.ephs = _EphHandlerModule(dataNav)

    def load_data(self, config):
        # Read
        self._read_and_parse(config)
        if manager.monitor.status:
            if config.use_pvt_ref_as_origin == False:
                self.meas.pvt_reference = []
            else:
                if (len(self.meas.pvt_reference) == 0):
                    self.meas.pvt_reference = config.pvt_reference
        return
    

class HelpersDataframe:
    """
    Helper class to handle data dataframe indexing for receiver results
    """

    def df_reset(df, index):
        return df.drop(index)

    def df_new(df, index):
        df.loc[index] = np.repeat(np.nan, len(df.columns))

    def df_add(df, data, columns, index = -1, is_new = False):
        if index == -1:
            index = len(df)
        if is_new:
            HelpersDataframe.df_new(df, index)
        if index == len(df):
            index -= 1
        df.loc[index, columns] = data

    def df_get(df, columns, index):
        if np.size(index) == 1:
            if index == -1:
                index = len(df) - 1
        return df.loc[index][columns]
    

class Manager:
    """
    Manager class which encompasses Monitor, Processing, Interface and Time classes to control the processing flow.
    This class also invokes the data configuration (user defined modes) and data loading (from OBS or NAV RINEX files into respective modules).
    """

    def __init__(self):
        # Monitor
        self.monitor = Monitor()
        # Processing
        self.processing = Processing()
        # Interface
        self.interface = Interface()
        # Time control
        self.timeEpochs = []
        self.currentTimeEpoch = []
    
    def load_config(self):
        # Config
        self.config = Config()

        # Assign idx
        for ii in range(len(manager.config.modes)):
            manager.config.modes[ii].idx = ii
        
        # Put imgsdir in lowercase
        manager.config.imgsdir = str.lower(manager.config.imgsdir)

        if os.listdir('imgs/').count(self.config.imgsdir):
            print(f'\nERROR: Directory imgs/{self.config.imgsdir} already exists (check similar lowercase/uppercase folders). Delete or rename it to avoid mistakenly ovewriting previous results.')
            manager.monitor.status = False

        # Results
        self.results = ResultHandlers(self.config.modes)
    
    def load_data(self):
        # Load data in Processing
        self.processing.load_data(self.config)
        if self.monitor.status == False:
            return
        # Time control
        self.timeEpochs = [Time(t, t_ix, self.processing.meas.interval) for t_ix, t in enumerate(self.processing.meas.t)]
        self.timeEpochs = self.timeEpochs[:int(manager.config.maxEpochsPercentage/100 * self.processing.meas.numEpochs)]
        self.currentTimeEpoch = []
        # Set numEpochs for Interface
        self.interface.numEpochs = int(manager.config.maxEpochsPercentage/100 * self.processing.meas.numEpochs)
        return

manager = Manager()
"""
Manager is accessible everywhere in the program, so it is defined here in order to be common for all modules and files. Variable name is `manager`.
"""

