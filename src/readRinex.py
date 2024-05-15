# readRinex.py

import georinex as rx
from ftplib import FTP_TLS
from os import getcwd

from utils import TimeConverters

class ReadRinex:
    """
    Main class to read rinex
    """

    fileObs = ''
    fileNav = ''
    dataDir = ''

    def __init__(self, fileObs_, fileNav_):
        self.fileObs = fileObs_
        self.fileNav = fileNav_
        self.dataDir = getcwd() + '/data'

    def _download_from_server(self, directory, filename, ext):
        ftps = FTP_TLS(host = 'gdc.cddis.eosdis.nasa.gov')
        ftps.login(user='anonymous')
        ftps.prot_p()
        ftps.cwd(directory)
        fileNav = self.dataDir + '/nav/' + filename + 'gz'
        ftps.retrbinary("RETR " + filename + ext, open(fileNav, 'wb').write)
        return fileNav

    def _download_nav_data(self, ext):
        doy, year = TimeConverters.day_of_year(self.dataObs.time[0])
        filename = 'brdc' + '%.3d' % doy + '0.' + str(year)[2:] + 'n.'
        directory = '/gnss/data/daily/' + str(year) + '/' + '%.3d' % doy + '/' + str(year)[2:] + 'n/'
        fileNav = self._download_from_server(directory, filename, ext)
        return fileNav
    

    def read_rinex_file(self):
        """
        Main function to read RINEX files called by Manager. \n
        Observation and navigation files are read, and if navigation file is not provided\n
        the program will retrieve it from the CDDIS server.

        :return retcode: bool stating if the read was successful or not.
        """
        retcode = True
        # Open Observation file
        try:
            self.dataObs = rx.load(self.dataDir + '/obs/' + self.fileObs)
        except:
            print('ERROR: Failed opening RINEX input files.')
            retcode = False
        
        # Open Navigation file if provided
        if len(self.fileNav) > 0:
            try:
                self.dataNav = rx.load(self.dataDir + '/nav/' + self.fileNav)
            except:
                print('ERROR: Failed opening RINEX input files.')
                retcode = False
        else: # If not provided, try downloading
            try:
                self.fileNav = self._download_nav_data('gz')
                self.dataNav = rx.load(self.fileNav)
            except:
                print('ERROR: Failed opening RINEX input files.')
                retcode = False
            return retcode
        
    def get_rinex_data(self):
        """
        Get read observation and navigation data structures.
        """
        return self.dataObs, self.dataNav



