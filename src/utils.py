# utils.py
import numpy as np
from numba import jit

@jit(nopython = True)
def dot_vec_vec(x,y):
    """
    Vector-vector dot multiplication. \n
    Numba functions to calculate x @ y, since the @ operator is not allowed in Numba.
    
    :param x: Lx1 vector
    :param y: Lx1 vector

    :return: dot product
    """
    return (x * y).sum()

@jit(nopython = True)
def dot_mat_vec(X,y):
    """
    Matrix-vector multiplication. \n
    Numba functions to calculate X @ y, since the @ operator is not allowed in Numba.
    
    :param X: MxL matrix
    :param y: Lx1 vector

    :return: Mx1 array containing product multiplication
    """
    res = np.zeros(X.shape[0])
    for rr in range(X.shape[0]):
        res[rr] = (X[rr,:] * y).sum()
    return res

class SysInfo:
    """
    Class storing system constants to be used in calculation. Currently only GPS L1 supported.
    """
    
    GPS_INDICATOR   = 'G'
    GPS_MAX_SV      = 32
    GPS_L1_FREQ_HZ  = 1575420000
    GPS_L1_WAVELENGTH_M = 0.1905
    GPS_L1_TCOH_MAX_S = 0.02
    GPS_L2_FREQ_HZ  = 1227600000
    GPS_L2_WAVELENGTH_M = 0.2445
    GPS_L2_TCOH_MAX_S = 0.02 # CHECK
    GPS_GAMMA_L1L2  = (77 / 60)**2

class Constants(SysInfo):
    """
    General use constants
    """
    
    speedOfLight    = 299792458         # m/s
    muGravConstant  = 3.986005e14       # m^3/s^2
    forceF          = -4.442807633e-10  # s/m^0.5
    ecc             = 0.08181919084261345
    ecc_sec         = 0.0820944379497174
    omegaEDot       = 7.2921151467e-5   # (r/s)
    semiMajorAxis   = 6378137.0         # m
    semiMinorAxis   = 6356752.3142      # m
    secondsInDay    = 86400             # s
    secondsInWeek   = 604800            # s
    secondsInHalfWeek = 302400          # s
    # Epsilon error boundaries (e.g. x < epsilon, epsilon = 1e-10)
    epsilonErrMag3  = 1e-3
    epsilonErrMag6  = 1e-6
    epsilonErrMag10 = 1e-10
    epsilonErrMag12 = 1e-12
    # Alpha values for different efficiencies in robust estimation
    alpha_efficiency_95 = 4.658

class DTypes:
    BOOL  = 'bool'
    UINT  = 'uint'
    INT   = 'int'
    FLOAT = 'float'
    OBJECT = 'object'

class Rotation:
    """
    Class to handle rotations and convertions between planes. 
    """
    
    def conversion_ecef_2_wgs84(xyz):
        """
        ECEF to WGS84 conversion. \n
        Following Table 2.1 "Determination of Geodetic Height and Latitude in Terms of ECEF Parameters"
        
        :param xyz: ECEF coordinates

        :return llh: WGS84 coordinates
        """ 
        p = np.linalg.norm(xyz[:2],2)
        if p > 0:
            tan_u = xyz[2]/p * Constants.semiMajorAxis / Constants.semiMinorAxis
        else:
            tan_u = 0

        diff_tan_u = 1
        while (np.abs(diff_tan_u) > 1e-12):
            if p > 0:
                cos2u = 1 / (1 + tan_u**2)
            else:
                cos2u = 0
            sin2u = 1 - cos2u
            tan_phi = xyz[2] + Constants.ecc_sec**2 * Constants.semiMinorAxis * np.sqrt(sin2u)**3
            tan_phi /= (p - Constants.ecc**2 * Constants.semiMajorAxis * np.sqrt(cos2u)**3)
            diff_tan_u = tan_u
            tan_u = Constants.semiMinorAxis / Constants.semiMajorAxis * tan_phi
            diff_tan_u -= tan_u; 

        phi = np.arctan(tan_phi)

        N = Constants.semiMajorAxis / np.sqrt(1 - Constants.ecc**2 * np.sin(phi)**2)
        h = 0
        if (int(np.abs(phi)*256) != int(np.pi/2 * 256)):
            h = p / np.cos(phi) - N
        elif (int(abs(phi)*256) != 0):
            h = xyz[2] / np.sin(phi) - N + Constants.ecc**2 * N

        if (int(xyz[0]*256) >= 0):
            lam = np.arctan(xyz[1] / xyz[0])
        elif (int(xyz[0]*256) < 0) and (int(xyz[1]*256) >= 0):
            lam = np.pi + np.arctan(xyz[1] / xyz[0])
        elif (int(xyz[0]*256) < 0) and (int(xyz[1]*256) < 0):
            lam = -np.pi + np.arctan(xyz[1] / xyz[0])

        llh = np.array([phi, lam, h ])

        return llh
    
    def conversion_wgs84_2_ecef(llh):
        """
        WGS84 to ECEF conversion \n
        Follwoing 2.2.3.2 Conversion from Geodetic Coordinates to Cartesian Coordinates in ECEF Frame
        
        :param llh: WGS84 coordinates
        :param xyz: ECEF coordinates
        """

        phi = llh[0]
        lam = llh[1]
        h = llh[2]

        N = Constants.semiMajorAxis
        N /= np.sqrt(1 + (1 - Constants.ecc**2) * np.tan(phi)**2)
        x = np.cos(lam) * N + h * np.cos(lam) * np.cos(phi)
        y = np.sin(lam) * N + h * np.sin(lam) * np.cos(phi)
        z = Constants.semiMajorAxis * (1 - Constants.ecc**2) * np.sin(phi)
        z /= np.sqrt(1 - Constants.ecc**2 * np.sin(phi)**2)
        z += h * np.sin(phi)

        xyz = np.array([x, y, z])
        return xyz

    
    def matrix_wgs84_2_enu(llh):
        """
        Rotation matrix from ECEF to ENU, needs WGS84 coordindates.

        :param llh: WGS854 coordinates

        :return rotmat: 3x3 rotation matrix
        """
        lat_cos = np.cos(llh[0])
        lat_sin = np.sin(llh[0])
        lon_cos = np.cos(llh[1])
        lon_sin = np.sin(llh[1])
        rotmat = np.array([[-lon_sin,            lon_cos,            0      ],\
                           [-lat_sin * lon_cos, -lat_sin * lon_sin,  lat_cos], \
                           [ lat_cos * lon_cos,  lat_cos * lon_sin,  lat_sin]])
        return rotmat


class TimeConverters:
    """
    Utils class to handle time calculations: day of year and second of week.
    """

    def day_of_year(times):
        year = np.array(times).astype('datetime64[Y]').astype(int) + 1970
        firstDayOfYear = np.datetime64(str(year) + '-01-01')
        doysArray = np.unique((np.array(times).astype('datetime64[D]') - firstDayOfYear).astype(int))
        if doysArray.size > 1:
            print('Data measured over day trasition, using 1st day for ephemeris only.')
        doy = doysArray[0] + 1
        return doy, year
    

    def sec_of_week(time, tIx = 0, interval = 0, t_sow0 = None):
        
        if t_sow0 == None:
            # Calculate fraction of current week, and multiply by seconds of week.
            time_ref = np.datetime64('1980-01-06 00:00:00')
            elapsedSeconds = np.array((time.astype('datetime64[ns]') - time_ref)*1e-9).astype('uint')
            elapsedWeeks = elapsedSeconds / Constants.secondsInWeek
            weekFraction = elapsedWeeks - np.floor(elapsedWeeks)
            t_sow = np.round(weekFraction * Constants.secondsInWeek)
        else:
            # Use the interval field in dataObs for extra security to avoid non-unique intervals.
            if interval != 0:
                interval = np.unique(np.diff(t_sow[:100]))[0]

            #t_sow = t_sow[0] + np.arange(0, len(t_sow)) * interval
            t_sow = t_sow0 +  tIx * interval
        return t_sow
    
    

