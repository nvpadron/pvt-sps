import time
import os
import numpy as np

from manager import manager
from utils import SysInfo, DTypes, Rotation
from satellite import SvPVTCalculation
from receiver import RxPVTCalculation
from plots import Plots

# Load RINEX files
manager.load_config()
if not manager.monitor.status:
    exit()
manager.load_data()
if not manager.monitor.status:
    exit()

# Start Satellite and Receiver modules
svModule = SvPVTCalculation(SysInfo.GPS_INDICATOR)
rxModule = RxPVTCalculation(SysInfo.GPS_INDICATOR)
timeStart = time.time()
print('Start processing')

# Loop processing
for manager.currentTimeEpoch in manager.timeEpochs:
    svModule.process()
    rxModule.process()

    manager.interface.show_progress(manager.currentTimeEpoch)
    manager.results.svResultHandler.reset()

timeEnd = time.time()
print('Finish processing, elapsed time = %(dt).2fs' % {'dt' : timeEnd - timeStart} )

# Create directory where to store results
os.mkdir('imgs/' + manager.config.imgsdir)

# Show statistics
print('')
Plots.showStats()
print('')

# Show mean and median XYZ (meaningful in static case).
xyz = np.array(manager.results.rxResultHandler.pvt[0][['x','y','z']]).astype(DTypes.FLOAT)
idx = ~np.isnan(xyz[:,0])
xyz_mean = np.mean(xyz[idx,:],axis=0)
llh_mean = Rotation.conversion_ecef_2_wgs84(xyz_mean)
llh_mean[:2] = np.rad2deg(llh_mean[:2]) 
xyz_median = np.median(xyz[idx,:],axis=0)
llh_median = Rotation.conversion_ecef_2_wgs84(xyz_median)
llh_median[:2] = np.rad2deg(llh_median[:2])

# Retrieve reference
xyz_ref = manager.processing.meas.pvt_reference
llh_ref = Rotation.conversion_ecef_2_wgs84(xyz_ref)
llh_ref[:2] = np.rad2deg(llh_ref[:2])

print('')
print(f'Mean XYZ is (valid in static scenario) = \n ECEF: {xyz_mean}\n WGS84: {llh_mean}')
print(f'Median XYZ is (valid in static scenario) = \n ECEF: {xyz_median}\n WGS84: {llh_median}')
print(f'Reference XYZ is (valid in static scenario) = \n ECEF: {xyz_ref}\n WGS84: {llh_ref}')
print('')

# Plots
pwin = Plots.make_plots()

print('Analysis finished')
input("Press Enter to continue.")