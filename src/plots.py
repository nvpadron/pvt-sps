import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import use
from manager import manager, HelpersDataframe
from utils import Constants, DTypes
from third.plotWindow import plotWindow

class Plots:

    def make_plots():
        pwin = plotWindow()
        plt.ion()

        f,fn = Plots.enu_stats()
        plt.pause(1)
        for ii in reversed(range(0,len(f))):
            pwin.addPlot(fn[ii], f[ii])
            plt.savefig('imgs/' + manager.config.imgsdir + '/' + fn[ii] + '.png', bbox_inches='tight')
            plt.close()

        f, fn = Plots.enu()
        plt.pause(1)
        for ii in reversed(range(0,len(f))):
            pwin.addPlot(fn[ii], f[ii])
            plt.savefig('imgs/' + manager.config.imgsdir + '/' + fn[ii] + '.png', bbox_inches='tight')
            plt.close()

        f,fn = Plots.dops()
        plt.pause(1)
        for ii in reversed(range(0,len(f))):
            plt.show()
            pwin.addPlot(fn[ii], f[ii])
            plt.savefig('imgs/' + manager.config.imgsdir + '/' + fn[ii] + '.png', bbox_inches='tight')
            plt.close()

        f, fn = Plots.meas()
        plt.pause(1)
        for ii in reversed(range(0,len(f))):
            plt.show()
            pwin.addPlot(fn[ii], f[ii])
            plt.savefig('imgs/' + manager.config.imgsdir + '/' + fn[ii] + '.png', bbox_inches='tight')
            plt.close()

        f,fn = Plots.skyplot()
        plt.pause(1)
        for ii in reversed(range(0,len(f))):
            plt.show()
            pwin.addPlot(fn[ii], f[ii])
            plt.savefig('imgs/' + manager.config.imgsdir + '/' + fn[ii] + '.png', bbox_inches='tight')
            plt.close()

        f,fn = Plots.skyplot(plot_single=False)
        plt.pause(1)
        for ii in reversed(range(0,len(f))):
            plt.show()
            pwin.addPlot(fn[ii], f[ii])
            plt.savefig('imgs/' + manager.config.imgsdir + '/' + fn[ii] + '.png', bbox_inches='tight')
            plt.close()

        return pwin
        
    def meas():

        # Get SV visible during more time
        tIx = len(manager.results.rxResultHandler.pvt[0])
        t = manager.results.rxResultHandler.pvt[0]['tsow']
        t = t - t[0]
        dataSys = manager.processing.meas.get_data(Constants.GPS_INDICATOR)

        prnVisibleCount = dataSys.visible[:tIx,:].sum(axis = 0)
        prnIxMostVisible = np.argmax(prnVisibleCount)
        tIxVisible = np.arange(tIx) * dataSys.visible[:tIx,prnIxMostVisible].astype('uint8')
        sv = manager.processing.meas.sv
        prnsVisible = sv[sv[:,1] == Constants.GPS_INDICATOR,0][prnVisibleCount > 0]
        # Get data to plot
        cn0 = dataSys.cn0[tIxVisible,:]
        pseudo = dataSys.pseudorange[tIxVisible,:] * 1e-3
        elev = np.rad2deg(dataSys.elevation[tIxVisible,:])

        # Plot all CN0 vs elevation
        f = {}; fn = {}
        f[0], ax = plt.subplots(2,1)
        fn[0] = 'All CN0'
        ax[0].plot(elev, cn0)
        ax[0].grid()
        ax[0].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[0].set_title('CN0 vs elevation')
        ax[0].set_xlabel('Elevation [degrees]')
        ax[1].set_ylabel('CN0 [dB/Hz]')
        ax[1].plot(t, cn0)
        ax[1].grid()
        ax[1].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[1].set_title('CN0 vs time')
        ax[1].set_xlabel('Time [seconds]')
        ax[1].set_ylabel('CN0 [dB/Hz]')
        f[0].tight_layout()

        # Plot 1 SV CN0 vs elevation
        f[1], ax = plt.subplots(2,1)
        fn[1] = 'Single CN0'
        ax[0].plot(elev[:,prnIxMostVisible], cn0[:,prnIxMostVisible])
        ax[0].grid()
        ax[0].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[0].set_title('CN0 vs elevation for most visible SV')
        ax[0].set_xlabel('Elevation [degrees]')
        ax[1].set_ylabel('CN0 [dB/Hz]')
        ax[1].plot(t, cn0[:,prnIxMostVisible])
        ax[1].grid()
        ax[1].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[1].set_title('CN0 vs time for most visible SV')
        ax[1].set_xlabel('Time [seconds]')
        ax[1].set_ylabel('CN0 [dB/Hz]')
        f[1].tight_layout() 

        # Plot all pseudoranges vs elevation
        f[2], ax = plt.subplots(2,1)
        fn[2] = 'All Pseudoranges'
        ax[0].plot(elev, pseudo)
        ax[0].grid()
        ax[0].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[0].set_title('pseudoranges vs elevation')
        ax[0].set_xlabel('Elevation [degrees]')
        ax[1].set_ylabel('Pseudoranges [Km]')
        ax[1].plot(t, pseudo)
        ax[1].grid()
        ax[1].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[1].set_title('pseudoranges vs time')
        ax[1].set_xlabel('Time [seconds]')
        ax[1].set_ylabel('Pseudoranges [Km]')
        f[2].tight_layout()    

        # Plot 1 SV CN0 vs elevation
        fn[3] = 'Single Pseudorange'
        f[3], ax = plt.subplots(2,1)
        ax[0].plot(elev[:,prnIxMostVisible], pseudo[:,prnIxMostVisible])
        ax[0].grid()
        ax[0].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[0].set_title('pseudoranges vs elevation for most visible SV')
        ax[0].set_xlabel('Elevation [degrees]')
        ax[1].set_ylabel('Pseudoranges [Km]')
        ax[1].plot(t, pseudo[:,prnIxMostVisible])
        ax[1].grid()
        ax[1].legend(['sv' + str(prn) for prn in prnsVisible], loc="lower left", ncol = 8, mode="expand")
        ax[1].set_title('Pseudoranges vs time for most visible SV')
        ax[1].set_xlabel('Time [seconds]')
        ax[1].set_ylabel('Pseudoranges [Km]')
        f[3].tight_layout()
        plt.show()
        return f,fn
    
    def enu():
        f = {}; fn = {}
        f[0] = plt.figure()
        fn[0] = 'ENU'

        lgnd = []
        for mode in manager.config.modes:
            idxhdop = Plots.getIdxhdop(mode.idx)
            plt.scatter(np.array(manager.results.rxResultHandler.pvt[mode.idx]['n'][idxhdop], dtype = DTypes.FLOAT),
                        np.array(manager.results.rxResultHandler.pvt[mode.idx]['e'][idxhdop], dtype = DTypes.FLOAT))
            lgnd.append(mode.name)

        plt.grid()    
        plt.legend(lgnd)
        plt.title('ENU')
        plt.xlabel('North [m]')
        plt.ylabel('East [m]')
        
        f[0].tight_layout()
        plt.show()
        return f,fn

    def enu_stats():
        f = {}; fn = {}
        f[0] = plt.figure()
        fn[0] = 'ENU Statistics'
        
        gs = GridSpec(2,3,f[0])

        t = manager.results.rxResultHandler.pvt[0]['tsow']
        t = t - t[0]

        # Create Axes
        f[0].add_subplot(gs[0])
        f[0].add_subplot(gs[0,1:])
        f[0].add_subplot(gs[1,0])
        f[0].add_subplot(gs[1,1:])
        ax = f[0].axes
        lgnd = []
        for mode in manager.config.modes:
            idxhdop = Plots.getIdxhdop(mode.idx)
            northError = np.array(manager.results.rxResultHandler.pvt[mode.idx]['n'][idxhdop], dtype = DTypes.FLOAT)
            eastError = np.array(manager.results.rxResultHandler.pvt[mode.idx]['e'][idxhdop], dtype = DTypes.FLOAT)
            ax[0].hist(northError)
            ax[1].plot(t[idxhdop], northError,'.')
            ax[2].hist(eastError)
            ax[3].plot(t[idxhdop], eastError,'.')
            lgnd.append(mode.name)
        
        ax[0].grid()
        ax[0].legend(lgnd)
        ax[0].set_title('Histogram North error')
        ax[0].set_xlabel('North Error [m]')
        
        ax[1].grid()
        ax[1].legend(lgnd)
        ax[1].set_title('Plot North error over time')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('North Error [m]')
        
        ax[2].grid()
        ax[2].legend(lgnd)
        ax[2].set_title('Histogram East error')
        ax[2].set_xlabel('Easth Error [m]')
        
        ax[3].grid()
        ax[3].legend(lgnd)
        ax[3].set_title('Plot East error over time')
        ax[3].set_xlabel('Time [s]')
        ax[3].set_ylabel('Easth Error [m]')

        f[0].tight_layout()
        plt.show()
        return f,fn

    def dops():
        f = {}; fn = {}
        f[0] = plt.figure()
        fn[0] = 'DOP'
        gs = GridSpec(3,1,f[0])

        t = manager.results.rxResultHandler.pvt[0]['tsow']
        t = t - t[0]

        # Create Axes
        f[0].add_subplot(gs[0])
        f[0].add_subplot(gs[1])
        f[0].add_subplot(gs[2])
        ax = f[0].axes
        lgnd = []
        for mode in manager.config.modes:
            idxhdop = Plots.getIdxhdop(mode.idx)
            hdop = np.array(manager.results.rxResultHandler.pvt[mode.idx]['hdop'][idxhdop], dtype = DTypes.FLOAT)
            vdop = np.array(manager.results.rxResultHandler.pvt[mode.idx]['vdop'][idxhdop], dtype = DTypes.FLOAT)
            resnorm = np.array(manager.results.rxResultHandler.pvt[mode.idx]['resnorm'][idxhdop], dtype = DTypes.FLOAT)
            ax[0].plot(t[idxhdop], hdop,'.')
            ax[1].plot(t[idxhdop], vdop,'.')
            ax[2].plot(t[idxhdop], resnorm,'.')
            lgnd.append(mode.name)
        
        ax[0].grid()
        ax[0].legend(lgnd)
        ax[0].set_title('HDOP')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('meters')

        ax[1].grid()
        ax[1].legend(lgnd)
        ax[1].set_title('VDOP')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('meters')

        ax[2].grid()
        ax[2].legend(lgnd)
        ax[2].set_title('Norm of Pseudorange residual')
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('meters')

        f[0].tight_layout()
        plt.show()
        return f,fn
    
    

    def skyplot(plot_single = True):
    
        # Get SV visible during more time
        tIx = len(manager.results.rxResultHandler.pvt[0])
        t = manager.results.rxResultHandler.pvt[0]['tsow']
        t = t - t[0]
        dataSys = manager.processing.meas.get_data(Constants.GPS_INDICATOR)

        elev = np.rad2deg(dataSys.elevation)
        azim = np.rad2deg(dataSys.azimuth) + 180

        tIxVisible = np.arange(tIx)
        sv = manager.processing.meas.sv

        if plot_single:
            ix = int(np.median(tIxVisible))
            prnVisibleCount = dataSys.visible[ix,:]
            prnsVisible = sv[sv[:,1] == Constants.GPS_INDICATOR,0][prnVisibleCount]
            elevation = elev[ix,prnVisibleCount]
            azimuth = azim[ix,prnVisibleCount]
        else:
            ix = tIxVisible
            prnVisibleCount = dataSys.visible[ix,:].sum(axis = 0) > 0
            prnsVisible = sv[sv[:,1] == Constants.GPS_INDICATOR,0][prnVisibleCount]
            elevation = elev[ix,:][:,prnVisibleCount]
            azimuth = azim[ix,:][:,prnVisibleCount]

        # Create a polar plot
        f = {}; fn = {}
        f[0] = plt.figure()
        if plot_single:
            fn[0] = 'Skyplot fix time'
        else:
            fn[0] = 'Skyplot all time'
        ax = f[0].add_axes([0.1,0.1,0.8,0.8],polar=True)

        # Plot the elevation vs azimuth
        for el, az in  zip(elevation.T,azimuth.T):
            ax.plot(np.deg2rad(az), 90 - el, marker='o', markersize = 10)
        ax.legend(['sv' + str(prn) for prn in prnsVisible])

        # Set the direction of North to be at the top
        ax.set_theta_zero_location('N')

        # Set the direction of azimuth to be clockwise
        ax.set_theta_direction(-1)

        # Set the title and labels
        plt.title('Skyplot', va='bottom')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')

        ax.set_yticks(range(0, 90+10, 10))                   # Define the yticks
        yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
        ax.set_yticklabels(yLabel)
        plt.show()
        return f,fn
    
    
    def getIdxhdop(modeidx):
            meanhdop = np.nanmean(np.array(manager.results.rxResultHandler.pvt[modeidx]['hdop']).astype(DTypes.FLOAT))
            stdhdop = np.nanstd(np.array(manager.results.rxResultHandler.pvt[modeidx]['hdop']).astype(DTypes.FLOAT))
            if manager.config.plotsExcludeDOP:
                idxhdop = np.array(manager.results.rxResultHandler.pvt[modeidx]['hdop']).astype(DTypes.FLOAT) < (meanhdop + 0.5 * stdhdop)
            else:
                idxhdop = np.ones(manager.currentTimeEpoch.t_ix + 1, dtype=DTypes.BOOL)
            return idxhdop

    def showStats():
        # Prepare Stats table
        for mode in manager.config.modes:
            idxhdop = Plots.getIdxhdop(mode.idx)
            # Calculate STDs and RMSs on ENU data
            data = np.array(manager.results.rxResultHandler.pvt[mode.idx][['e','n','u']][idxhdop], dtype = 'float')
            idx = ~np.isnan(data[:,0])
            mean_enu = data[idx,:].mean(axis=0)
            std_enu = np.array(data[idx,:], dtype = 'float').std(axis = 0)
            rms_sq_enu = mean_enu**2 + std_enu**2
            rms_2d = np.sqrt(rms_sq_enu[:2].sum())
            rms_3d = np.sqrt(rms_sq_enu.sum())
            cep50 = rms_2d / 1.2
            # Calculate average HDOP and VDOP
            data = np.array(manager.results.rxResultHandler.pvt[mode.idx][['hdop','vdop']][idxhdop], dtype = 'float')
            idx = ~np.isnan(data[:,0])
            avg_dops = data[idx,:].mean(axis=0)
            datainfo = np.hstack([std_enu, np.sqrt(rms_sq_enu), rms_2d, rms_3d, cep50, avg_dops[0], avg_dops[1]])
            # Add info into stats dataframe
            HelpersDataframe.df_add(manager.results.rxResultHandler.stats, datainfo, manager.results.rxResultHandler.stats.columns, mode.name, True)

        # Show stats
        print(manager.results.rxResultHandler.stats)

        manager.results.rxResultHandler.stats.astype('float').to_csv('imgs/' + manager.config.imgsdir + '/stats.csv', sep=';', decimal=',')
        print('Stats table stored in: ' + 'imgs/' + manager.config.imgsdir + '/stats.csv')
        
        
    