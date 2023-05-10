import numpy as np
from numpy.fft import fft, fftshift
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy

def freqEstimation_FD(Srx, Fs, FE):

    # input parser
    nPol, nSamples = Srx.shape
    if 'blockSize' not in FE or FE['blockSize'] == np.inf:
        FE['blockSize'] = nSamples
    if 'blockJump' not in FE:
        FE['blockJump'] = FE['blockSize']
    if 'zeroPadFactor' not in FE:
        FE['zeroPadFactor'] = 1
    if 'demodQAM' not in FE:
        FE['demodQAM'] = '4thPower'
    if 'jointPol' not in FE:
        FE['jointPol'] = False
    if 'nTaps' not in FE:
        FE['nTaps'] = 1

    # Input Parameters
    K = FE['zeroPadFactor']     #zero-pad upsampling factor
    blockSize = FE['blockSize'] #FE block-size
    fRes = Fs/(blockSize*K)     #frequency resolution [Hz]
    f = np.arange(-blockSize*K/2, blockSize*K/2)*fRes

    #Demod QAM
    if FE['demodQAM'] == '4thPower':
        Ademod = Srx**4
        c = 4
    else:
        Ademod = Srx
        c = 1

    # Set Frequency Range for Estimation
    if 'freqRange' not in FE:
        FE['freqRange'] = [min(f/c), max(f/c)]
        maxIdx = nSamples - 1
        minIdx = 0
    else:
        maxIdx = np.argwhere(f/c >= max(FE['freqRange']))[0][0]
        minIdx = np.argwhere(f/c <= min(FE['freqRange']))[-1][0]


    # Apply Frequency Estimation
    fw = FE['blockJump']
    ind_k = np.arange(np.ceil(blockSize/2)-1,nSamples-np.floor(blockSize/2),fw).astype(int)

    Aw = np.zeros((nPol,len(f)))
    df = np.empty((nPol,nSamples))
    df[:] = np.NaN

    for n in range(0,nPol):
        ind0 = np.arange(0,blockSize)
        for k in ind_k:
            df[n,k], Aw[n,:] = findSpectralPeak(Ademod[n,ind0],K,minIdx,maxIdx,fRes,c)
            ind0 = ind0 + fw

    # Interpolate Frequency Estimation
    idx = np.argwhere(~np.isnan(df[0])).flatten()
    df_interp = np.empty((nPol,nSamples))
    df_interp[:] = np.NaN
    if len(idx) == 1:
        for n in range(0,nPol):
            df_interp[n,:] = np.tile(df[n,idx],nSamples)
    else:
        for n in range(0,nPol):
            df_interp[n,:] = interp1d(idx, df[n,idx], kind='nearest', fill_value='extrapolate')(range(0, nSamples))

    # Low-Pass Filtering the Frequency Estimation Vector
    if FE['nTaps']:
        for n in range(0,nPol):
            df_interp[n,:] = np.convolve(df_interp[n,:],np.ones(int(FE['nTaps']),dtype='float'), 'same')/np.convolve(np.ones(len(df_interp[n,:])),np.ones(int(FE['nTaps'])), 'same')

    # Remove Frequency Offset
    t = np.arange(0,nSamples)/Fs
    for n in range(0,nPol):
        Srx[n,:] = Srx[n,:] * np.exp(-1j*2*np.pi*df_interp[n,:]*t)

    # Output Parameters
    FE['df'] = df_interp
    FE['freqResolution'] = fRes/c
    FE['updateRate'] = Fs/fw

    # Debug
    for n in range(0, np.size(Aw,0)):
        plotSpectrum_FE(Aw[n,:],Fs,fRes,df[n,idx[-1]],c,FE['freqRange'])
        plotFreqOverTime_FE(df_interp,Fs,fRes)

    return Srx,FE

def findSpectralPeak(Ademod,K,minIdx,maxIdx,fRes,c):
    try:
        shape = (Ademod.shape[0], Ademod.shape[1])
    except IndexError:
        shape = (1, Ademod.shape[0])
    nPol = shape[0]
    nSamples = shape[1]

    Ademod_copy = copy.deepcopy(Ademod)
    # Apply Zero-Padding:
    if nPol == 1:
        Ademod_copy = np.concatenate((Ademod_copy, np.zeros(nSamples*(K-1))))
        nSamples = len(Ademod_copy)
    else:
        Ademod_copy = np.concatenate((Ademod_copy, np.zeros((nPol,nSamples*(K-1)))))
        nSamples = np.size(Ademod_copy,1)
    Aw = np.zeros((1,nSamples))

    if nPol == 1:
        Aw = Aw[0] + abs(fftshift(fft(Ademod_copy[:])))
    else:
        for n in range(0, nPol):
            Aw = Aw + abs(fftshift(fft(Ademod_copy[n,:])))

    ind = np.argmax(Aw[minIdx:maxIdx+1])
    ind = ind + minIdx
    df = (ind-nSamples/2)*fRes/c

    return df,Aw

def plotSpectrum_FE(Aw,Fs,fRes,df,c=0,freqRange=0):
    # Check units
    Fs, units, unitFactor = setFreqUnits(Fs)
    df_print, unitsDf = setFreqUnits(df)[0:2]
    fRes = fRes*unitFactor
    df = df*unitFactor

    # Plots
    f = np.arange(-Fs/2,Fs/2,fRes)/c
    ind = np.argmin(abs(f-df))
    y = 20*np.log10(Aw/max(Aw))
    plt.plot(f,y,linewidth = 0.8)
    plt.scatter(f[ind],y[ind],s=110,marker='o', edgecolor='r', facecolor=(0,1,0,0))
    plt.grid()
    # Add frequency range limits:
    plt.axvline(x = freqRange[0]*unitFactor, color = 'r', linestyle = 'dashed')
    plt.axvline(x = freqRange[1]*unitFactor, color = 'r', linestyle = 'dashed')
    plt.xlabel(r'$f$ [units]', fontsize=11)
    plt.ylabel('Normalized PSD [dB]', fontsize=11)
    plt.axis([min(f),max(f),min(y),5])
    plt.text(f[-1]-0.02*Fs, y[ind], r'$\Delta f=$' +
             '{:.1f} {}'.format(np.mean(df_print), unitsDf),
             horizontalalignment='right', verticalalignment='top',
             color='red', fontsize=12, backgroundcolor='w')
    plt.show()

def plotFreqOverTime_FE(df,Fs,fRes):

    nPol, nSamples = df.shape
    t = np.arange(0,nSamples)/Fs

    for n in range(0,nPol):
        plt.plot(t*1e6,df[n,:]*1e-6)

        plt.xlabel('$t$ [$\mu$s]', fontsize=11)
        plt.ylabel('Frequency Estimation [MHz]', fontsize=11)

        plt.show()

    # Axis properties

def setFreqUnits(f):
    # Choose Frequency Units
    if abs(f) > 1e12:
        units = 'THz'
        unitFactor = 1e-12
    elif abs(f) > 1e9:
        units = 'GHz'
        unitFactor = 1e-9
    elif abs(f) > 1e6:
        units = 'MHz'
        unitFactor = 1e-6
    elif abs(f) > 1e3:
        units = 'KHz'
        unitFactor = 1e-3
    else:
        units = 'Hz'
        unitFactor = 1

    # Convert Into Frequency Units
    f = f * unitFactor

    return f, units, unitFactor