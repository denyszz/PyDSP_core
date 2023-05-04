from numpy.fft import fft, ifft, fftshift
import numpy as np
import decimal

from PyDSP_core.TX.pulseShaper import rcosdesign

def LPF_apply(S,LPF,Fs,Rs):
    # Input Parser
    if 'fc' in LPF:
        fc = LPF['fc']
    elif 'fc' not in LPF and 'fcn' in LPF:
        fc = LPF['fcn']*Fs
        LPF['fc'] = fc
    if 'f0' not in LPF:
        LPF['f0'] = 0

    # Input Parameters
    nPol, nSamples = S.shape

    # Rx Signal Filtering
    if LPF['type'] == 'RC' or LPF['type'] == 'raised-cosine':
        f = np.arange(-nSamples/2,nSamples/2) * (Fs/nSamples)
        TF = RC_transferFunction(f,Rs,LPF['rollOff'])

        for n in range(0,nPol):
            S[n,:] = ifft(fftshift(TF) * fft(S[n,:]))

    elif LPF['type'] == 'Gaussian':
        TF = superGaussian_TF(fc,LPF['order'],LPF['f0'],Fs,nSamples)[0]

        for n in range(0,nPol):
            S[n,:] = ifft(fftshift(TF) * fft(S[n,:]))

    elif LPF['type'] == 'RRC' or LPF['type'] == 'root-raised-cosine':
        nSpS_in = Fs/Rs
        nSpS = int(decimal.Decimal(nSpS_in).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        # Resample nao implementado
        if 'nTaps' in LPF:
            nTaps = LPF['nTaps']
        else:
            nTaps = 64 * nSpS
        if 'implementation' in LPF:
            implementation = LPF['implementation']
        else:
            implementation = 'FFT'
        W = rcosdesign(LPF['rollOff'], nTaps/nSpS, nSpS, 'sqrt')
        W = W/sum(W)

        if implementation == 'FFT':
            nSamples = np.size(S,1)
            zeroEnd = False
            if np.remainder(nSamples,2):
                S = np.concatenate((S, np.zeros((nPol,1))))
                nSamples = nSamples + 1
                zeroEnd = True
            W_f = np.concatenate((np.zeros(int((nSamples-nTaps)/2)), W, np.zeros(int((nSamples-nTaps)/2-1))))

            if implementation == 'FFT':
                for n in range(0,nPol):
                    S[n,:] = fftshift(ifft(fft(S[n,:]) * fft(W_f)))
            TF = fft(W)

    return S, LPF

def RC_transferFunction(f,Rs,a):
    # Calculate Passing and Decaying Bands
    passBand = abs(f) <= (1-a)*Rs/2
    decayBand = np.logical_and(abs(f) > (1-a)*Rs/2, abs(f) <= (1+a)*Rs/2)

    # Apply RC Transfer Function
    TF = np.zeros(len(f))
    TF[passBand] = 1
    TF[decayBand] = 0.5*(1+np.cos(np.pi/(a*Rs)*(abs(f[decayBand])-(1-a)*Rs/2)))

    return TF

def superGaussian_TF(fc,order,f0,Fs,nSamples):
    # Input Parameters
    BW = np.floor(fc*nSamples/Fs)
    f0 = f0/Fs*nSamples
    f = np.linspace(-nSamples/2,nSamples/2,num=nSamples)
    fn = (f-f0)/BW

    # Generate Super-Gaussian Transfer Function
    TF = np.exp(-np.log(np.sqrt(2))*fn**(2*order))

    return TF,fn