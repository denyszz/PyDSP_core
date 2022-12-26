from numpy.fft import fft, ifft, fftshift
import numpy as np

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