from scipy.signal import upfirdn,convolve
from numpy.fft import fft, ifft, fftshift
import numpy as np
import decimal


def pulseShaper(Sin,nSpS,PS,Fs=None):
    # Input Parser
    [nPol, nSamples] = Sin.shape
    ####################################################################################################
    Sout = np.zeros((nPol,nSamples*2), dtype='complex_') #Rever isto por causa do tamanho do array
    ####################################################################################################

    # Select Pulse Shaping Filter
    if PS['type'] in ['Rect','rect','rectangular','none']:
        for n in range(0,nPol):
            Sout[n] = rectpulse(Sin[n],nSpS)

    elif PS['type'] in ['RC','raised-cosine','raisedCos','Nyquist']:
        a = PS['rollOff']
        if 'nTaps' in PS:
            nTaps = PS['nTaps']
        else:
            nTaps = 64 * nSpS

        k = np.arange(-np.floor(nTaps/2), np.ceil(nTaps/2))
        tK = k/nSpS
        W = np.sinc(tK) * np.cos(a*np.pi*tK) / (1-4*pow(a,2)*pow(tK,2))
        # W[np.isinf(W)] = 0 (it can used only when symbol period is included in the pulse shaper impulse response)
        np.where(tK==(1/(-2*a)), W, (np.pi/4)*np.sinc(-1/(2*a)))
        np.where(tK==(1/(2*a)), W, (np.pi/4)*np.sinc(1/(2*a)))

        if 'onlyImpulse' in PS and PS['onlyImpulse'] == True:
            for n in range(0,nPol):
                Sout[n] = np.convolve(Sin[n],W)
        else:
            for n in range(0,nPol):
                #Sout[n] = np.convolve(np.pad(upfirdn([1],Sin[n],nSpS),(0,nSpS-1)), W, mode='same')

                # !!! When the length of the shorter input array is even, there
                # is an ambiguity in how it should be handled when the method is "same".
                # Apparently Matlab and numpy adopted different conventions... !!!

                # Alternativa: https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
                npad = len(W) - 1
                u_padded = np.pad(np.pad(upfirdn([1],Sin[n],nSpS),(0,nSpS-1)), (npad//2, npad - npad//2), mode='constant')
                Sout[n]=np.convolve(u_padded, W, mode='valid')
        PS['W'] = W

    elif PS['type'] in ['RRC','root-raised-cosine']:

        resampleFlag = abs(decimal.Decimal(nSpS).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP) - nSpS) > 1e-3
        if resampleFlag:
            nSpS_in = nSpS
            nSpS = np.ceil(nSpS)

        if 'nTaps' in PS:
            nTaps = PS['nTaps']
        else:
            nTaps = 256*nSpS

        implementation = 'FFT'  # apenas realiza este metodo
        W = rcosdesign(PS['rollOff'], nTaps/nSpS, nSpS, 'sqrt')
        W = W/sum(W)

        if implementation == 'FFT':
            zeroEnd = False
            if np.remainder(nSamples,2):
                Sin = np.concatenate((Sin, np.zeros((nPol,1))))
                nSamples = nSamples + 1
                zeroEnd = True
            W_f = np.concatenate((np.zeros(int((nSpS*nSamples-nTaps)/2)), W, np.zeros(int((nSpS*nSamples-nTaps)/2-1))))
        if implementation == 'FFT':
            for n in range(0,nPol):
                X = np.pad(upfirdn([1],Sin[n,:],nSpS),(0,nSpS-1))
                Sout[n,:] = fftshift(ifft(fft(X) * fft(W_f)))
            if zeroEnd:
                Sout = Sout[:,0:-nSpS]
        ### Resample nao implementado ###
        # if resampleFlag:

        PS['W'] = W

    return Sout, PS

def rectpulse(x, Nsamp):

    try:
        shape = (x.shape[0], x.shape[1])
    except IndexError:
        shape = (1, x.shape[0])

    wid = shape[0]
    len = shape[1]
    #print(wid)
    #print(len)

    if wid == 1 and len != 1:
        y = np.reshape(np.dot(np.ones((Nsamp,1)), np.reshape(x,(1,wid*len),order='F')),(wid,len*Nsamp),order='F') #order='F' >> Fortran-like index ordering"
    else:
        y = np.reshape(np.dot(np.ones((Nsamp,1)), np.reshape(x,(1,wid*len),order='F')),(wid*Nsamp,len),order='F')
    #print(y)
    return y

def rcosdesign(beta: float, span: float, sps: float, shape='normal'):
    """ Raised cosine FIR filter design
    Calculates square root raised cosine FIR
    filter coefficients with a rolloff factor of `beta`. The filter is
    truncated to `span` symbols and each symbol is represented by `sps`
    samples. rcosdesign designs a symmetric filter. Therefore, the filter
    order, which is `sps*span`, must be even. The filter energy is one.
    Keyword arguments:
    beta  -- rolloff factor of the filter (0 <= beta <= 1)
    span  -- number of symbols that the filter spans
    sps   -- number of samples per symbol
    shape -- `normal` to design a normal raised cosine FIR filter or
             `sqrt` to design a sqrt root raised cosine filter
    """

    if beta < 0 or beta > 1:
        raise ValueError("parameter beta must be float between 0 and 1, got {}"
                         .format(beta))

    if span < 0:
        raise ValueError("parameter span must be positive, got {}"
                         .format(span))

    if sps < 0:
        raise ValueError("parameter sps must be positive, got {}".format(span))

    if ((sps*span) % 2) == 1:
        raise ValueError("rcosdesign:OddFilterOrder {}, {}".format(sps, span))

    if shape != 'normal' and shape != 'sqrt':
        raise ValueError("parameter shape must be either 'normal' or 'sqrt'")

    eps = np.finfo(float).eps

    # design the raised cosine filter

    delay = span*sps/2
    t = np.arange(-delay, delay)

    if len(t) % 2 == 0:
        t = np.concatenate([t, [delay]])
    t = t / sps
    b = np.empty(len(t))

    if shape == 'normal':
        # design normal raised cosine filter

        # find non-zero denominator
        denom = (1-np.power(2*beta*t, 2))
        idx1 = np.nonzero(np.fabs(denom) > np.sqrt(eps))[0]

        # calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1])*(np.cos(np.pi*beta*t[idx1])/denom[idx1])/sps

        # fill in the zeros denominator indices
        idx2 = np.arange(len(t))
        idx2 = np.delete(idx2, idx1)

        b[idx2] = beta * np.sin(np.pi/(2*beta)) / (2*sps)

    else:
        # design a square root raised cosine filter

        # find mid-point
        idx1 = np.nonzero(t == 0)[0]
        if len(idx1) > 0:
            b[idx1] = -1 / (np.pi*sps) * (np.pi * (beta-1) - 4*beta)

        # find non-zero denominator indices
        idx2 = np.nonzero(np.fabs(np.fabs(4*beta*t) - 1) < np.sqrt(eps))[0]
        if idx2.size > 0:
            b[idx2] = 1 / (2*np.pi*sps) * (
                    np.pi * (beta+1) * np.sin(np.pi * (beta+1) / (4*beta))
                    - 4*beta           * np.sin(np.pi * (beta-1) / (4*beta))
                    + np.pi*(beta-1)   * np.cos(np.pi * (beta-1) / (4*beta))
            )

        # fill in the zeros denominator indices
        ind = np.arange(len(t))
        idx = np.unique(np.concatenate([idx1, idx2]))
        ind = np.delete(ind, idx)
        nind = t[ind]

        b[ind] = -4*beta/sps * (np.cos((1+beta)*np.pi*nind) +
                                np.sin((1-beta)*np.pi*nind) / (4*beta*nind)) / (
                         np.pi * (np.power(4*beta*nind, 2) - 1))

    # normalize filter energy
    b = b / np.sqrt(np.sum(np.power(b, 2)))
    return b