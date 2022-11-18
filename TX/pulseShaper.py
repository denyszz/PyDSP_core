from scipy.signal import upfirdn,convolve
import numpy as np

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