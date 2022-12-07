import numpy as np
import numpy.matlib
import scipy.optimize as sciopt
import copy

def symDemapper(Srx,Stx,C,DEMAPPER=None):

    if DEMAPPER == None:
        DEMAPPER = {}

    normMethod = 'MMSE'
    use_centroids = False
    decoding = 'normal'
    calc_LLR = True
    M = np.size(C,0)
    nBpS = np.log2(M)
    applySYNC = False
    SYNC = {}
    SYNC['method'] = 'complexField'
    SYNC['debug'] = False
    normalizeTX = True
    getSymsRX = True
    useGPU = False

    nPol = len(Srx)
    if np.size(C,1) == 1:
        C = np.matlib.repmat(C,1,nPol)

    for n in range(0,nPol):
        # Emular o a funcao uniquetol() do Matlab:
        C_u = uniquetol(C[:,n].real, 1e-6) + 1j * uniquetol(C[:,n].imag, 1e-6)
        Stx_u = uniquetol(Stx[n,:].real, 1e-6) + 1j * uniquetol(Stx[n,:].imag, 1e-6)

        if np.size(C_u) != np.size(Stx_u):
            Stx_u = Stx_u[abs(Stx_u) == min(abs(Stx_u))]
            C_u = C_u[abs(C_u) == min(abs(C_u))]

        fun = lambda h: np.vdot((h*Stx_u-C_u),np.transpose((h*Stx_u-C_u)))
        scaleFactor = sciopt.fmin(func = fun, x0 = 1)
        Stx[n,:] = Stx[n,:] * scaleFactor

    txSyms = signal2symbol(Stx,C,[],useGPU)

    if normMethod == 'avgPower':
        c = np.sqrt(np.mean(abs(Srx)**2, axis=1) / np.mean(abs(Stx)**2, axis = 1))
    elif normMethod == 'MMSE':
        for n in range(0,nPol):
            c = np.zeros(nPol)
            fun = lambda h: np.vdot((h*Stx[n,:]-Srx[n,:]), (np.transpose((h*Stx[n,:]-Srx[n,:])))) #USAR 'np.vdot', e nao 'np.dot'!!!
            c[n]=sciopt.fmin(func = fun, x0 = 1)
    else:
        c = np.ones(nPol)

    for n in range(0, nPol):
        Stx[n,:] = Stx[n,:] * c[n]
        C[:,n] = C[:,n] * c[n]

    # Demap Rx Symbols
    if getSymsRX:
        rxSyms = signal2symbol(Srx,C,[],useGPU)

    # Calculate Noise Variance
    N0,lixo =  getN0_MMSE(Stx,Srx)

    Prx = np.var(np.transpose(Srx))
    N0 = N0 * Prx
    SNR_dB = pow2db(Prx) - pow2db(N0)

    # Calculate LLRs
    if calc_LLR:
        for n in range(0,nPol):
            LLRs = np.empty((0,nPol))
            # Calculate Symbol Probability:
            symProb = np.histogram(txSyms[n,:], bins=np.arange(-0.5,(M+1)-0.5,1), density=True)[0]
            LLRs = np.concatenate((LLRs, LLR_eval(Srx[n,:], N0[n], C[:,n], symProb)))

    # Transmitted and Received Bits
    # Nao esta implementada descodificaçao diferencial
    if not np.remainder(nBpS,1):
        DEMAPPER['txBits'] = sym2bit(txSyms, nBpS)
        if getSymsRX:
            DEMAPPER['rxBits'] = sym2bit(rxSyms, nBpS)
    else:
        DEMAPPER['txBits'] = np.NaN
        DEMAPPER['rxBits'] = np.NaN

    # Remove NaNs from LLRs (may happen when SNR is too high)
    #if calc_LLR:
    #if np.any(np.isnan(LLRs)):
    #LLRs[np.isnan(LLRs)] = 0

    # Output Demapper Struct
    DEMAPPER['SYNC'] = SYNC
    DEMAPPER['txSyms'] = txSyms
    if getSymsRX:
        DEMAPPER['rxSyms'] = rxSyms
    DEMAPPER['C'] = C
    DEMAPPER['scaleFactor'] = c
    if calc_LLR:
        DEMAPPER['LLRs'] = LLRs
    DEMAPPER['N0'] = N0
    return DEMAPPER,Stx


def signal2symbol(sig, C, normPower, useGPU):

    nPol, nSyms = sig.shape
    if np.size(C,1) == 1:
        C = np.matlib.repmat(C,1,nPol)

    # Normalizacao nao é feita aqui.

    # Symbol Decoder
    syms = np.zeros((nPol, nSyms)) - 1
    for n in range(0,nPol):
        thisC = C[:,n]
        thisC=np.vstack(thisC)
        thisSig = sig[n,:]
        err = abs(thisSig - thisC)
        ind = np.argmin(err,axis=0)

        syms[n,:] = ind

    return syms


def uniquetol(a, tol):

    if type(a) != list:
        a=list(a)

    tol = max(map(lambda x: abs(x), a)) * 0.3
    a.sort()
    results = [a.pop(0), ]

    for i in a:
        # Skip items within tolerance.
        if abs(results[-1] - i) <= tol:
            continue
        results.append(i)
    results = np.asarray(results)

    return results

def getN0_MMSE(Stx,Srx):
    # Input Parameters
    nPol = len(Stx)

    # Calculate Added Noise over Srx
    c = np.empty(nPol)
    N0 = np.empty(nPol)
    c[:]=np.NaN
    N0[:]=np.NaN

    Stx_copy = copy.deepcopy(Stx)
    Srx_copy = copy.deepcopy(Srx)

    for n in range(0, nPol):
        Stx_copy[n,:] = Stx_copy[n,:]/np.sqrt(np.mean(abs(Stx_copy[n,:])**2))
        Srx_copy[n,:] = Srx_copy[n,:]/np.sqrt(np.mean(abs(Srx_copy[n,:])**2))
        fun = lambda h: np.vdot((h*Stx_copy[n,:]-Srx_copy[n,:]), (np.transpose(h*Stx_copy[n,:]-Srx_copy[n,:])))
        c[n] = sciopt.fmin(func = fun, x0 = 1, xtol = 1e-6, ftol = 1e-6, maxiter = 1e3) # Por alguma razao difere alguns milésimos do matlab...
        N0[n] = (1-(c[n]**2))/(c[n]**2)

    return N0, c

def pow2db(y):
    ydB = (10*np.log10(y)+300)-300

    return ydB

def LLR_eval(Srx,N0,C,symProb):
    nSyms = np.size(Srx,0)
    M = len(C)
    nBpS = np.log2(M)

    bitMap = np.zeros((M,int(nBpS)))
    for n in range (0, M):
        b = bin(n)[2:].zfill(int(nBpS))
        bitMap[n] = np.fromiter(b, (str,int(nBpS)))

    # Calculate LLRs
    LLRs = np.zeros((int(nSyms*nBpS),1))
    LLRs[:] = np.NaN
    for n in range(0,int(nBpS)):
        # Get Subsets Xk_b and Pk_b:
        idx = bitMap[:,n] == 0
        Xk_0 = C[idx]
        Pk_0 = symProb[idx]
        idx = bitMap[:,n] == 1
        Xk_1 = C[idx]
        Pk_1 = symProb[idx]

        # Calculate Numerator and Demominator of the Likelihood Ratio:
        A = np.zeros(nSyms)
        for k in range(0, len(Xk_0)):
            A = A + np.exp((-((abs(Srx - Xk_0[k]))**2)/N0)) * Pk_0[k]
        B = np.zeros(nSyms)
        for k in range(0, len(Xk_1)):
            B = B + np.exp((-((abs(Srx - Xk_1[k]))**2)/N0)) * Pk_1[k]

        # Calculate the Log-Likelihood Ratio:
        LLRs[n::int(nBpS),0] = -np.log(B/A)[:]

    return LLRs

def sym2bit(syms, nBpS):
    # Input Params
    nPol, nSyms = syms.shape
    M = 2**nBpS

    # Transform Symbols into Bits
    bits = np.zeros((int(nPol),int(np.log2(M)*nSyms)))
    bin_vector = np.vectorize(np.binary_repr)
    string_bits = bin_vector(syms.astype(int),int(nBpS))
    for m in range(0,nPol):
        bits[m] = np.asarray(list(map(int, ''.join(string_bits[m]))))

    # Codigo abaixo muito pouco efeciente (demora muito)
    #for m in range(0,nPol):
    #arr=[]
    #for word in string_bits[m]:
    #b=np.frombuffer(word.encode("ascii"), dtype="u1") - 48
    #arr=np.concatenate((arr, b))
    #bits[m] = arr[:]

    return bits