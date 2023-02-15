import numpy as np
from scipy.interpolate import interp1d
from PyDSP_core.TX.pulseShaper import rectpulse
from PyDSP_core.DSP.symDemapper import signal2symbol
from PyDSP_core.TX.Tx_QAM import symbol2signal
from PyDSP_core.DSP.sync import pilotSymbols_rmv

def carrierPhaseEstimation(Srx,Stx,CPE,C=None,nSpS=1):
    # Assumir que o sinal ja tem o donwsampling feito.

    # Check if a Preset Phase Compensation is Available
    if 'phi' in CPE:
        Srx = Srx * np.exp(-1j*rectpulse(CPE['phi'],nSpS))

    # Normalize Constellation to the Signal Power
    if C is not None:
        C = C * np.sqrt(np.mean(abs(Srx[0,:])**2)/np.mean(abs(C)**2))

    # Select and Apply CPE Method
    if CPE['method'] == 'pilot-based':
        Srx, CPE['phi'] = CPE_pilotBased(Srx,Stx,nSpS,CPE)
    elif CPE['method'] == 'BPS' or CPE['method'] == 'blindPhaseSearch':
        Srx, CPE['phi'] = CPE_BPS(Srx,Stx,nSpS,C,CPE)

    return Srx, CPE

def CPE_BPS(Srx,Stx,nSpS,C,CPE):

    # Input parser
    if 'pilotAided' not in CPE:
        pilotAided = False
    else:
        pilotAided = CPE['pilotAided']

    # Input parameters
    nPol = np.size(Srx,0)
    nSamples = np.size(Srx,1)
    B = CPE['nTestPhases']
    phiInt = CPE['angleInterval']
    nTaps = CPE['nTaps']
    applyUnwrap = True
    if 'applyUnwrap' in CPE:
        applyUnwrap = False

    # Apply Blind Phase Search
    Srx_CPE = Srx
    phi = np.zeros((nPol,nSamples))
    d = np.zeros((B+1,nSamples))
    unwrapFactor = 2*np.pi/phiInt

    for n in range(0, nPol):
        for b in range(0, B+1):
            phiTest = (b/B-1/2)*phiInt
            Srot = Srx_CPE[n,:]*np.exp(1j*phiTest)
            Sref = symbol2signal(signal2symbol(np.asarray([Srot]),C,[],False),C)[0]
            if pilotAided:
                Sref[CPE['pilotsIdx']] = Stx[CPE['pilotsIdx']]
            err = abs(Srot-Sref)**2
            d[b,:] = np.convolve(err,np.ones(nTaps,dtype='float'), 'same')/np.convolve(np.ones(len(err)),np.ones(nTaps), 'same')
        ind = np.argmin(d, axis = 0)
        phi[n,:] = -((ind)/B-1/2)*phiInt
        if applyUnwrap:
            phi[n,:] = np.unwrap(phi[n,:]*unwrapFactor)/unwrapFactor

    # Correct Carrier Phase
    Srx = Srx*np.exp(-1j*rectpulse(phi,nSpS))

    return Srx, phi

def CPE_pilotBased(Srx,Stx,nSpS,CPE):

    # Input Parameters
    nPol = np.size(Srx,0)
    nSamples = np.size(Srx,1)
    nTaps = CPE['nTaps']
    PIL_rate = CPE['PILOTS']['rate']
    if PIL_rate == 0:
        PIL_syms = Stx
    else:
        if 'pilotSequence' in CPE['PILOTS']:
            PIL_syms = CPE['PILOTS']['pilotSequence']

    # If Signal is Oversampled, Perform Downsampling
    ts0 = 1
    if 'ts0' in CPE:
        ts0 = CPE['ts0']
    K = 1
    if 'downSampleFactor' in CPE:
        K = CPE['downSampleFactor']
    Srx_tmp = Srx[:, ts0-1:nSamples:nSpS]

    # Synchronize and Extract Pilots
    z,zz,Srx_PIL,Stx_PIL,PIL_idx,zzz = pilotSymbols_rmv(Srx_tmp,Stx,PIL_rate,PIL_syms)

    # Calculate Phase using Pilots
    phi = np.zeros((nPol, int(nSamples/nSpS)))
    for n in range(0, nPol):
        F = Stx_PIL[0,n] * np.conj(Srx_PIL[0,n])
        F = F[0:nSamples:K]
        # Apply Moving Average to Average Out Noise:
        H = np.convolve(F,np.ones(nTaps,dtype='float'), 'same')/np.convolve(np.ones(len(F)),np.ones(nTaps), 'same')
        # Unwrap the Phase:
        phi_CPE = np.unwrap(np.arctan2(np.imag(H), np.real(H)))
        # Interpolate:
        phi[n,:] = interp1d(PIL_idx[n][0][0:len(PIL_idx[n][0]):K], phi_CPE, kind='linear', fill_value='extrapolate')(range(0, int(nSamples/nSpS)))

    # Correct Carrier Phase
    phi = -rectpulse(phi,nSpS)
    Srx = Srx_tmp*np.exp(-1j*phi)

    return Srx, phi