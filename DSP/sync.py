import numpy as np
import sympy
from scipy import signal

def pilotSymbols_rmv(Srx,Stx,PIL_rate,PIL_syms):
    # Input Parameters
    nPol = len(Stx)

    # Synchronize Pilots
    A, B = sympy.nsimplify(PIL_rate,tolerance=(1e-6)*PIL_rate,rational=True).as_numer_denom()
    A=int(A)
    B=int(B)
    if B-A > 1:
        print('Current version of pilot-based CPE only supports rates of (N-1)/N')

    # Upsample
    PIL = np.zeros((nPol,B*len(PIL_syms[0])), dtype=complex)
    for i in range(0,nPol):
        PIL[i]=np.pad(signal.upfirdn([1],PIL_syms[i],B),(0,B-1))
    tmp,SYNC = syncSignals_NxN(Stx,PIL)

    pilot_idx = np.empty(nPol, dtype=object)
    nPilots = np.empty(nPol)
    lastPilot_idx = np.empty(nPol)
    for n in range(0,nPol):
        pilot_idx[n] = np.where(tmp[n,:])
        nPilots[n] = len(pilot_idx[n][0])
        lastPilot_idx[n] = pilot_idx[n][0][-1]
    # Assumir que o numero de pilots é o mesmo em todas a polarizaçoes

    nSamples = np.size(Stx,1)
    # Extract Pilot Symbols to Separate Vectors
    Stx_PIL = np.empty((1,nPol), dtype=object)
    Srx_PIL = np.empty((1,nPol), dtype=object)
    for n in range(0,nPol):
        Stx_PIL[0,n] = PIL[n,pilot_idx[n][0] - int(SYNC['syncPoint'][n])+1]
        ################################## Rever isto!!! #########################################
        Srx_PIL[0,n] = Srx[n,pilot_idx[n][0]]
        ##########################################################################################

    # Remove Pilot Symbols
    Stx_out = np.zeros((nPol,int(nSamples-nPilots[0])), dtype=complex)
    Srx_out = np.zeros((nPol,int(nSamples-nPilots[0])), dtype=complex)
    for n in range(0,nPol):
        Stx_out[n,:] = Stx[n,np.setdiff1d(np.arange(0,nSamples), pilot_idx[n][0])]
        Srx_out[n,:] = Srx[n,np.setdiff1d(np.arange(0,nSamples), pilot_idx[n][0])]

    return Srx_out,Stx_out,Srx_PIL,Stx_PIL,pilot_idx,SYNC

def syncSignals_NxN(RX,TX,SYNC={}):
    # Input Parser
    # Check for Sync Method and Debug Flag:
    SYNC_method = 'complexField'
    avoidSingularity = True

    if 'method' in SYNC:
        SYNC_method = SYNC['method']
    if 'avoidSingularity' in SYNC:
        avoidSingularity = True
    SYNC['method'] = SYNC_method

    #Input Parameters
    nPol = len(RX)

    #Test all Synchronization Combinations
    polRot = np.zeros(nPol, dtype=bool)

    tmp = np.empty((nPol,nPol), dtype=object)
    D = np.empty((nPol,nPol))
    G = np.empty((nPol,nPol))
    R = np.empty((nPol,nPol))

    TX_sync =  np.zeros((nPol,len(RX[0])), dtype=complex)
    syncPoint = np.zeros(nPol)

    for n in range(0,nPol):
        for k in range(0,nPol):
            tmp[n,k], D[n,k], G[n,k], R[n,k] = syncSignals(RX[n,:], TX[k,:], SYNC)

    # Choose Best Synchronization
    # Find maximum synchronization strength among all polarizations:
    idx1, idx2 = np.where(G == np.max(G[:]))
    idx = [idx1[0], idx2[0]]
    TX_sync[idx[0],:] = tmp[idx[0],idx[-1]]
    syncPoint[idx[0]] = D[idx[0],idx[-1]]

    if len(idx) > 0 and abs(np.diff(idx)) > 0:
        polRot[idx[1]] = True

    # Synchronize the other polarization(s):
    # 2 polarizations only
    if avoidSingularity and nPol == 2:
        polRot[idx[1]] = polRot[idx[0]]
        idx[0] = np.remainder(idx[0]+1,2)
        idx[1] = np.remainder(idx[1]+1,2)
        TX_sync[idx[0],:] = tmp[idx[0],idx[1]]
        syncPoint[idx[0]] = D[idx[0],idx[1]]
    # 3+ polarizations
    else:
        for n in range(0, nPol):
            if n != idx[0]:
                idx_new = np.argmax(G[n,:])
                TX_sync[idx_new] = tmp[n,idx_new]
                syncPoint[idx_new] = D[n, idx_new]
                polRot[idx_new] = idx_new != n

    # Output SYNC Struct
    SYNC['syncPoint'] = syncPoint + 1
    SYNC['G'] = G
    SYNC['delay'] = D
    SYNC['rotation'] = R
    SYNC['polRotation'] = polRot

    return TX_sync, SYNC

def syncSignals(A,B,SYNC):

    showPlots = False
    maxDelay = np.inf
    minDelay = -np.inf
    evalDelay = True

    if 'SYNC' not in locals():
        SYNC['method'] = 'complexField'
    elif isinstance(SYNC, str):
        SYNC_tmp={}
        SYNC_tmp['method'] = SYNC
        SYNC = SYNC_tmp
    if 'method' not in SYNC:
        SYNC['method'] = 'complexField'
    if 'presetDelay' in SYNC:
        delay = SYNC['presetDelay']
        evalDelay = False
    if 'maxDelay' in SYNC:
        maxDelay = SYNC['maxDelay']
    if 'minDelay' in SYNC:
        minDelay = SYNC['minDelay']
    if 'nPeakIgnore' not in SYNC:
        SYNC['nPeakIgnore'] = 0
    if 'applyRotation' not in SYNC:
        SYNC['applyRotation'] = True

    # Input parameters
    N_A = np.size(A,0)
    N_B = np.size(B,0)
    if N_A > N_B:
        A = A[0:N_B]

    # Calculate Cross Correlation Between A and B
    if evalDelay:
        if SYNC['method'] == 'abs':
            AB = signal.correlate(abs(A) - np.mean(abs(A)), abs(B) - np.mean(abs(B)), mode='full')
            lags = signal.correlation_lags(A.size, B.size, mode='full')
        else:
            AB = signal.correlate(A, B, mode='full')
            lags = signal.correlation_lags(A.size, B.size, mode='full')

        AB = AB[lags <= maxDelay]
        lags = lags[lags <= maxDelay]

        AB = AB[lags >= minDelay]
        lags = lags[lags >= minDelay]

    # Calculate SyncPoint
    if evalDelay:
        meanAB = np.mean(abs(AB))
        maxAB = max(abs(AB))
        maxABind = np.argmax(abs(AB))

        # Ignore N first correlation peaks, if applicable:
        # for n in range(0,SYNC['nPeakIgnore']):
        #### Nao implementado. ####

        peakGain = 10 * np.log10(maxAB/meanAB)
        delay = lags[maxABind]

    # Synchronize B relatively to A
    if delay >= 0:
        Bhead = B[len(B)-delay:len(B)]
        Btail = B[0:np.remainder(N_A-delay,N_B)]
        B_sync = np.concatenate((Bhead, np.matlib.repmat(B,1,int(np.floor((N_A-delay+1)/N_B)))[0], Btail))
    else:
        Bhead = B[abs(delay)+1:-1]
        Btail = B[0:np.remainder(N_A-len(Bhead), N_B)]
        B_sync = np.concatenate((Bhead, np.matlib.repmat(B,1,int(np.ceil((N_A-abs(delay+1))/N_B)))[0], Btail))
    # Truncate B if lenght(B)>length(A)
    if len(B_sync > N_A):
        B_sync = B_sync[0:N_A]

    # Find Rotation
    if evalDelay and SYNC['applyRotation']:
        if abs(AB[maxABind].imag) > abs(AB[maxABind].imag):
            if AB[maxABind].imag < 0:
                rot = -np.pi/2
            else:
                rot = np.pi/2
        else:
            if AB[maxABind].real < 0:
                rot = -np.pi
            else:
                rot = 0
    else:
        rot = 0

    B_sync = B_sync*np.exp(1j*rot)

    return B_sync,delay,peakGain,rot