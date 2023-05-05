import numpy as np
import decimal #para o round()
import warnings
import numpy.matlib
import copy
import scipy.io as sio

##############Custom Exception#############
class Error(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
###########################################

#############################################################################
def SC_adaptiveEQ(Srx,Stx,nSpS,MIMO):
    Stx_teste = copy.deepcopy(Stx)
    Srx,MIMO = adaptiveEq(Srx,Stx,MIMO,nSpS)
    Stx_teste = Stx_teste[:,np.arange(int(MIMO['demux']['idx'][0]), int(MIMO['demux']['idx'][1])+1)]

    return Srx,Stx_teste,MIMO

def adaptiveEq(Srx,Stx,EQ,nSpS):

    Srx = normSignalPower(Srx,1,True)
    Stx = normSignalPower(Stx,1,True)

    #Input Parser
    if EQ['method']=='none' or EQ['nTaps'] == 0 or not EQ['nTaps']:
        return
    EQ = adaptiveEq_inputParser(EQ,Stx,nSpS)
    nTaps = EQ['nTaps']

    # MIMO Training
    # MIMO - Multiple-In Multiple-Out
    if 'train' in EQ:
        nf = 0
        #Set current stage to training:
        EQ['current'] = EQ['train']
        # Select signal indices for training:
        ni = nf
        nf = ni + EQ['current']['nSamples']
        indT = np.arange(ni,nf)

        #Select adaptive equalizer function:
        EQ = MIMO_NxN(Srx[:,indT.astype(int)], Stx[:,indT.astype(int)], EQ)[1] # Assim so preciso de receber o 2º valor q a funçao retorna
        EQ['train'] = EQ['current']
        EQ['train']['idx'] = [ni,nf]
        EQ['W'] = EQ['train']['W']
        # Remove current field from EQ struct:
        del EQ['current']

    # MIMO Demux
    # Set current stage to demux:
    EQ['current'] = EQ['demux']
    # Select signal indices for demux:
    if EQ['train2demux'] == 'goBack' or 'train' not in EQ:
        i1 = 0
    else:
        i1 = indT[-1]+1 - nTaps - np.remainder(indT[-1]+1,nSpS) + np.remainder(nTaps,nSpS) + 1
        #print('i1= ',i1)

    indD = np.arange(i1-1,np.size(Srx,1))
    #print(indD)
    Srx, EQ = MIMO_NxN(Srx[:,indD.astype(int)], Stx[:,indD.astype(int)], EQ)
    EQ['demux'] = EQ['current']
    #Remove current field from EQ struct:
    del EQ['current']

    # Remove Edge Samples
    Srx, idx = rmvEdgeSamplesFIR(Srx,EQ,nSpS)
    indD = indD[idx.astype(int)]
    EQ['demux']['idx']= [indD[0], indD[-1]]

    return Srx,EQ

def rmvEdgeSamplesFIR(S,FIR,nSpS):
    # Input Parser
    if 'implementation' not in FIR:
        FIR['implementation'] = 'conv'
    if 'nIter' not in FIR:
        FIR['nIter'] = 1

    # Input Parameters
    nTaps = FIR['nTaps']
    nIter = FIR['nIter']

    if FIR['implementation'] == 'vector' or FIR['implementation'] == 'conv':
        firstSample = np.ceil(np.max(nTaps)/2)*nIter
        lastSample = np.size(S,1) - np.floor(np.max(nTaps)/2)*nIter
    elif FIR['implementation'] == 'filter' or FIR['implementation'] == 'FIR':
        firstSample = custom_round(np.max(nTaps)/2)*nIter
        lastSample = np.size(S,1)
    else:
        firstSample = 1
        lastSample = np.size(S,1)

    ## No matlab so entram apartir daqui se verificarem nargin == 3 ##
    # Avoid changing the sampling instant:
    firstSample = firstSample + nSpS - np.remainder(firstSample,nSpS) + 1
    #Remove Edge Samples
    remSamples = np.floor(np.remainder(lastSample-firstSample+1,nSpS))
    ## ... ##

    idx = np.arange(firstSample-1, lastSample-remSamples)
    S = S[:,idx.astype(int)]

    return S, idx


def MIMO_NxN(Srx,Stx,MIMO):
    if MIMO['method'] == 'LMS':
        MIMO_method = 'LMS'
    elif MIMO['method'] == 'CMA':
        MIMO_method = 'CMA'
    if 'saveTaps_overTime' not in MIMO:
        MIMO['saveTaps_overTime'] = False
    if 'saveTaps_upRate' not in MIMO:
        MIMO['saveTaps_upRate'] = 1
    # Check if MIMO is real-valued:
    isReal = MIMO['isReal']
    # Check if .mex file is to be used:
    useMEX = MIMO['useMEX']
    #Get MIMO Size:
    N = len(Srx)

    # Set MIMO Parameters
    nTaps = MIMO['nTaps']
    mu = MIMO['current']['stepSize']
    upRate = MIMO['current']['updateRate']
    upOff = MIMO['current']['updateOffset']
    xPol = MIMO['current']['applyPolDemux']
    nBits = MIMO['nBits_taps']
    nTapsCPE = MIMO['nTapsCPE']
    updateRule = MIMO['current']['updateRule']
    constrained = False
    if updateRule == 'RDE' and 'constrained' in MIMO['current']:
        constrained = MIMO['current']['constrained']
    DEBUG = MIMO['current']['DEBUG']

    #Set Reference Signal and Constellation
    if MIMO_method == 'CMA':
        if updateRule =='DA':
            F = abs(Stx)**2
            C = np.NaN
        elif updateRule =='RDE':
            F = np.NaN
            C = np.reshape(MIMO['current']['radius'],(1,MIMO['current']['radius'].size))[0]
    elif MIMO_method == 'LMS':
        if updateRule =='DA':
            F = Stx
            C = np.NaN
        elif updateRule == 'DD':
            C = np.unique(Stx[0,:])
            F = np.NaN

    # Initialize Filter Taps
    if 'W' in MIMO:
        W = MIMO['W']
    else:
        W = np.zeros((N*nTaps,N), dtype= 'complex_')
        if MIMO_method == 'CMA' or updateRule == 'DD':
            n = int(np.floor(nTaps/2)+1)
            for k in range(0,N):
                W[(n-1)+(k)*nTaps,k] = 1

    # Set mask for filter taps:
    W_mask = np.ones((N*nTaps,N))
    # Taps for phase tracker:
    nCPE = N
    W_CPE = np.single(np.ones((nCPE,nTapsCPE))).astype(complex)

    # Apply MIMO NxN (sem mex):
    Srx,W,err,MIMO['Wt'] = MIMO_NxN_complex(Srx,F,C,MIMO_method,
                                            updateRule,W,W_mask,W_CPE,nTaps,mu,upRate,upOff,nBits,
                                            constrained,MIMO['saveTaps_overTime'],MIMO['saveTaps_upRate'])
    # Output parameters
    MIMO['current']['W'] = W
    MIMO['current']['err'] = []
    MIMO['current']['F'] = F

    return Srx, MIMO

def MIMO_NxN_complex(X,F,C,method,upRule,W,W_mask,W_CPE,
                     nTaps,mu,upRate,upOff,nBits,constrained,
                     saveTaps_overTime,saveTaps_upRate):
    # Input Parameters
    # Check for quantized taps:
    if np.isinf(nBits):
        quantizedTaps = False
        nLevels = np.single(0)
    # Check for phase tracker:
    if W_CPE.size == 0:
        phaseTracker = False
    # Define method:
    isCMA = False
    isLMS = False
    if method == 'CMA':
        isCMA = True
    if method == 'LMS':
        isLMS = True
    # Define update rule:
    isDA = False
    isDD = False
    isRDE = False
    if upRule == 'DA':
        isDA = True
    if upRule == 'DD':
        isDD = True
    if upRule == 'RDE':
        isRDE = True
    # Check for mask on filter taps
    W_mask_apply = False
    # Initialize variables:
    nSamples = np.single(np.size(X,1))
    N = np.single(len(X))
    #print(nSamples)
    #print(N)

    aux = np.empty((N.astype(int),nSamples.astype(int)),dtype = 'complex_')
    aux[:] = np.nan #variavel auxiliar
    aux2 = np.empty((N.astype(int),nSamples.astype(int)),dtype = 'complex_')
    aux[:] = np.nan #variavel auxiliar

    Y = aux ################################## Porquê formar uma array complexo se depois o single() vai eliminar a parte imaginaria?####################################################################################################################
    #print(Y)
    err = aux2
    #print(err)
    U = np.zeros(int(N)*int(nTaps),dtype = 'complex_')
    #print(U)
    Fref = np.zeros(int(N),dtype = 'complex_') # No matlab é um array vertical, faz diferença?
    #print(Fref)
    if not saveTaps_overTime:
        Wt = np.NaN

    # Run MIMO NxN
    # Determine initial indices:
    idx_taps = np.arange(nTaps,0,-1)
    idx_Wt = 0
    idx_up = 0

    for n in range(int(np.ceil(nTaps/2)), int(nSamples-np.floor(nTaps/2) + 1)):
        #for n in range(51,52):
        for k in range(0,int(N)):
            #print(U.shape)
            #print(X.shape)
            U[(k)*nTaps+np.arange(0,(k+1)*nTaps)] = X[k,idx_taps-1]
            #print("lindo")
            #print(U)

        # MIMO NxN (as vector multiplication):
        Y[:,n-1] = np.transpose(np.dot(U,W))
        #print(Y)

        # Update MIMO Taps:
        if np.fmod(n+upOff,upRate) == 0: # Equivalente ao rem() do matlab. Nao confundir com o mod()
            #Format equalized signal:
            thisY = Y[:,n-1]
            if isCMA:
                R = abs(thisY)
            else:
                R = np.single(0)

            #Choose reference signal:
            if isDA:
                Fref = F[:,n-1]
            elif isDD:
                #Slicer:
                for k in range(0,int(N)):
                    idx = np.argmin(abs(C - thisY[k]))
                    Fref[k] = C[idx]
            elif isRDE:
                #Determine the closest radius:
                for k in range(0,int(N)):
                    r = np.argmin(abs(C - R[k]))
                    Fref[k] = C[r]**2

            # Phase Tracker nao entra aqui.
            # Calculate error:
            if isLMS:
                err[:,n-1] = Fref - thisY
            elif isCMA:
                err[:,n-1] = (Fref - R**2) * thisY

            # Update MIMO Taps:
            for k in range(0,int(N)):
                W[:,k] = W[:,k] + mu*np.transpose(np.conj(U))*err[k,n-1]

            #  Update Counter:
            idx_up = idx_up + 1

        # Update Tap Indeces:
        idx_taps = idx_taps +1

    return Y, W, err,Wt

def adaptiveEq_inputParser(EQ,Stx,nSpS):

    # Check Input Signal
    #print(Stx)
    nSamples = np.size(Stx,1)

    # Set Default General Parameters
    # Set default flag for real-valued equalizer:
    if 'isReal' not in EQ:
        if EQ['method'] == '4x4' or EQ['method'] == '8x8':
            EQ['isReal'] = True
        else:
            EQ['isReal'] = False

    #Set default flag for symmetric SC processing (e.g. 8x8):
    if 'symmetricSCs' not in EQ:
        if EQ['method'] =='8x8':
            EQ['symmetricSCs'] = True
        else:
            EQ['symmetricSCs'] = False

    #Set default precision for filter taps:
    if 'nBits_taps' not in EQ:
        EQ['nBits_taps'] = np.Inf

    #Set default number of taps for phase tracker:
    if 'nTapsCPE' not in EQ:
        EQ['nTapsCPE'] = 0

    # Set default strategy for switching between training and demux (continue /goBack):
    if 'train2demux' not in EQ:
        EQ['train2demux'] = 'continue'

    #Set default of reference subcarriers (in case of subcarrier mux):
    if 'nRefCarriers' not in EQ:
        EQ['nRefCarriers'] = np.Inf

    #Set default precision:
    if 'precision' not in EQ:
        EQ['precision'] = 'double'

    #Set default .mex option:
    useMEX = False
    if 'useMEX' in EQ:
        useMEX = EQ['useMEX']
    else:
        if 'mex' in EQ:
            useMEX = EQ['mex']

    EQ['useMEX'] = useMEX
    if EQ['useMEX']:
        EQ['precision'] = 'single'

    #Set Default Training Parameters
    if 'train' in EQ:
        if EQ['train']['updateRule'] in ['DA','data-aided','LMS']:
            EQ['train']['updateRule'] = 'DA'
        elif EQ['train']['updateRule'] in ['DD','decision-directed']:
            EQ['train']['updateRule'] = 'DD'
        elif EQ['train']['updateRule'] in ['blind','QPSK','RDE','radius-directed']:
            EQ['train']['updateRule'] = 'RDE'
        else:
            raise Error("Unknown adaptive equalizer update rule")

        if 'updateRate' not in EQ['train']:
            EQ['train']['updateRate'] = nSpS
        if 'updateOffset' not in EQ['train']:
            EQ['train']['updateOffset'] = nSpS - 1
        if 'stepSize' not in EQ['train']:
            EQ['train']['stepSize'] = 1e-3
        if 'DEBUG' not in EQ['train']:
            EQ['train']['DEBUG'] = False
        if EQ['method'] == 'CMA':
            if 'constrained' not in EQ['train']:
                EQ['train']['constrained'] = False
            if EQ['train']['updateRule'] == 'RDE' and 'radius' not in EQ['train']:
                EQ['train']['radius'] = np.unique(abs(Stx[0,:]))

        #Default number of samples for training stage:
        if 'nSamples' not in EQ['train']:
            if 'percentageTrain' in EQ['train']:
                EQ['train']['nSamples'] = custom_round(EQ['train']['percentageTrain']*nSamples)     # Em casos de 0.5, 1.5, 3.5 etc, o round() do python arredonda
            else:                                                                                      # para o numero par mais proximo, ao contrario do matlab que
                EQ['train']['nSamples'] = 0.5*nSamples                                              # arredonda para o numero maior.
                EQ['train']['percentageTrain'] = 0.5
        elif EQ['train']['nSamples'] > nSamples:
            warnings.warn('The number of required training samples is higher than the total number of samples. The total number of samples will be used instead.')
            EQ['train']['nSamples'] = nSamples
            EQ['train']['percentageTrain'] = 1
        else:
            EQ['train']['percentageTrain'] = EQ['train']['nSamples']/nSamples
        #Set default pol-demux flag:
        if 'applyPolDemux' not in EQ['train']:
            EQ['train']['applyPolDemux'] = True

    #Set Default Demux Parameters
    if 'demux' in EQ:
        if EQ['demux']['updateRule'] in ['DA','data-aided','LMS']:
            EQ['demux']['updateRule'] = 'DA'
        elif EQ['demux']['updateRule'] in ['DD','decision-directed']:
            EQ['demux']['updateRule'] = 'DD'
        elif EQ['demux']['updateRule'] in ['blind','QPSK','RDE','radius-directed']:
            EQ['demux']['updateRule'] = 'RDE'
        else:
            raise Error("Unknown adaptive equalizer update rule")

        if 'updateRate' not in EQ['demux']:
            EQ['demux']['updateRate'] = nSpS
        if 'updateOffset' not in EQ['demux']:
            EQ['demux']['updateOffset'] = nSpS - 1
        if 'stepSize' not in EQ['demux']:
            EQ['demux']['stepSize'] = 1e-3
        if 'DEBUG' not in EQ['demux']:
            EQ['demux']['DEBUG'] = False
        if EQ['method'] == 'CMA':
            if 'constrained' not in EQ['demux']:
                EQ['demux']['constrained'] = False
            if EQ['demux']['updateRule'] == 'RDE' and 'radius' not in EQ['demux']:
                x1=abs(Stx[0,:])*(1e3)
                x2=np.vectorize(custom_round)(x1)
                EQ['demux']['radius'] = np.unique(x2.astype(str).astype(float)/(1e3))
        #Set default pol-demux flag:
        if 'applyPolDemux' not in EQ['demux']:
            EQ['demux']['applyPolDemux'] = True

    return EQ

def normSignalPower(S,normPower,jointNorm):
    """
    Rever a package matlib!
    """
    nSig = len(S)
    # Input Parameters
    if not isinstance(normPower, (frozenset, list, set, tuple, np.ndarray)):
        normPower = np.matlib.repmat(normPower,1,nSig)[0]

    # Normalization Factors
    normFactor = np.ones(nSig)

    for n in range(0,nSig):
        normFactor[n] = np.sqrt(np.mean(abs(S[n,:])**2) / normPower[n])

    # Get Average Normalization Factor
    if np.any(normFactor == 0):
        jointNorm = False
    if jointNorm:
        normFactor = np.matlib.repmat(np.mean(normFactor),1,nSig)[0] #################ver outravez!!!!

    #Normalize Optical Field
    normFactor[normFactor == 0] = 1
    for k in range(0,nSig):
        S[k] = S[k] / normFactor[k]

    # Mean Signal Power After Normalization
    meanP = np.mean(abs(S)**2, axis=1)
    #print(S)
    #return S,normPower,jointNorm
    return S

def custom_round(x):
    return decimal.Decimal(x).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)