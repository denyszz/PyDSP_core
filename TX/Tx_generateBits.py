import math
import numpy as np
from .PRBS_generator import PRBS_generator

def Tx_generateBits(nSyms,M,nPol,BIT):
    #Set Default Bit Source
    if 'source' not in BIT:
        BIT.source = 'randi'

    nBits = int(np.floor(nSyms*np.log2(M)))
    txBits = np.empty((nPol, nBits))
    txBits[:] = np.NaN

    if BIT['source'] == 'randi':
        if 'seed' in BIT:
            for n in range (0, nPol):
                rng = np.random.default_rng((BIT['seed']))
                txBits[n,:] = rng.integers(2, size = nBits)
        else:
            for n in range (0, nPol):
                rng = np.random.default_rng()
                txBits[n,:] = rng.integers(2, size = nBits)

    elif BIT['source'] == 'PRBS':
        # Falha para alguns casos: quando 'nBits' é maior que o tamanho de 'prbs', ou seja,
        # quando, nextpow2(nBits) = log2(nBits)
        for n in range (0, nPol):
            prbs = PRBS_generator(1, nextpow2(nBits), BIT['seed'] + n)
            txBits[n] = prbs[0][0:nBits] # Acrescentar [0] porque neste caso Num_PRBS é sempre 1

    elif BIT['source'] == 'PRBS-QAM':
        txBits = QAM_PRBSgenerator(BIT,M,nPol,nSyms) # # - funçao nao imlementada - # #

    elif BIT['source'] == 'findChannelImpulseResponse':
        txBits = 1

    return txBits

def nextpow2(x):
    return 1 if x == 0 else math.ceil(math.log2(x))