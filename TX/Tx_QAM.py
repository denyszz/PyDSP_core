import math
import numpy as np

def Tx_QAM(QAM,txBits):
    # Generate Symbols from Bits
    if QAM['encoding'] == 'normal':
        txSyms = bit2sym(txBits, math.log2(QAM['M']))
        Stx = symbol2signal(txSyms, QAM['IQmap'])
    elif QAM['encoding'] == 'diff-quad':
        txSyms = bit2sym_DiffQuad(txBits, math.log2(QAM['M']))
        Stx = symbol2signal(txSyms, QAM['IQmap'])

    return Stx, txSyms

def bit2sym(bits, nBpS):
    # validateattributes() do Matlab nao é feito.

    # Input Parameters
    [nSig, nBits] = bits.shape
    nSyms = int(nBits/nBpS)

    # Bit-to-Symbol Assignment
    syms = np.zeros((nSig, nSyms))
    for k in range(0, nSig):
        for n in range(0, int(nBpS)):
            syms[k] = syms[k] + np.dot(bits[k][n:bits[k].size:int(nBpS)], pow(2, nBpS-n-1))

    return syms

def symbol2signal(syms,C):
    # Input Parameters
    [nSig,nSyms] = syms.shape

    # Signal to Symbol
    signal = np.zeros((nSig,nSyms), dtype = 'complex_')

    for n in range(0,nSig):
        signal[n] = C[syms[n][0:].astype(int)][:,0]

    return signal

def bit2sym_DiffQuad(bits,nBpS):
    # Input Parameters
    [nSig,nBits] = bits.shape
    nSyms = int(nBits/nBpS)
    r = np.log2(pow(2,nBpS)) - 4
    symsDiff = np.zeros((nSig,nSyms))

    # Apply Differential Encoding
    for k in range(0,nSig):
        bitsMat = bits[k].reshape(-1,nBpS)
        Qencoded = bit2sym(np.array(bitsMat[0,0:2], ndmin = 2), 2)

        Qactual = bit2sym(bitsMat[:,0:2],2)
        print(bitsMat[:,0:2])
        baseSym = bit2sym(bitsMat,nBpS)
        for n in range(0,nSyms):
            Qencoded = DiffQuadEncoder(Qactual[n],Qencoded)
            symsDiff[k,n] = baseSym[n] - Qactual[n]*4*(pow(2,r))+Qencoded*4*(pow(2,r))

    return symsDiff

def DiffQuadEncoder(Qactual, Qstart):
    # Based on Qactual we determine how many to jump from the Qstart quadrant.
    # In this case we gray encode the jump
    if Qactual == 0:
        Qjump = 0   # No jump
    elif Qactual == 1:
        Qjump = 1   # 1 quadrant jump
    elif Qactual == 2:
        Qjump = 3   # 3 quadrant jump
    elif Qactual == 3:
        Qjump = 2   # 2 quadrant jump

    Qencoded = (Qjump+Qstart)%4

    return Qencoded