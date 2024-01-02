import numpy as np


def EVM_eval(Srx, Stx, tMem=None, calc_EVM_per_sym=None):
    # Calculate Average EVM
    EVM = np.sqrt(sum((abs(Srx - Stx) ** 2)) / sum(abs(Stx) ** 2)) * 100

    # EVM per Symbol e EVM Evolution in Time nao sao feitos aqui
    return EVM


def BER_eval(txBits, rxBits, BER_sym_eval=None, M=None):
    nBits = len(txBits)
    # Evaluate BER
    errPos = np.where(txBits != rxBits)[0]
    nBitErr = len(errPos)
    BER = nBitErr / nBits

    # Para já nao avalia o BER por simbolo
    return BER, errPos


def MI_eval(Srx,Stx,C,N0,symProb=None):
    if symProb == None:
        symProb = np.matlib.repmat(1/len(C), len(C),1)

    qYonX = np.exp((-np.abs(Srx-Stx)**2)/N0)
    qY = np.sum(np.matlib.repmat(symProb,1, len(Srx)) * np.exp((-np.abs(Srx-np.vstack(C))**2)/N0),0)
    MI = np.mean(np.log2(np.maximum(qYonX, np.spacing(1))/np.maximum(qY, np.spacing(1))))

    return MI

def GMI_eval(Srx,txBits,C,N0,symProb=0):
    ### So funciona para uma polarizacao ###
    #Input Parameters
    if symProb == 0:
        symProb = np.matlib.repmat(1/len(C),len(C),1)
    nSyms = np.size(Srx,1)
    M = len(C)
    nBpS = int(np.log2(M))
    bMap = np.zeros((M, int(np.log2(M))))
    for n in range (0, M):
        b = bin(n)[2:].zfill(int(np.log2(M)))
        bMap[n] = np.fromiter(b, (str,int(np.log2(M))))

    #Evaluate Channel Transition Probability
    CC = np.exp(-abs(Srx-C)**2/N0)*np.matlib.repmat(symProb,1,len(Srx))
    B = np.sum(CC,0)

    #Calculate LLRs
    Z=np.zeros(nSyms)
    for n in range(0,nBpS):
        aux1=bMap[:,n]
        aux2=txBits[0,np.arange(n,n+(nSyms)*nBpS,nBpS)]
        idx = (aux1[:, np.newaxis] == aux2[np.newaxis, :]).astype(int)
        Z = Z - np.log2(np.maximum(sum(CC * idx),np.spacing(1)))
    Z = Z + nBpS*np.log2(B)
    G = np.mean(Z)

    #Evaluate GMI
    Hx = entropy_eval(symProb)
    GMI = Hx - G
    #Normalized GMI
    NGMI = 1 - (Hx-GMI)/nBpS

    return GMI,NGMI

def entropy_eval(symProb):
    symProb = symProb / sum(symProb)
    tmp = np.log2(symProb)
    tmp[np.isinf(tmp)] = 0
    entropy = -sum(symProb*tmp)

    return entropy