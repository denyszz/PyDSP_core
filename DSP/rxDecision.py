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


def MI_eval(Srx, Stx, C, N0, symProb=None):
    if symProb == None:
        symProb = np.matlib.repmat(1 / len(C), len(C), 1)
    qYonX = np.exp((-np.abs(Srx - Stx) ** 2) / N0)
    qY = np.dot(
        np.sum(np.matlib.repmat(symProb, 1, len(Srx)), np.exp((-np.abs(np.subtract(Srx, np.vstack(C))) ** 2) / N0)))
    MI = np.mean(np.log2(np.maximum(qYonX, np.spacing(1)) / np.maximum(qY, np.spacing(1))))

    return MI
