import math
import numpy as np
import scipy.io as sio
from PyDSP_core.TX import modem
import os

def QAM_config(*args):
    # Input Parser
    # Default parameter values:
    QAM = {}
    nPol = 2
    encoding = 'normal'
    modulation = 'QAM'
    classe = ""
    symbolOrder = 'gray'
    phaseRot = 0

    # Assignment of input parameters:
    if len(args)==1:    # Caso em que a funcao recebe 1 argumento (dict).
        SIG = args[0]
        M = SIG['M']
        if 'nPol' in SIG:
            nPol = SIG['nPol']
        if 'encoding' in SIG:
            encoding = SIG['encoding']
        if 'modulation' in SIG:
            modulation = SIG['modulation']
        if 'classe' in SIG:
            classe = SIG['classe']
        if 'symbolOrder' in SIG:
            symbolOrder = SIG['symbolOrder']
        if 'phaseRot' in SIG:
            phaseRot = SIG['phaseRot']
    else:
        for n in range(0,len(args),2):
            varName = args[n]
            varValue = args[n+1]
            if varName.lower() == 'm':
                M = varValue
            elif varName.lower() == 'npol':
                nPol = varValue
            elif varName.lower() == 'encoding':
                encoding = varValue
            elif varName.lower() == 'modulation':
                modulation = varValue
            elif varName.lower() == 'classe':
                classe = varValue
            elif varName.lower() == 'symbolorder':
                symbolOrder = varValue
            elif varName.lower() == 'phaserot':
                phaseRot = varValue
    if modulation == 'QAM' and not classe:
        if math.sqrt(M) % 1 == 0:
            classe = 'square'
        else:
            classe = 'cross'

    # Assign Constellation
    flag = 0
    symbolMap = np.arange(0, M)
    if modulation == 'QAM':
        if math.sqrt(M) % 1 != 0:
            if classe == 'cross':
                flag = 1
                MF_ID = f'{M}' + modulation + '_' + classe + ".mat"
                mat = sio.loadmat(os.path.dirname(__file__) + "\\constellations\\" + MF_ID)
                const = mat['Constellation'][0]
                symbolMap = mat['SymbolMapping'][0]
            elif M == 8 and not classe:
                classe = 'rect'
                M_rect = np.array([2, 4])
        if flag == 0:
            MF_ID = f'{M}' + modulation + '_' + classe + ".mat"
            mat = sio.loadmat(os.path.dirname(__file__) + "\\constellations\\" + MF_ID)
            const = mat['Constellation'][0]
            symbolMap = mat['SymbolMapping'][0]
            #const = modem.QAMModem(M).constellation
    elif modulation == 'PAM':
        const = modem.PAMModem(M).constellation
    elif modulation == 'PSK':
        const = modem.PSKModem(M).constellation
    # const = const * math.exp(1j*phaseRot)##################################### rotaçao nao esta a funcionar

    # Configure Modulation Format Parameters
    # Determine all radii in the constellation:
    radius = np.unique(abs(const))
    radius = sorted(radius/max(radius), reverse=True)
    modFormat = f'{M}' + modulation
    # If there are two polarization, change modulation format ID accordingly:
    if nPol == 2:
        modFormat = "PM-" + modFormat

    # Determine Constellation Mapping
    # Symbol mapping and indices:
    symbolInd = np.zeros(M)
    for n in range(0,M):
        symbolInd[n] = np.where(symbolMap == n)[0]
    IQmap = np.vstack(const[symbolInd.astype(int)])

    # Mapping symbols to bits:
    if math.log2(M) % 1 == 0:                           # Os simbolos das constelaçoes ja vêm ordenados, de modo a
        sym2bitMap = np.zeros((M, int(math.log2(M))))   # que haja uma correspondecia direta entre os conjuntos de
        for n in range (0, M):                          # bits que formam uma palavra e os seus indices.
            b = bin(n)[2:].zfill(int(math.log2(M)))     # Ex: 0000 -> sym 1 ; 0001 -> sym 2 ; 0010 -> sym3 ...
            sym2bitMap[n] = np.fromiter(b, (str,int(math.log2(M))))
    # LSB bit map for differential encoding:
    if encoding == 'diff-quad':
        QAM['LSB_bitMap'] = np.empty((M, int(math.log2(M))-2))
        QAM['LSB_bitMap'][:] = np.NaN

    # Calculate average and maximum constellation powers:
    S_meanP = np.mean(pow(abs(IQmap),2))
    S_maxP = np.max(pow(abs(IQmap),2))

    # Output QAM Struct
    QAM['modFormat'] = modFormat
    QAM['mode'] = modulation
    if QAM['mode'] == 'QAM':
        QAM['class'] = classe
        if QAM['class'] == 'rect':
            QAM['M_rect'] = M_rect
    QAM['M'] = M
    QAM['nBpS'] = math.log2(M)
    QAM['entropy'] = math.log2(M)
    QAM['radius'] = radius
    QAM['nPol'] = nPol
    QAM['meanConstPower'] = S_meanP
    QAM['maxConstPower'] = S_maxP
    QAM['IQmap'] = IQmap
    if 'sym2bitMap' in locals():
        QAM['sym2bitMap'] = sym2bitMap
    QAM['encoding'] = encoding

    return QAM