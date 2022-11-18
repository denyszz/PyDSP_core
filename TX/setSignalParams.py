import math

def setSignalParams(*args):         #Funçao capaz de receber um numero variavel de argumentos.
    SIG = {}
    if len(args)<=3:                #Caso em que a funcao recebe 3 ou menos parametros,
        SIG = args[0]               #assumindo que o primeiro é um dict.
        symRate = SIG['symRate']
        if 'nBpS' in SIG:
            nBpS = SIG['nBpS']
        if 'nPol' in SIG:
            nPol = SIG['nPol']
        if 'rollOff' in SIG:
            rollOff = SIG['rollOff']
        if len(args)==2:            #Caso em que a funcao recebe 2 parametros, sendo
            PARAM = args[1]         #que o segundo tambem é um dict (com 2 campos).
            sampRate = PARAM['sampRate']
            nSamples = PARAM['nSamples']
        elif len(args)==3:          #Caso em que a funcao recebe 3 parametros, sendo
            sampRate = args[1]      #que o segundo e terceiro NAO sao dicts.
            nSamples = args[2]
    else:
        for n in range(0,len(args),2):
            varName = args[n]
            varValue = args[n+1]
            if varName == 'symRate' or varName == 'symbol-rate':
                symRate = varValue
            if varName == 'M':
                SIG['M'] = varValue
            if varName == 'nBpS':
                nBpS = varValue
            if varName == 'nPol':
                nPol = varValue
            if varName == 'roll-off':
                rollOff = varValue
            if varName == 'sampRate':
                sampRate = varValue
            if varName == 'nSpS':
                nSpS = varValue
            if varName == 'nSyms':
                nSyms = varValue
            if varName == 'encoding':
                SIG['encoding'] = varValue
            if varName == 'modulation':
                SIG['modulation'] = varValue

    try:
        SIG['M']
    except KeyError:
        print('You must specify the constellation size, M')

    if 'nBpS' not in locals():
        nBpS = math.log2(SIG['M'])
    if 'nPol' not in locals():
        nPol = 2
    if 'rollOff' not in locals():
        rollOff = 0.5
    if 'encoding' not in SIG:
        SIG['encoding'] = 'normal'
    if 'modulation' not in SIG:
        SIG['modulation'] = 'QAM'
    if 'sampRate' not in locals():
        if 'nSpS' in locals():
            SIG['nSpS'] = nSpS
            sampRate = nSpS * symRate
    if 'nSamples' not in locals():
        if 'nSyms' in locals():
            SIG['nSyms'] = nSyms
            if 'sampRate' in locals():
                nSamples = sampRate/symRate * nSyms

    #Secondary Parameters
    bitRate = symRate * nBpS * nPol
    tSym = 1/symRate
    tBit = nPol/bitRate

    #Signal Parameters that Depend on the Simulation Parameters
    if 'sampRate' in locals() and 'nSamples' in locals():
        SIG['nSpS'] = sampRate/symRate
        SIG['nSyms'] = nSamples/SIG['nSpS']
        SIG['nBits'] = SIG['nSyms'] * nBpS

    #Set QAM fields
    SIG['symRate'] = symRate
    SIG['bitRate'] = bitRate
    SIG['nBpS'] = nBpS
    SIG['nPol'] = nPol
    SIG['tSym'] = tSym
    SIG['tBit'] = tBit
    SIG['rollOff'] = rollOff

    return SIG