import sympy
import numpy as np
from PyDSP_core.TX.pulseShaper import rectpulse

def Tx_addPilots(Stx,PILOTS,C):
    # Input parser
    if 'option' in PILOTS:
        pilotOption = PILOTS['option']
    else:
        pilotOption = 'meanQPSK'
    if 'offset' in PILOTS:
        offset = PILOTS['offset']
    else:
        offset = 0

    # Input Parameters
    nPol, nSyms = Stx.shape
    pilotRate = PILOTS['rate']
    if pilotRate == 1:
        Stx_pilot = np.empty()
        idx_pilots = np.empty()
        return Stx,Stx_pilot,idx_pilots
    if nSyms > 0:
        meanP = np.mean(abs(Stx[:])**2)
    else:
        meanP = np.mean(abs(C[:])**2)

    # Calculate Symbol Indices for Tx Pilots and Payload
    A, B = sympy.nsimplify(pilotRate,tolerance=(1e-6)*pilotRate,rational=True).as_numer_denom()  #consegue retornar exatamente o mesmo resultado do rat() do matlab
    A=int(A)
    B=int(B)
    if A > 0:
        idx_pilots = np.arange(A,B)
        nAB = np.floor(nSyms/A)
    else:
        idx_pilots = 1
        nAB = PILOTS['nPilots']

    nPilots = len(idx_pilots)
    idx_pilots = np.matlib.repmat(idx_pilots,1,int(nAB))[0] + B*(rectpulse(np.arange(0,nAB),nPilots)[0])-offset
    nPilots = len(idx_pilots)
    nSyms_withPilots = nSyms + nPilots
    idx_payload = np.setdiff1d(np.arange(0,nSyms_withPilots), idx_pilots)
    idx_pilots = np.matlib.repmat(idx_pilots,nPol,1)
    idx_payload = np.matlib.repmat(idx_payload,nPol,1)

    # Generate Pilot Symbols
    if pilotOption == 'outerQPSK':  # only works for square QAM!
        C_pilot = C[abs(C) == max(abs(C))]

    Stx_pilot = C_pilot[np.random.randint(len(C_pilot),size=(nPol, nPilots))]

    # Add Pilot Symbols to the Transmitted Signal
    Stx_pilots = np.empty((nPol, nSyms_withPilots),dtype = 'complex_')
    Stx_pilots[:] = np.nan
    for n in range(0, nPol):
        Stx_pilots[n,idx_pilots[n,:].astype(int)] = Stx_pilot[n,:]
        Stx_pilots[n,idx_payload[n,:]] = Stx[n,:]
    Stx = Stx_pilots

    return Stx,Stx_pilot,idx_pilots