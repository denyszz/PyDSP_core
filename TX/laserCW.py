import numpy as np

def laserCW(LASER,Fs,nSamples):
    # Input Parser
    if 'linewidth' not in LASER:
        LASER['linewidth'] = 0
    if 'RIN_dB' not in LASER:
        LASER['RIN_dB'] = -np.inf
    if 'phase0' not in LASER:
        LASER['phase0'] = 0
    if 'P0_dBm' not in LASER:
        LASER['P0_dBm'] = 30    # 1 watt per polarization
    if  LASER['linewidth'] > 0 or LASER['RIN_dB'] > -np.inf:
        # Set random number generator seed:
        if 'noiseSeed' in LASER:
            np.random.seed(LASER['noiseSeed'])
        else:
            seed = np.random.randint(2**31)
            LASER['noiseSeed'] = seed
            np.random.seed(LASER['noiseSeed'])

    # Input Parameters
    lw = LASER['linewidth']       # laser linewidth [Hz]
    RIN_dB = LASER['RIN_dB']      # relative intensity noise [dB/Hz] \
    ph0 = LASER['phase0']         # laser initial phase [rad]
    P0_dBm = LASER['P0_dBm']      # laser emitted power [dBm]
    P0 = 10**((P0_dBm-30)/10)

    # LASER phase noise
    if lw:
        phVar = 2*np.pi*lw/Fs
        phNoise = np.sqrt(phVar)*np.random.randn(nSamples)
        phNoise = np.cumsum(phNoise)
    else:
        phVar = 0
        phNoise = 0

    # LASER intensity noise
    if RIN_dB > -np.inf:
        intVar = 10^(RIN_dB/10)*Fs*P0**2
        intNoise = np.sqrt(intVar)*np.random.randn(nSamples)
    else: intNoise = 0

    # LASER transmitted optical field
    A = np.sqrt(P0 + intNoise) * np.exp(1j*(ph0 + phNoise))

    # Output LASER Struct
    LASER['phaseVar'] = phVar         # phase variance
    LASER['phaseNoise'] = phNoise     # phase noise [rad]
    LASER['RIN_dB'] = RIN_dB          # relative intensity noise [dB/Hz] \
    LASER['intNoise'] = intNoise      # intensity noise [W]

    return A, LASER