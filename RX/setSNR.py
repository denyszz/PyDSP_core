import numpy as np

def setSNR(S,SNR,Fs,Rs):
    # Input Parser
    if type(SNR) == int or type(SNR) == float:
        SNR_tmp ={
            'SNRout_dB': SNR
        }
        SNR = SNR_tmp
    if 'SNRin_dB' not in SNR:
        SNR['SNRin_dB'] = np.inf

    nPol, nSamples = S.shape
    #Determine Signal Power
    Ps_in = np.mean(abs(S)**2,1)
    if 'Pin' in SNR:
        Ps = SNR['Pin']
    else:
        Ps = Ps_in

    # Determine Noise Power
    if 'Pn' in SNR:
        Pn_Fs = SNR['Pn']
        if len(Pn_Fs) == 1:
            Pn_Fs = np.tile(Pn_Fs,(1,nPol))
        if Fs in locals() and Rs in locals():
            Pn_Rs = Pn_Fs*Rs/Fs
    else:
        SNRout = 10**(SNR['SNRout_dB']/10)
        SNRin = 10**(SNR['SNRin_dB']/10)
        Pn0 = Ps/SNRin
        Pn_Rs = Ps/SNRout - Pn0
        Pn_Fs = Pn_Rs*Fs/Rs

    # Generate Noise
    # Set random number generator seed:
    ### -- Standard Normal Distribution (randn) -- ###
    # - Matlab and NumPy use different transformations
    # - to create samples from the standard normal distribution.
    # - As sequencias do matlab geradas pelo randn() nao
    # - podem ser replicadas em python.
    if 'noiseSeed' in SNR:
        np.random.seed(SNR['noiseSeed'])
    else:
        seed = np.random.randint(2**31)
        SNR['noiseSeed'] = seed
        np.random.seed(SNR['noiseSeed'])

    # Generate noise in the I and Q components:
    noise_I = np.zeros((nPol,nSamples))
    noise_Q = np.zeros((nPol,nSamples))
    for n in range(0,nPol):
        noise_I[n,:] = np.random.randn(nSamples) * np.sqrt(Pn_Fs[n]/2)
        noise_Q[n,:] = np.random.randn(nSamples) * np.sqrt(Pn_Fs[n]/2)

    # Color the noise:
    ### falta fazer isto ###

    # Create the complex-valued noise:
    noise = noise_I + 1j*noise_Q

    # Add Noise to the Signal
    S = S + noise

    # Calculate Ouput SNR
    Pn_Fs = np.mean(Pn_Fs)
    if 'Pn_Rs' in locals():
        SNR_out = (10*np.log10(np.mean(Ps_in)/np.mean(Pn_Rs))+300)-300
    else:
        SNR_out = np.NaN

    return S, Pn_Fs, SNR_out, noise