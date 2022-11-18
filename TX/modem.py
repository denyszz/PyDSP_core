"""
'Digital Modulations using Python'
Mathuranathan Viswanathan

Alterei algumas coisas para bater certo com o matlab,
mas o codigo nao está muito efeciente.
"""
import numpy as np
import abc
import matplotlib.pyplot as plt

class Modem:
    __metadata__ = abc.ABCMeta
    # Base class: Modem
    # Attribute definitions:
    #    self.M : number of points in the MPSK constellation
    #    self.name: name of the modem : PSK, QAM, PAM
    #    self.constellation : reference constellation
    def __init__(self,M,constellation,name): # constructor
        if (M<2) or ((M & (M -1))!=0): # if M not a power of 2
            raise ValueError('M should be a power of 2')

        self.M = M # number of points in the constellation
        self.name = name # name of the modem : PSK, QAM, PAM
        self.constellation = constellation # reference constellation
    
    def plotConstellation(self):
        """
        Plot the reference constellation points for the selected modem
        """
        from math import log2
        
        fig, axs = plt.subplots(1, 1)
        axs.plot(np.real(self.constellation),np.imag(self.constellation),'o')        
        for i in range(0,self.M):
            axs.annotate("{0:0{1}b}".format(i,int(log2(self.M))), (np.real(self.constellation[i]),np.imag(self.constellation[i])))
        
        axs.set_title('Constellation')
        axs.set_xlabel('I');axs.set_ylabel('Q')
        fig.show()

class PAMModem(Modem):
    # Derived class: PAMModem
    ####################################################################################################################
    #####################################-Alterado para codificaçao em gray-############################################
    ####################################################################################################################
    def __init__(self, M):
        m = np.arange(0,M) # Sequential address from 0 to M-1 (1xM dimension)
        n = np.asarray([x^(x>>1) for x in m]) # convert linear addresses to Gray code
        constellation_temp = 2*m+1-M + 1j*0 # constelaçao temporaria

        constellation_dict = dict(zip(n, constellation_temp)) # dicionario -> [0: -15 +0j]...
        sorted_constellation_dict = dict(sorted(constellation_dict.items())) # dicionario ordenado
        constellation = np.array(list(sorted_constellation_dict.values()))
        # reference constellation
        Modem.__init__(self, M, constellation, name='PAM') # set the modem attributes
            
class PSKModem(Modem):
    # Derived class: PSKModem
    ####################################################################################################################
    #####################################-Alterado para codificaçao em gray-############################################
    ####################################################################################################################
    def __init__(self, M):        
        # Generate reference constellation
        m = np.arange(0,M) # all information symbols m={0,1,...,M-1}
        # I = 1/np.sqrt(2)*np.cos(m/M*2*np.pi)
        # Q = 1/np.sqrt(2)*np.sin(m/M*2*np.pi)
        I = np.cos(m/M*2*np.pi)
        Q = np.sin(m/M*2*np.pi) # sem a normalizacao de 1/sqrt(2)
        constellation = I + 1j*Q # reference constellation

        Modem.__init__(self, M, constellation, name='PSK') # set the modem attributes
        
class QAMModem(Modem):
    # Derived class: QAMModem
    def __init__(self,M):
        if (M==1) or (np.mod(np.log2(M),2)!=0): # M not a even power of 2
            raise ValueError('Only square MQAM supported. M must be even power of 2')
        
        n = np.arange(0,M) # Sequential address from 0 to M-1 (1xM dimension)
        a = np.asarray([x^(x>>1) for x in n]) # convert linear addresses to Gray code
        D = np.sqrt(M).astype(int) # Dimension of K-Map - N x N matrix
        a = np.reshape(a,(D,D)) # NxN gray coded matrix
        oddRows=np.arange(start = 1, stop = D ,step=2) # identify alternate rows
        a[oddRows,:] = np.fliplr(a[oddRows,:]) # Flip rows - KMap representation

        nGray=np.reshape(a,(M)) # reshape to 1xM - Gray code walk on KMap

        # Construction of ideal M-QAM constellation from sqrt(M)-PAM
        (x,y)=np.divmod(n,D) # element-wise quotient and remainder

        Ax=2*x+1-D # PAM Amplitudes 2d+1-D - real axis
        Ay=2*y+1-D # PAM Amplitudes 2d+1-D - imag axis
        constellation = Ax+1j*Ay

        #constellation = np.reshape(constellation,(8,8))
        #constellation = np.rot90(np.flip(constellation),k=-1)
        #acabar isto...

        Modem.__init__(self, M, constellation, name='QAM') # set the modem attributes