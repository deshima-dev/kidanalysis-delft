from .misc import c, h, kB, N0
from .Prad import dPrad
import numpy as np

def occupation(freq, T_BB):
    """
    Fermi ditribution for frequency nu [Hz], black body temperature T_BB [K]
    """
    return 1/(np.exp(h*freq/(kB*T_BB)) - 1)

def CalcNqp(T, l, w, t, Delta):
    """
    Calculate thermal equilibrium Nqp for responsivity calculation
    
    T :: 1-D array of Temperature [K]
    l :: length of Al part [um]
    w :: width of Al part [um]
    t :: thickness of Al part [um]
    Delta :: half superconductiong gap energy [J]

    Returns 1-D array of Nqp
    """
    V = l*w*t # um^3
    return V * 2.*N0* np.sqrt(2.*np.pi*kB*T*Delta) * np.exp(-Delta/kB/T)

#### NEP calculation
def NEP2_GR(tau_qp, tau0, l, w, t, Delta, eta_pb=0.57):
    """
    Calculate NEP**2 of generation-recombination noise for dark measurement
    Nqp is calculated from measured tau_qp in this function

    tau_qp :: measured quasi-particle life time [s]
    tau0 :: extracted value from fitting of temperature vs tau_qp
    Delta :: half superconductiong gap energy [J]
    l :: length of Al part [um]
    w :: width of Al part [um]
    t :: thickness of Al part [um]
    eta_pb :: pair breaking efficiency

    Returns NEP**2
    """
    V = l*w*t # um^3
    kTc = 2./3.52 * Delta # [J] from 2*Delta = 3.52*kB*Tc
    Nqp = 1./tau_qp * tau0*N0*(kTc**3)*V/2./(Delta**2)

    return 4.*(Delta**2)/eta_pb * Nqp/tau_qp

def dNEP2_photon_R(P, Delta, eta_pb=0.57):
    """
    Calculate NEP**2 from recombination in photon-noise-limit region

    P :: Radiation power [W]
    Delta :: half superconductiong gap energy [J]
    eta_pb :: pair breaking efficiency

    Returns NEP**2
    """
    #return 2*P*Delta/eta_pb
    return 4*P*Delta/eta_pb

def dNEP2_photon_Poisson(freq, P, eta_pb=0.57):
    """
    Calculate NEP**2 per bin from photon's Poisson statistics in photon-noise-limit region

    freq :: 1-D array of frequency [Hz]
    P :: 1-D array of Radiation power [W/bin]
    eta_pb :: pair breaking efficiency

    Returns 1-D array of NEP**2 per bin
    """
    return 2*P*h*freq

def dNEP2_photon_wave(freq, dP, trans, etendue, T_BB, n_pol=1):
    """
    Calculate NEP**2 per bin from photon bunching in photon-noise-limit region

    freq :: 1-D array of frequency [Hz]
    dP :: 1-D array of Radiation power [W/bin]
    trans :: 1-D array of transmittance
    etendue :: optical throughput [m^2 sr]
    T_bb :: blackbody temperature [K]

    Returns 1-D array of NEP**2 per bin
    """
    #lmd = c / freq
    #return 2*dP*h*freq*trans*etendue/(lmd**2)*occupation(freq, T_BB)
    return 2*dP*h*freq*trans*occupation(freq, T_BB)*n_pol

def dNEP2_photon_wave_new(freq, dP, trans):
    """
    Calculate NEP**2 per bin from photon bunching in photon-noise-limit region

    freq :: 1-D array of frequency [Hz]
    dP :: 1-D array of Radiation power [W/bin]
    trans :: 1-D array of transmittance

    Returns 1-D array of NEP**2 per bin
    """
    dfreq = freq[1] - freq[0] # Hz
    RJBW = np.sum( trans/np.amax(trans)*dfreq ) # Hz
    P = np.sum(dP)

    return 2*dP*P/RJBW

def NEP_photon(freq, trans, T_BB, etendue, Delta, n_pol=1):
    """
    Calculate photon NEP in photon-noise-limited region.

    freq :: 1-D array of frequency [Hz]
    P :: 1-D array of radiation power [W/bin]
    trans :: 1-D array of transmittance
    T_bb :: blackbody temperature [K]
    etendue :: optical throughput [m^2 sr]
    Delta :: half superconductiong gap energy [J]

    Returns (NEP_tot, NEP_r, NEP_p, NEP_w)
      where
        NEP_tot : total photon NEP
        NEP_r : photon NEP from recombination
        NEP_p : photon NEP from Poisson statistics
        NEP_w : photon NEP from bunching
    """

    dP_tot = dPrad(freq, trans, T_BB, etendue, n_pol) # [W per bin]
    P_tot  = sum(dP_tot) # [W]

    dNEP2_R  = dNEP2_photon_R(P_tot, Delta)
    dNEP2s_P = dNEP2_photon_Poisson(freq, dP_tot)
    dNEP2s_w = dNEP2_photon_wave(freq, dP_tot, trans, etendue, T_BB, n_pol)
            
    dNEP2_tot = dNEP2_R + np.sum(dNEP2s_P) + np.sum(dNEP2s_w)

    return (np.sqrt(dNEP2_tot), np.sqrt(dNEP2_R),
            np.sqrt( np.sum(dNEP2s_P) ), np.sqrt( np.sum(dNEP2s_w) ))

