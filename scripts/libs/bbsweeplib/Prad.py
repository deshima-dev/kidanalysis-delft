import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from .misc import c, h, kB

def read_filterfile(filter_path, additional_points, factor=100):
    """
    read filter transmission files with columns
      invlambda[/cm], transmission


    filter_path :: path to filter file
    additional_points :: a tuple or list of (freq, transmission)
    factor :: factor to multiply invlambda (default: 100 to convert [/cm] to [/m])


    Returns (invlambda, transmission)
      where invlambda is 1-D array of wave number [/m],
            transmission is 1-D array of transmissions.
    """
    filt = np.loadtxt(filter_path, unpack=True)
    invlambda    = filt[0] * factor
    transmission = filt[1]
    for fr, tr in additional_points:
        invlambda    = np.concatenate(([fr], invlambda))
        transmission = np.concatenate(([tr], transmission))
    inds = np.argsort(invlambda)
    arr = np.array([invlambda[inds], transmission[inds]])
    return arr



#### Power calculation

def blackbody_brilliance(nu, T_BB, n_pol=1):
    """calculate blackbody brilliance

    nu :: 1-D array of frequency [Hz]
    T_BB :: blackbody temperature
    n_pol :: 1 for single pol, 2 for dual pol

    Returns 1-D array of brilliance [W / m sr^2 Hz].
    """
    return n_pol*h*nu**3/c**2/(np.exp(h*nu/(kB*T_BB))-1)

def dPrad(freq, trans, T_BB, Etendue, n_pol=1):
    """calculate radiant power with specifiled filter and throughput..

    Parameters:
      freq: a 1-D array of frequency [Hz]. Equidistance is assumed.
      trans: a 1-D array of filter transmittance
      T_BB: blackbody temperature [K]
      Etendue: optical throughput [m sr^2]

    Returns a 1-D array of power [W per bin].
    """
    dfreq = freq[1] - freq[0]
    P_BB = blackbody_brilliance(freq, T_BB, n_pol) # [W / m sr^2 Hz]
    dP_tot = P_BB*Etendue*trans*dfreq # [W per bin]
    return dP_tot
