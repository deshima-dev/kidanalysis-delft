import numpy as np

c = 2.9998e8
h  = 6.626068e-34
kB = 1.3806503e-23
eVtoJ = 1.60218e-19 # 1 eV = 1.60218e-19 J
N0 = 1.72e+10/eVtoJ # um^-3 eV^-1 for Al: convert to um^-3 J^-1

def ampl_phase(rw_tod):
    """
    get (amplitude, phase) tuple from rewinded data

    rw_tod :: a 1-D array of complex value.
              intended to used with data returned by md.KidFitResult.rewind()

    Returns (amplitude, phase)
      where amplitude is a 1-D array of real value,
            phase is a 1-D array of real value
    """
    ampl = 2*np.abs(rw_tod)
    angl = -np.angle(-rw_tod)
    return ampl, angl

def ampl_phase_loopback(IQ):
    """
    get (amplitude, phase) tuple from data measured with loopback setup.

    IQ :: a 1-D array of complex value.
          intended to used with data like md.FixedData.iq, taken with loopback setup.

    Returns (amplitude, phase)
      where amplitude is a 1-D array of real value,
            phase is a 1-D array of real value
    """
    ampl = np.abs(IQ)/np.abs(average(IQ))
    angl = np.angle(IQ)
    return ampl, angl

#http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)
