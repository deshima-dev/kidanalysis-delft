import numpy as np

from . import deglitch
from .misc import ampl_phase

def kids_with_both_blinds(kidslist, tod=None, allow_without_blind=False):
    """
    search kids its nearest blind tones.

    kidslist :: a tuple of (info, kids, blinds, powers),
                that read_kidslist returns.
    tod      :: None or a OrderedDict.
                if not None, only search KID in `tod`.

    Return a list of list of form
      [[signal0, left0, right0],
       [signal1, left1, right1],
         :
         :                     
       [signalp, leftp, rightp]],
      where
        signalN : carrier bin number for KID,
        leftN   : carrier bin number for nearest left blind tone, 
        rightN  : carrier bin number for nearest right blind tone,
      respectively.
      this list is sorted by bin number in lowest-first order.
    """
    
    info, kids, blinds, powers = kidslist
    if tod:
        todkeys = tod.keys()
        kids    = np.intersect1d(kids, todkeys)

    allkeys = sorted(list(kids) + list(blinds))
    results = []
    for k in sorted(kids):
        pair = []
        pos = allkeys.index(k) - 1
        while pos >= 0:
            key = allkeys[pos]
            if (((key in blinds) and not tod) or
                ((key in blinds) and tod and (key in todkeys))):
                pair.append(allkeys[pos])
                break
            pos = pos - 1
        else:
            if allow_without_blind:
                pair.append(None)
            else:
                continue
        pos = allkeys.index(k) + 1
        while pos < len(allkeys):
            key = allkeys[pos]
            if (((key in blinds) and not tod) or
                ((key in blinds) and tod and (key in todkeys))):
                pair.append(key)
                break
            pos = pos + 1
        else:
            if allow_without_blind:
                pair.append(None)
            else:
                continue
        results.append([k] + pair)
    return np.array(results)

def deglitch_tods(rs, tods):
    """
    deglitch TOD IQ data using fitting result.

    rs   :: a list of md.KidFitResult
    tods :: a OrderedDict of md.FixedData
            representing measured KID data.

    Returns (ampls_deglitch, phases_deglitch)
      where
        ampls_deglitch is an array of 1-D array of amplitudes
        phases_deglitch is an array of 1-D array of phases
    """

    ampls, phases = [], []
    for r in rs:
        k   = r.info['bin']
        tod = tods[k]
        rw  = r.rewind(tod.frequency, tod.iq)
        ampl, phase = ampl_phase(rw)
        ampls.append(ampl)
        phases.append(phase)

    ## deglitch!
    print( 'deglitching amplitude...' )
    ampls_deglitch = deglitch.deglitch(ampls,
                                       debug=0)
    print( 'deglitching phase...' )
    phases_deglitch = deglitch.deglitch(phases, sources=ampls,
                                        debug=0)

    return ampls_deglitch, phases_deglitch








