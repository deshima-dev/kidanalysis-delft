import numpy as np
from astropy.io import fits as pyfits
import mkid_data as md

"""
functions to read data files.
"""

def read_kidslist(fname):
    '''Read kidslist file.

    Return (info, kids, blinds, powers)
      where info   is a dict with keys 'LO' (LO freq in Hz) and
                      framelen (frame count);
            kids   is a 1D list of bin indices for KIDs;
            blinds is a 1D list of bin indices for blind tones;
            powers is a OrderedDict, whose key is bin indices 
                      and value is power for that bin.
    '''
    mode = ['header']
    kids   = []
    blinds = []
    powers = dict()
    def checkmode(line):
        if line[:5] == '#KIDs':
            mode[0] = 'kids'
            return True
        elif line[:7] == '#blinds':
            mode[0] = 'blinds'
            return True
        else:
            return False

    with open(fname) as f:
        info = dict()
        for l in f:
            l = l.strip()
            if not l:
                continue
            elif checkmode(l):
                continue
            if mode[0] == 'header':
                #print l
                key, val = l[1:].split(':')
                if key == 'LO':
                    info['LO'] = float(val)*1e6
                elif key == 'framelen':
                    info['framelen'] = int(val)
                else:
                    raise RuntimeError('error')
            else:
                values = l.split()
                nbin  = int(values[0])
                power = float(values[1])
                if mode[0] == 'kids':
                    kids.append(nbin)
                    powers[nbin] = power
                else:
                    blinds.append(nbin)
                    powers[nbin] = power
    return info, np.array(kids), np.array(blinds), powers


def read_fits(fname):
    """Read fits data of TOD.

    Returns a OrderedDict of md.FixedData's, keyed by bin number.
    It is sorted by bin number, in lowest-first order.
    """

    kSampleRate = 2e9
    hud = pyfits.open(fname)
    bintable = hud[1]

    fftgain  = bintable.header['fftgain']
    framert  = bintable.header['framert']
    nbins    = bintable.header['nbins']
    npoints  = bintable.header['npoints']
    lofreq   = bintable.header['lofreq']

    bins     = np.array([_to_nbit_signed(bintable.header['BIN%d' % i], npoints)
                         for i in range(nbins)])
    if_freq      = kSampleRate * bins / 2**npoints
    carrier_freq = if_freq + lofreq

    from collections import OrderedDict
    data = OrderedDict()

    info = dict()
    info['bins']   = bins
    info['freqs']  = carrier_freq
    info['header'] = bintable.header
    bindata = bintable.data
    cols = hud[1].columns

    rawdata = bindata['data']
    IQdata  = rawdata / fftgain
    rows, cols = IQdata.shape
    indices = np.arange(rows)
    Ts      = bindata['timestamp']
    for i, (b, f) in enumerate(zip(bins, carrier_freq)):
        I = IQdata[:, 2*i    ]
        Q = IQdata[:, 2*i + 1]
        d = md.FixedData(f, Ts, I, Q)
        d.info = info
        data[b]  = d
    hud.close()
    result = OrderedDict(sorted(data.iteritems(), key=lambda x: x[0])) # sort by bin
    return result

def read_fits_single(fname, ind):
    """Read 1 kid TOD data from fits file.

    Returns a md.FixedData.
    """

    kSampleRate = 2e9
    hud = pyfits.open(fname)
    bintable = hud[1]

    fftgain  = bintable.header['fftgain']
    framert  = bintable.header['framert']
    nbins    = bintable.header['nbins']
    npoints  = bintable.header['npoints']
    lofreq   = bintable.header['lofreq']

    bins     = np.array([_to_nbit_signed(bintable.header['BIN%d' % i], npoints)
                         for i in range(nbins)])
    if_freq      = kSampleRate * bins / 2**npoints
    carrier_freq = if_freq + lofreq

    from collections import OrderedDict

    info = dict()
    info['bins']   = bins
    info['freqs']  = carrier_freq
    info['header'] = bintable.header
    bindata = bintable.data
    cols = hud[1].columns

    rawdata = bindata['data']
    # IQdata  = rawdata / fftgain
    rows, cols = IQdata.shape
    indices = np.arange(rows)
    Ts      = bindata['timestamp']

    i = list(bins).index(ind)
    if i is None:
        raise RuntimeError('carrier of bin number %d Not found' % ind)
    b = bins[i]
    f = carrier_freq[i]
    I = IQdata[:, 2*i    ]
    Q = IQdata[:, 2*i + 1]
    d = md.FixedData(f, Ts, I, Q)
    d.info = info
    data  = d
    hud.close()
    return data


def read_localsweep(sweepfname, kidslistfname=None, framelen=None):
    """Read local sweep file.

    Returns a OrderedDict of md.SweepData's, keyed by bin number.
    It is sorted by bin number, in lowest-first order.
    """

    sweep    = _read_sweep(sweepfname)
    
    from collections import OrderedDict
    lofreqs, bins, data = sweep
    bins = np.array(bins)

    if not framelen:
        kidslist = read_kidslist(kidslistfname)
        info, kids, blinds, powers  = kidslist
        framelen = info['framelen']

    sweeps = OrderedDict()
    dfreq    = 2e9 / (2**framelen)

    for b, d in zip(bins, data.T):
        swpdata = md.SweepData()
        super(md.SweepData, swpdata).__init__('I-Q', (np.real(d), np.imag(d)))
        swpdata._Hz = lofreqs + b * dfreq
        sweeps[b] = swpdata

    result = OrderedDict(sorted(sweeps.items(), key=lambda x: x[0])) # sort by bin
    return result


### private functions


def _read_sweep(fname):
    """Read local sweep file.

    Return (lofreqs, bins, data)
      where lofreqs is an 1D array of frequency in Hz;
            bins is an 1D array of bin indices;
            data is an 2D array of complex demodulated amplitudes,
            one row per LO frequency, one column per bin.
    """
    rawdata = np.loadtxt(fname)
    nrow, ncol = rawdata.shape
    
    #print nrow, ncol
    bins = list( map(int, rawdata[0, 1::3]) )
    data     = 1.0 * rawdata[:, 2::3] + 1.0j * rawdata[:, 3::3]
    lofreqs  = rawdata[:, 0]*1e6
    framelen = 2**16
    return lofreqs, bins, data


def _to_nbit_signed(x, n):
    if x > 2**(n-1):
        return -((~x & (2**n-1))+1)
    else:
        return x
