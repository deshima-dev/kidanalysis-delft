import numpy as np
import scipy
import matplotlib.pyplot as plt

import fit

from .misc import MKIDDataException

def oddify(n):
    return int(np.ceil(n/2.0)*2+1)

def find_peaks(freq, ampl, fc, smooth=None, threshold=None, minq=10000, maxratio=0.5, mindist=5e-3, plot = False):
    """
    Search peaks in data.
    """

    dfreq = freq[1] - freq[0]
    if smooth is None:
        #smooth = 15#sigma*0.3
        #smooth = 50
#        smooth = 30
        smooth = 15
        #print( smooth )

    if hasattr(scipy.signal, 'savgol_filter'):
        from scipy.signal import savgol_filter
    else:
        savgol_filter = savgol
    deriv2 = savgol_filter(ampl, oddify(smooth), 3, 2, delta=dfreq)
    ampl_s = savgol_filter(ampl, oddify(smooth), 3, 0, delta=dfreq)

    if threshold is None:
        d2sigma   = np.std(deriv2)
        d2sigma   = np.std(deriv2[np.abs(deriv2)<2*d2sigma])
        #threshold = 5*d2sigma
        #threshold = 4*d2sigma
        threshold = 3*d2sigma

    if plot == 'all':
        fig = plt.figure(1, figsize=(12,4))
        ax1 = fig.add_subplot(211)
        # fig1, ax1 = plt.subplots()
        ax1.plot(freq, ampl)
        if fc>=freq[0] and fc<=freq[-1]:
            ax1.plot(fc, np.interp(fc, freq, ampl), 'go')
        ax2 = fig.add_subplot(212)
        ax2.plot(freq, deriv2, '-')
        ax2.plot(freq, d2sigma + np.zeros_like(freq), '-.')
        ax2.plot(freq, threshold + np.zeros_like(freq), '-')
    elif plot:
        # fig = plt.figure()
        ax1 = plt.subplot(111)
        # fig1, ax1 = plt.subplots()
        ax1.plot(freq, ampl)
        if fc>=freq[0] and fc<=freq[-1]:
            ax1.plot(fc, np.interp(fc, freq, ampl), 'go')

    # collect all peaks
    kid_indices = []
    for k in np.where(deriv2 > threshold)[0]:
        if k < len(deriv2)-1 and deriv2[k-1] <= deriv2[k] and deriv2[k] >= deriv2[k+1]:
            kid_indices.append(k)

    # traverse to zero-crossing
    kids = []
    nbadq = 0
    nbadd = 0
    if not kid_indices:
        return []

    for k in kid_indices:
        l, r = k, k
        while l > 0           and deriv2[l] > 0: l -= 1
        while r < len(deriv2) and deriv2[r] > 0: r += 1
        w = (r - l + 1) * dfreq
        w = w * 6.0 / np.sqrt(3) # convert to FWHM
        l = int((l-k) * 6.0 / np.sqrt(3) + k)
        r = int((r-k) * 6.0 / np.sqrt(3) + k)
        if l < 0:
            l = 0
        if r >= len(freq):
            r = len(freq) - 1

        q0 = freq[k] / w
        f1, q1, d1, bg = fitLorentzian(freq[l:r], ampl[l:r], freq[k], q0)
        ##### refitting by extending fit range
        if (bg-d1)/bg > maxratio:
            if l-10 >= 0 and r+10 < len(freq):
                f1, q1, d1, bg = fitLorentzian(freq[l-10:r+10], ampl[l-10:r+10], freq[k], q0)
                print( 'rough refitting with Lorentzian...' )
        #####
        if plot == 'all':
            x = freq[l:r]
            y = ampl[l:r]
            a = bg
            b = -d1
            d = f1
            c = abs(q1*2.0/d)
            #print( array([a,b,c,d]) )
            fity = a + b / (((x-d)*c)**2 + 1)
            ax1.plot(x, a + np.zeros_like(x), '--k')

            if q1 < minq:
                nbadq += 1
                ax1.plot(x, fity, 'r--')
                continue
            if (bg-d1)/bg > maxratio:
                nbadd += 1
                ax1.plot(x, fity, 'r--')
                continue
            ax1.plot(x, fity, 'r')
        else:
            if q1 < minq:
                nbadq += 1
                continue
            if (bg-d1)/bg > maxratio:
                nbadd += 1
                continue

        kids.append((f1, q1, d1, bg))
    del l, r, f1, q1, d1, bg
    if nbadq > 0:
        print( 'removed', nbadq, 'peaks with bad Q' )
    if nbadd > 0:
        print( 'removed', nbadd, 'peaks with bad S21min' )
    # sort by frequency
    kids.sort()

    ## pick up a peak which is closest to the carrier frequency (when fc in freq range)
    if len(kids)>0 and fc>=freq[0] and fc<=freq[-1]:
        idx = np.argmin( abs(np.array(kids).T[0]-fc) )
        #print( np.array(kids).T[0] )
        #print( kids )
        #print( len(kids), idx )
        kids = [kids[idx]]
        #print( len(kids) )
    #else:
    #    ## eliminate too close peaks
    #    nkill = 0
    #    if mindist > 0:
    #        # remove close kids weaker than preceding kids
    #        p = 0
    #        while p + 1 < len(kids):
    #            f0, q0, _, _ = kids[p]
    #            f1, q1, _, _ = kids[p+1]
    #            if f1 - f0 < mindist and q0 > q1:
    #                del kids[p+1]
    #                nkill += 1
    #            else:
    #                p += 1
    #        # remove close kids weaker than following kids
    #        p = len(kids)-1
    #        while p - 1 >= 0:
    #            f0, q0, _, _ = kids[p]
    #            f1, q1, _, _ = kids[p-1]
    #            if f0 - f1 < mindist and q0 > q1:
    #                del kids[p-1]
    #                nkill += 1
    #                p -= 1
    #            else:
    #                p -= 1

    for i, (f, q, depth, bg) in enumerate(kids):
        f0ind = np.argmin(abs(freq - f))
        w     = f/q
        dl    = w/2.0/dfreq
        dr    = w/2.0/dfreq
        bg_l  = ampl_s[max(int(f0ind - 3*dl), 0)]
        if int(f0ind-3*dl)<0:
            bg_r = ampl_s[0]
        else:
            bg_r  = ampl_s[min(int(f0ind - 3*dl), len(freq) - 1)]
        a_off = (bg_l + bg_r)/2.0
        a_on  = ampl_s[f0ind]
        kids[i] = {'Q': q, 'f0': f, 'f0ind': f0ind,
                  'dl': int(dl), 'dr': int(dr),
                   'a_off': a_off, 'a_on': a_on}
    return kids

#def search_peaks(data, fc, Q_search=10000, S21min=10**(-0.05), plot = False): # backward compat
def search_peaks(data, fc, Q_search=100, S21min=1, plot = False): # backward compat
    return find_peaks(data.x, data.amplitude, fc, minq=Q_search, maxratio=S21min, plot=plot)

#def search_peak(data, Q_search=10000, S21min=0.5, plot = False):
def search_peak(data, Q_search=100, S21min=1, plot = False):
    peaks = find_peaks(data.x, data.amplitude, fc=-1., minq=Q_search, maxratio=S21min, plot=plot)
    if not peaks:
        raise MKIDDataException('peak find failure')
    center  = (data.x[0] + data.x[-1])/2.0
    minind  = 0
    mindist = abs(peaks[0]['f0'] - center)

    for i in range(1, len(peaks)):
        if mindist > abs(peaks[i]['f0'] - center):
            minind  = i
            mindist = abs(peaks[i]['f0'] - center)
    # print( peaks, center, minind, mindist )
    return peaks[minind]

def fitLorentzian(freq, ampl, f0, q0):
    """
    Fit data with lorenzian curve.

    :param freq: a 1-D array of frequency
    :param ampl: a 1-D array of amplitude
    :param f0: initial parameter for center frequency
    :param q0: initial parameter for quality factor

    :return: (fc, q, d, bg)

    - **fc**: fit of center frequency
    - **q**: fit of quality factor
    - **d**: fit of amplitude for Lorentzian curve
    - **bg**: fit of constant background level
    """
    def f(x):
        (a, b, c, d) = x # background, amplitude, 2/FWHM, freq. center
        y = a + b / (((freq-d)*c)**2 + 1)
        return y - ampl
    def fprime(x):
        (a, b, c, d) = x
        g = np.zeros((len(freq)), 4)
        g[:, 0] = 1.0
        g[:, 1] = 1.0 / (1.0 + ((freq - d)*c)**2)
        g[:, 2] = -2.0 * b * c * (freq - d)**2 / (1.0 + ((freq - d)*c)**2)**2
        g[:, 3] =  2.0 * b * c**2 * (freq - d) / (1.0 + ((freq - d)*c)**2)**2

    a = np.median(ampl)
    b = -0.8 * a
    c = 2.0 * q0 / f0
    d = f0
    x0 = np.array([a, b, c, d])
    x1 = scipy.optimize.leastsq(f, x0)[0]
    (a, b, c, d) = x1
    #print( x1 )
    f = d
    q = abs(c * d / 2.0)

    return (f, q, -b, a)


#### savgol filter function for environment with old version of scipy

def savgol(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1):
    from scipy.linalg import lstsq
    from math import factorial
    from scipy.ndimage import convolve1d
    cval = x[0]
    def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0):
        halflen, rem = divmod(window_length, 2)
        pos = halflen
        x = np.arange(-pos, window_length - pos)
        if rem == 0: raise ValueError('window_length must be odd')
        A = x[::-1] ** np.arange(polyorder + 1).reshape(-1, 1)
        y = np.zeros(polyorder + 1)
        y[deriv] = factorial(deriv) / (delta ** deriv)
        coeffs, _, _, _ = lstsq(A, y)
        return coeffs

    x = np.asarray(x)
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    y = convolve1d(x, coeffs, axis=axis, mode="reflect", cval=cval)

    return y


### search glitch
def _numdif_2(y, dx):
    return (y[2:]+y[:-2]-2*y[1:-1])/dx**2

def _clusterize_indices(indices, threshold):
    """
    Fill small gap (within `threshold`) in indices.

    e.g.
     If threshold == 1,
      [True, False, False, True, True] => [True, False, False, True, True].
     If threshold == 2,
      [True, False, False, True, True] => [True, True, True, True, True].


    Parameters:
        indices:   an 1-D array of Boolean
        threshold: an interger, allowed gap between True's in indices
    """

    results = np.copy(indices)
    prev  = 0
    first = True
    for i, x in enumerate(results):
        if x:
            if (not first) and i - prev <= threshold + 1:
                for j in range(prev, i):
                    results[j] = True
            prev  = i
            first = False
    return results

def open_indices(indices, size):
    """
    extend regions of True in given array.

    e.g.
     If size == 1,
      [False, True, True, False, False, False, False, False, True]
         => [False, True, True, True, False, False, False, True, True].
     If size == 2,
      [False, False, True, False, False, False, False, False, True]
         => [True, True, True, True, True, False, True, True, True].



    :param indices:   an 1-D array of Boolean
    :param size: an interger, count to extend True region in `indices`
    """
    results = [True in indices[:size]]*size
    for i in range(size, len(indices)-size):
        # print( i, len(indices), indices[i-size:i+size+1], '->', True in indices[i-size:i+size+1] )
        if True in indices[i-size:i+size+1]:
            results.append(True)
        else:
            results.append(False)
    results.extend([True in indices[-size:]]*size)
    return np.array(results)

def close_indices(indices, size):
    """
    extend regions of False in given array.
    equivalent to ~open_indices(~indices, size)

    :param indices:   an 1-D array of Boolean
    :param size: an interger, count to extend True region in `indices`
    """
    results = [all(indices[:size])]*size
    for i in range(size, len(indices)-size):
        # print( i, len(indices), indices[i-size:i+size+1], '->', True in indices[i-size:i+size+1] )
        if all(indices[i-size:i+size+1]):
            results.append(True)
        else:
            results.append(False)
    results.extend([all(indices[-size:])]*size)
    return np.array(results)

def count_indices_cluster(indices, target=True):
    """
    count `target` in `indices`, but contiguous `target` is counted once.

    :param indices: an 1-D array
    :param target: an scalar to search in `indices`
    """
    count = 0
    prev  = indices[0]
    if prev == target:
        count += 1

    first = True
    for x in indices:
        if x == target and prev != target:
            count += 1
        prev = x
    return count

def indices_to_slices(indices, target=True):
    """
    search `target` in `indices`, and return list of slice where `target` is contiguous.

    :param indices: an 1-D array
    :param target: an scalar to search in `indices`
    """
    if target == indices[0]:
        beg = 0
        prev = indices[0]
    else:
        beg = None
        prev = indices[0]

    slices = []
    for i, x in enumerate(indices):
        if x == target:
            if prev == target:
                pass
            else:
                beg = i
        else:
            if prev == target:
                end = i
                slices.append(slice(beg, end))
            else:
                pass
        prev = x
    if x == target:
        slices.append(slice(beg, None))
    return slices

def clustering(indices, numbers, window, cnt_thre=30):
    """
    Clustering function using multiplicity information.
    If multiplicity is smaller than cnt_thre at a event, not necessary to add time window to that event

    Parameters:
    :param indices:   an 1-D array of Boolean
    :param numbers:   an 1-D array of multiplicity
    :param window:    an interger, time window size after high-multiplicity event
    :param cnt_thre:  threshold for multiplicity
    """
    results = np.copy(indices)

    slices = indices_to_slices(indices)
    for i, s in enumerate(slices):
        multi = numbers[s]

        if multi.sum()>cnt_thre:
            beg = s.start
            if s.stop is None:
                end = None
            else:
                end = s.stop + window
                if end>len(results): end = None

            results[slice(beg, end)] = True

    return results

def tod_filt(dt, phase, flow=None, fhigh=100., opt_print=False):
    """
    Apply FIR filter to time trace data.

    :param dt: data duration b.w. each sampling point [s]
    :param phase: an array of 1-D arrays of data to deglitch
    :param flow: low cut frequency for FIR filter [Hz]
    :param fhigh: high cut freqeucny for FIR filter [Hz]
    :param opt_print: bool to choose whether to print messages or not
    
    :return: an 1-D array of filtered data
    :return: a delay value in filtered data [s]
    """
    #####
    ##### FIR filter   
    nyq = 1./dt /2.
    flag_lpf = False
    if flow is None:
        flag_lpf = True
        fe1 = np.nan
    else:
        fe1 = flow/nyq # lower cut off
    fe2 = fhigh/nyq # upper cut off
    if fe2>1.: fe2 = 0.999
    if opt_print:
        print()
        print( 'tod_filt::', len(phase), nyq, fe1, fe2 )

    size = 2**int( np.floor( np.log2(len(phase)) ) ) 
    divide = 512 - 1
    
    #numtaps = len(phase)
    step = len(phase[:size])/(divide+1)
    numtaps = step*2

    from scipy import signal
    if flag_lpf:
        b = signal.firwin(numtaps, fe2)
    else:
        b = signal.firwin(numtaps, [fe1, fe2], pass_zero=False)
        
    phase_filt = signal.lfilter(b, 1, phase)
    delay = (numtaps-1)/2*dt
    if opt_print:
        print( 'tod_filt::', size, divide, numtaps, dt, delay )
        print()
    
    #idx = np.where(ts<2*delay)[0]
    #phase_filt[idx] = 0.    
    phase_filt[:int(delay/dt)*2] = np.zeros( int(delay/dt)*2 )
    
    return phase_filt, delay

def find_glitch(yss,
                baseline_thresh = 6.0, glitch_thresh = 5.0, clusterize_thresh = 2, offset = 0,
                opt_diff2=True, smrange=1, returnval=False):
    """
    find glitches common to `yss`, assuming glitch exists at the same
    time of all data in `yss`. Too close glitches or broad glitch are
    treated as one glitch.

    :param yss: an array of 1-D arrays of data to deglitch
    :param baseline_thresh: threshold to decide baseline
    :param glitch_thresh: threshold to decide as glitch 
    :param clusterize_thresh: if gap between glitches are less than or equal to this, treat them as one glitch.
    :param offset: an integer offset added to returned bad array
    :param opt_diff2: bool to select whether to use 2nd derivative for glitch identification
    :param smrange: an integer to set range of rolling mean (used for glitch identification based on std)
    :param returnval: bool to choose whether to return threshold value or not
    
    :return: an 1-D array of boolean, that can be used as index of ys (elements of yss).
    :return: (option) threshold value for glitch identification (return when returnval=True)
    """
    ave = np.average(yss, axis=0)
    xs  = np.arange(len(yss[0]))
    dx  = xs[1] - xs[0]
    if opt_diff2:
        diff2 = np.array( _numdif_2(ave, dx) )
        sigma = np.std(diff2)
        good  = (np.abs(diff2) < (baseline_thresh*sigma))
        sigma = np.std(diff2[good])
        bad   = (np.abs(diff2) >= (glitch_thresh*sigma))
        
        if returnval:
            thre = glitch_thresh*sigma/np.sqrt(6)*(dx**2)
    else:
        #####
        ##### rolling mean
        weights = np.ones(smrange)/smrange
        assert( len(ave)>len(weights) )
        smooth  = np.convolve(ave, weights, 'same')
        
        invalid = int( (len(weights)-1)/2 )
        smooth[:invalid] = np.mean(ave[:invalid]) * np.ones( len(ave[:invalid]) )
        smooth[-invalid-1:] = np.mean(ave[-invalid-1:]) * np.ones( len(ave[-invalid-1:]) )
        
        ave_sm = ave - smooth
        
        mean = np.mean(ave_sm)
        sigma = np.std(ave_sm)
        good  = (np.abs(ave_sm-mean) < (baseline_thresh*sigma))
        #good  = (np.abs(ave_sm) < (baseline_thresh*sigma))
        mean = np.mean(ave_sm[good])
        sigma = np.std(ave_sm[good])
        bad   = (np.abs(ave_sm-mean) >= (glitch_thresh*sigma))
        #bad   = (ave_sm-mean >= (glitch_thresh*sigma)) # can be used only for phase deglitch
        
        if returnval:
            thre = glitch_thresh*sigma

    ## treat broad glitch (or too close glitches) as one glitch
    bad   = _clusterize_indices(bad, clusterize_thresh)

    if opt_diff2:
        bad = np.concatenate( ([bad[0]], bad, [bad[-1]]) )

    bad_ = np.copy(bad)
    if offset>0:
        for i,b in enumerate(bad):
            if b:
                bad_[i] = True
                for j in range(offset):
                    if (i+j+1)<len(bad): bad_[i+j+1] = True

    if returnval:
        return bad_, thre
    else:
        return bad_

def find_glitch_advanced(phase, setting, opt='raw'):
    """
    find glitches in phase.
    Too close glitches or broad glitch are treated as one glitch.

    :param phase: an 1-D array of data to deglitch
    :param setting: a dictionary to include following keys:
    - baseline_thresh, glitch_thresh, clusterize_thresh, interp_offset to be used in fing_glitch_opt
    - dt (data duration b.w. each sampling point [s]), smtime (time for smoothing [s])
    - flow (low cut frequency for tod filter, not used), tau_qp (recombination time used for high cut freqeuncy for tod filter)
    :param opt: 'raw' (use raw tod for deglitch) or 'fir' (use 'fir' filtered tod for deglitch) or 'both' (combine both)
    
    :return: a dictionary including following keys:
    - bad_raw, thre_raw (opt='raw' or 'both')
    - bad_fir, thre_fir, phase_fir, delay (opt='fir' or 'both')
    - bad_both (opt='both')
    """
    #####
    ##### deglitch setting
    baseline_thresh = setting['baseline_thresh']
    glitch_thresh = setting['glitch_thresh']
    clusterize_thresh = setting['clusterize_thresh']
    interp_offset = setting['interp_offset']
    
    dt     = setting['dt'] # sec
    smtime = setting['smtime'] # sec
    smrange = int( np.floor(smtime/dt) ) # number of points to smooth over
        
    #flow = setting['flow'] # Hz, not used
    fhigh = 1e+3 # Hz
    if setting.has_key('fhigh'):
        fhigh = setting['fhigh']

    ##### tau_qp is the highest priority to determine fhigh
    if setting.has_key('tau_qp'):
        tau_qp = setting['tau_qp'] # sec
        fhigh = 1./(2.*np.pi*tau_qp)
        #fhigh = 2./(2.*np.pi*tau_qp)
    
    if opt=='raw' or opt=='both':
        bad, thre = find_glitch([phase], baseline_thresh, glitch_thresh,
                                clusterize_thresh, interp_offset,
                                opt_diff2=True, smrange=1, returnval=True)

    if opt=='fir' or opt=='both':
        #####
        ##### TOD filter, lower cut off = 10 Hz, upper cut of = recombination time
        phase_filt_, delay = tod_filt(dt, phase-np.mean(phase), fhigh=fhigh, opt_print=False)
        
        phase_filt = np.zeros(len(phase_filt_))
        #idxtmp = np.where(ts<delay)[0]
        #phase_filt[:-idxtmp[-1]-1] = phase_filt_[idxtmp[-1]+1:]
        #print( idxtmp[-1], int(delay/dt), (numtaps-1)/2 )
        
        phase_filt[:-int(delay/dt)-1] = phase_filt_[int(delay/dt)+1:]
        #phase_filt[:-int(delay/dt)-2] = phase_filt_[int(delay/dt)+2:]
        
        bad_filt, thre_filt = find_glitch([phase_filt], baseline_thresh, glitch_thresh,
                                          clusterize_thresh, interp_offset,
                                          opt_diff2=False, smrange=smrange, returnval=True)
    
    if opt=='both':
        #####
        ##### combine bad and bad_filt
        #bad_both = bad | bad_filt
        bad_both = bad & bad_filt
    
    out = {}
    if opt=='raw':
        out['bad_raw'] = bad; out['thre_raw'] = thre
        
        return out
    elif opt=='fir':
        out['bad_fir'] = bad_filt; out['thre_fir'] = thre_filt
        out['phase_fir'] = phase_filt; out['delay'] = delay
        
        return out
    elif opt=='both':
        out['bad_raw'] = bad; out['thre_raw'] = thre
        out['bad_fir'] = bad_filt; out['thre_fir'] = thre_filt
        out['bad_both'] = bad_both
        out['phase_fir'] = phase_filt; out['delay'] = delay
        
        return out
    else:
        print( 'There is no such option: ', opt )
        return( out )

def find_glitch_both(ampl, phase, setting):
    """
    find glitches both in amplitude and phase.
    Too close glitches or broad glitch are treated as one glitch.

    :param ampl: an 1-D array of amplitude data to deglitch
    :param phase: an 1-D array of phase data to deglitch
    :param setting: a dictionary to include following keys:
    - baseline_thresh, glitch_thresh, clusterize_thresh, interp_offset to be used in fing_glitch_opt
    - dt (data duration b.w. each sampling point [s]), smtime (time for smoothing [s])
    
    :return: a dictionary including following keys:
    - bad_ampl, thre_ampl, bad_phase, thre_phase, bad_both
    """
    
    ##### deglitch setting
    baseline_thresh = setting['baseline_thresh']
    glitch_thresh = setting['glitch_thresh']
    clusterize_thresh = setting['clusterize_thresh']
    interp_offset = setting['interp_offset']
    
    dt     = setting['dt'] # sec
    smtime = setting['smtime'] # sec
    smrange = int( np.floor(smtime/dt) ) # number of points to smooth over
        
    opt_diff2 = False
    if opt_diff2:
        ampbad, ampthre = find_glitch([ampl], baseline_thresh, glitch_thresh,
                                      clusterize_thresh, interp_offset+2,
                                      opt_diff2=True, smrange=1, returnval=True)
    else:
        ampbad_, ampthre = find_glitch([ampl], baseline_thresh, glitch_thresh,
                                       clusterize_thresh, interp_offset+3,
                                       opt_diff2=False, smrange=smrange, returnval=True)
    
        ampbad = np.zeros(len(ampbad_), dtype=bool)
        ampbad[:-1] = ampbad_[1:]

    phbad, phthre = find_glitch([phase], baseline_thresh, glitch_thresh,
                                clusterize_thresh, interp_offset,
                                opt_diff2=True, smrange=1, returnval=True)
                                #opt_diff2=False, smrange=smrange, returnval=True)

    ##### combine ampbad and phbad
    bad_both = ampbad & phbad          
    
    out = {}
    out['bad_ampl'] = ampbad
    out['thre_ampl'] = ampthre
    out['bad_phase'] = phbad
    out['thre_phase'] = phthre
    out['bad_both'] = bad_both
    
    return out

def interpolate_bad(ys, bad):
    """
    Linear interpolate `ys` in region `bad`.

    :param ys: an 1-D array
    :param bad: an 1-D array of boolean, with True indicating bad region in `ys`.

    :return: an 1-D array, with bad region interpolated.
    """
    xs  = np.arange(len(ys))
    deglitched = np.interp(xs, xs[~bad], ys[~bad])
    return deglitched

def deglitch(yss, sources=None,
             baseline_thresh = 6.0, glitch_thresh = 5.0, clusterize_thresh = 2,
             debug=False):
    """
    Deglitch `yss` using `sources`, assuming glitch exists at the same
    time of all data in `yss`. Too close glitches or broad glitch are
    treated as one glitch.

    Parameters:
        yss:               an array of 1-D arrays of data to deglitch
        sources:           source data to detect glitch. If None, yss is used
        baseline_thresh:   threshold to decide baseline
        glitch_thresh:     threshold to decide as glitch
        clusterize_thresh: if gap between glitches are less than or equal to this,
                           treat them as one glitch.

        debug:   do plot for debugging

    Return:
        results: an array of 1-D arrays, that is deglitched yss"""

    if sources is None:
        sources = yss
    bad = find_glitch(sources, baseline_thresh, glitch_thresh, clusterize_thresh)

    results = []
    for ys in yss:
        results.append(interpolate_bad(ys, bad))

    return results
