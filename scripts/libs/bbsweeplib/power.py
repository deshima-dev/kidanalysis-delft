"""Power and NEP calculation"""
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

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

@memoize
def getfilterform(filter_name, minfrequency, plot_filters=False):
    """get filter form. This is a port from matlab code in 'blackbody_10.m'.
    
    Comments in matlab code follows:
    % This function uses the "method.filter" component of the "method" struct to
    % determine the used filters. It then loads their filter characteristics
    % from file. And returns the cell-array "filterform" of length N. Here N is
    % the number of filters in the setup. Each cell contains an 2xM_i array with
    % [1/lambda filtertransmission]. NB: at the lowest F point a single
    % datapoint is patched to all filters: 0 for BPF/HPF and 1 for LPF. In the
    % integration interplation will be used between the last real datapoint and
    % this point."""
    c = 2.9998e8
    filterpath = '../data/NiceMuxResults201503/detail/filterfiles/'

    mininvwave = c / minfrequency # minimum frequency as 1/lambda
    
    def to_invm(filt, firstpoint, firstvalue):
        invlambda    = filt.T[0]*100 # from [/cm] to [/m]
        transmission = filt.T[1]
#         invlambda    = np.append(invlambda, lastpoint)
#         transmission = np.append(transmission, lastvalue)

        invlambda    = np.concatenate(([firstpoint], invlambda))
        transmission = np.concatenate(([firstvalue], transmission))

        return array([invlambda, transmission])
        
    if filter_name == '850 GHz':
        K1979 = np.loadtxt(filterpath + 'K1979 855GHz BPF.dat')
        K1981 = np.loadtxt(filterpath + 'K1981 1140GHz LPF.dat')
        K1980 = np.loadtxt(filterpath + 'K1980 990GHz LPF.dat')
        B588  = np.loadtxt(filterpath + 'B588 HPF.dat')
        
        #print len(K1979), len(K1981), len(K1980), len(B588)

        if plot_filters:
            plt.plot(*K1979.T, label='K1979')
            plt.plot(*K1981.T, label='K1981')
            plt.plot(*K1980.T, label='K1980')
            plt.plot(*B588.T, label='B588')

            plt.grid()
            plt.title('w/o mininvwave point')
            plt.xlabel('1/$\lambda$ [1/cm]')
            plt.ylabel('Transmission')
            plt.legend(loc='best')
            plt.show()

        K1979 = to_invm(K1979, mininvwave, 0) # BPF
        K1981 = to_invm(K1981, mininvwave, 1) # LPF
        K1980 = to_invm(K1980, mininvwave, 1) # LPF
        B588  = to_invm(B588,  mininvwave, 0) # HPF
        
        if plot_filters:
            plot(*K1979, label='K1979')
            plot(*K1981, label='K1981')
            plot(*K1980, label='K1980')
            plot(*B588, label='B588')
            grid()
            title('w/ mininvwave point')
            xlabel('1/$\lambda$ [1/m]')
            ylabel('Transmission')
            show()

        filterform = []
        # Filters on the 3K blackbody
        filterform.append(K1980)
        filterform.append(B588)
        filterform.append(K1979)
        # Filters on the 100mK outer box
        filterform.append(K1981)
        filterform.append(K1979)
        # Filters on the 100mK sample
        filterform.append(K1980)
        filterform.append(B588)
        filterform.append(K1979)

        return filterform
    else:
        raise RuntimeError('Unknown Filter')
        
# getfilterform('850 GHz', 9.12e+10, True)

def calc_filter(freq, aperture='2mmlens_3mm_17mm_850_GHz', filter='850 GHz', verbose_plot=0):
    """calc transmission for each frequency

    Input:
      freq: a 1-D array of frequency"""
    c = 2.9998e8
    filterform = getfilterform(filter, min(freq))
    filt = ones(len(freq))

    for invlambda, trans in filterform:
        #check invlambda is increasing
        assert np.all(np.diff(invlambda) > 0)

        # do linear interpolation, instead of pchip method of matlab code.
        intp = interp(freq/c, invlambda, trans)
        if verbose_plot >= 1:
            plot(freq/1e12, intp)
        filt *= intp
    if verbose_plot >= 1:
        plot(freq/1e12, filt, 'r', label='integrated')
        xlabel('Frequency [THz]')
        ylabel('Transmission')
        legend(loc='best')
        xlim(0, 4)

        grid()
        show()
    return filt



#### Power calculation
def occupation(nu, T_BB):
    c  = 2.9998e8
    h  = 6.626068e-34
    kB = 1.3806503e-23

    return 1/(exp(h*nu/(kB*T_BB)) - 1)

def blackbody_brilliance(nu, T_BB, n_pol=1):
    c  = 2.9998e8
    h  = 6.626068e-34
    kB = 1.3806503e-23
    
    return n_pol*h*nu**3/c**2/(exp(h*nu/(kB*T_BB))-1)

def calc_BB_power(freq, dfreq, T_BB, Etendue = 4.2651e-8,
                  aperture='2mmlens_3mm_17mm_850_GHz', filter='850 GHz', verbose_plot=0):
    """calculate power, with etendue and filter considered.

    Parameters:
      freq: a 1-D array of frequency [Hz]
      dfreq: width of each bin (freq[1] - freq[0]) [Hz]
      T_BB: blackbody temperature [K]
      Etendue: throughput [m sr^2]"""
    filt = calc_filter(freq, aperture, filter, verbose_plot)    
    P_BB = blackbody_brilliance(freq, T_BB) # [W / m sr^2 Hz]
    dP_tot = P_BB*Etendue*filt*dfreq # [W per bin]
    return dP_tot

#### NEP calculation
def dNEP2_photon_R(freq, P, Delta, eta_pb=0.57):
    return 2*P*Delta/eta_pb

def dNEP2_photon_Poisson(freq, P, eta_pb=0.57, h=6.626068e-34):
    return 2*P*h*freq

def dNEP2_photon_wave(freq, P, filter, etendu, T_BB, h=6.626068e-34):
    c=2.9998e8
    lmd = c / freq
    return 2*P*h*freq*filter*etendu/lmd**2*occupation(freq, T_BB)

def calc_NEP_photon(temps, # [K]
                    minfrequency=45.6e9*2, maxfrequency=5e12, dfreq=2e9, # [Hz]
                    Etendue = 4.2651e-8, # [m sr^2]
                    aperture='2mmlens_3mm_17mm_850_GHz', filter='850 GHz', verbose_plot=0):
    h=6.626068e-34
    Delta = 45.6 # [Optional, Default = 45.6] value of Delta (half the superconducting gap) in GHz

    Ts = array([])
    Ps = array([])
    NEP2s_R = array([]) # NEP^2 from Recombination
    NEP2s_P = array([]) # NEP^2 from Poisson statistics
    NEP2s_w = array([]) # NEP^2 from photon bunching

    freq = np.arange(minfrequency, maxfrequency, dfreq)
    filt = calc_filter(freq)
    
#     fig1 = figure(1, figsize=(12, 8))
#     fig2 = figure(2, figsize=(12, 8))
    fig1 = figure(1)
    fig3 = figure(3)

    for T_BB in temps:
        P_BB = blackbody_brilliance(freq, T_BB) # [W / m sr^2 Hz]

        dP_tot = P_BB*Etendue*filt*dfreq # [W per bin]
        P_tot  = sum(dP_tot) # [W]

        dNEP2_R = dNEP2_photon_R(freq, dP_tot, Delta*1e9*h)
        dNEP2_P = dNEP2_photon_Poisson(freq, dP_tot)
        dNEP2_w = dNEP2_photon_wave(freq, dP_tot, filt, Etendue, T_BB)

        Ts = append(Ts, T_BB)
        Ps = append(Ps, P_tot)
        NEP2s_R = append(NEP2s_R, sum(dNEP2_R))
        NEP2s_P = append(NEP2s_P, sum(dNEP2_P))
        NEP2s_w = append(NEP2s_w, sum(dNEP2_w))
            
    figure(3)
    NEP2s_tot = NEP2s_R + NEP2s_P + NEP2s_w
    loglog(Ts, sqrt(NEP2s_R), label='NEP, recombination')
    loglog(Ts, sqrt(NEP2s_P), label='NEP, Poisson')
    loglog(Ts, sqrt(NEP2s_w), label='NEP, photon bunching')
    loglog(Ts, sqrt(NEP2s_tot), 'k', label='NEP, Total')
    xlabel('T_BB [K]')
    ylabel('Photon NEP [W/Hz${}^{1/2}$]')
    title('calc NEP')
    legend(loc='best')

    grid()

    figure(1)
    semilogy(Ts, Ps/1e-18, 'o')
    xlabel('Temperature [K]')
    ylabel('Total Incident Power [10^-18 W]')
    grid()
    show()

    return sqrt(NEP2s_tot)

    

### fit responsing with quadratic curve
@memoize
def calc_power_array(minfreq=45.6e9*2, maxfreq=5e12, dfreq=2e9, # [Hz]
                     mintemp=3, maxtemp=33, dtemp = 0.01): # [K]
    "calculate power vs temperature for interpolation"
    Delta = 45.6 # [Optional, Default = 45.6] value of Delta (half the superconducting gap) in GHz
    dfreq          = 2e9 # [Hz]
    maxfrequency   = 5e12 # [Hz]
    minfrequency   = 2*Delta*1e9 # [Hz]
    freq = np.arange(minfrequency, maxfrequency, dfreq)

    pows  = []
    temps = np.arange(3, 33, dtemp)
    for T_BB in temps:
        pows.append(sum(calc_BB_power(freq, dfreq, T_BB)))
    return temps, array(pows)

import fit
def quad_fit_responsivity(tods, swps, kidslist, temperature_datfile, count=None):
    """Fit responsivity of kids and calculate responsivity dx/dPrad.

    Parameters:
      tods: a object of TOD data
      swps: a object of sweep data
      kidslist: a object or kidslist
      temperature_datfile: a string of temperature data file name
      count: a natural number or None. if not None, do calculation for only first `count` KIDS.

    Returns:
      response_ampl:  a function response_ampl(i, P_rad) to calculate dA/dP
                      for KID[i], at radiation power P_rad [W].
      response_angl:  a function response_angl(i, P_rad) to calculate dtheta/dP
                      for KID[i], at radiation power P_rad [W].
      dresponse_ampl: a function response_ampl(i, P_rad) to calculate dA/dP fitting error
                      for KID[i], at radiation power P_rad [W].
      dresponse_angl: a function response_angl(i, P_rad) to calculate dtheta/dP fitting error
                      for KID[i] at radiation power P_rad [W].
    """
    ts, ampls, angls = deglitch_kids(tods, swps, kidslist, slice=slice(count))
    if count is None:
        count = len(ampls)

    ss, Ts = np.loadtxt(temperature_datfile, skiprows=1).T
    temps = np.interp(ts, ss, Ts)
    pows = []
    for T_BB in temps:
        pows.append(interp(T_BB, *calc_power_array()))
    del T_BB
    pows = array(pows)

    print len(pows), 'points'

    x_angls = []
    x_ampls = []
    results_ampl = []
    results_angl = []
    for i in range(count):
        print 'KID[%d]' % i
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        pow0 = np.median(pows)
        factor = 1/pow0 # 1e15
        dp = (pows-pow0)*factor

        def quad(dp, a, b, c):
            y = a*dp**2 + b*dp + c
            return y

        def dquad(dp, a, b, c):
            y = 2*a*dp*factor + b*factor
            return y

        def deltadquad(dp, da, db, dc):
            return np.sqrt((2*dp*da)**2 + (db)**2)*factor

        dadp, a0, _, _, _ = scipy.stats.linregress(pows, ampls[i])
        dpdp, p0, _, _, _ = scipy.stats.linregress(pows, angls[i])

        result_ampl = fit.fit(quad, dp, ampls[i], a=0, b=dadp/factor, c=mean(ampls[i]))
        result_angl = fit.fit(quad, dp, angls[i], a=0, b=dpdp/factor, c=mean(angls[i]))

        plt.plot(pows, ampls[i], 'b.', label='Amplitude')
        plt.plot(pows, quad(dp, **result_ampl.values()), 'c', lw=2, alpha=0.3, label='fit')
        plt.plot(pows, angls[i], 'r.', label='Phase')
        plt.plot(pows, quad(dp, **result_angl.values()), 'm', lw=2, alpha=0.3, label='fit')

        plt.xlabel('Incident Power [W]')
        plt.ylabel('Normalized Response')
        plt.grid()
        plt.legend(loc='best')

        plt.subplot(122)

        plt.errorbar(pows, dquad(dp, **result_ampl.values()), yerr=deltadquad(dp, **result_ampl.errors()), color='c', label=r'$dA/dP_{rad}$')
        plt.errorbar(pows, dquad(dp, **result_angl.values()), yerr=deltadquad(dp, **result_ampl.errors()), color='m', label=r'$d\theta/dP_{rad}$')
        plt.xlabel('Incident Power [W]')
        plt.ylabel('Responsivity [/W]')

        plt.legend(loc='best')
        plt.grid()
        plt.show()

        results_ampl.append(result_ampl)
        results_angl.append(result_angl)

    def response_ampl(i, P_rad):
        dp = (P_rad - pow0)*factor
        return dquad(dp, **results_ampl[i].values())
    def dresponse_ampl(i, P_rad):
        dp = (P_rad - pow0)*factor
        return deltadquad(dp, **results_ampl[i].errors())
    def response_angl(i, P_rad):
        dp = (P_rad - pow0)*factor
        return dquad(dp, **results_angl[i].values())
    def dresponse_angl(i, P_rad):
        dp = (P_rad - pow0)*factor
        return deltadquad(dp, **results_angl[i].errors())

    return response_ampl, response_angl, dresponse_ampl, dresponse_angl


### for check
def plot_fitting(tods, swps, kidslist):
    for i, k in enumerate(sorted(kidslist[1])):
        print 'KID[%d]' % i
        swp = swps[k]
        tod = tods[k]
        
        err = md.get_error_sweep_iq(swp[:10])
        r   = md.fit_onepeak(swp, err)
        fitted = r.fitted(swp.x)
        figure(figsize=(16,4))
        ax1 = subplot(131)
        ax1.set_title('S21 vs Freq')
        plot(swp.x*1e3, swp.db, label='Sweep')
        plot(tod.frequency*1e3 + zeros_like(tod.db), tod.db, '.', label='TOD')
        plot(swp.x*1e3, md.amplitude_to_dB(abs(fitted)), 'y', lw=3, alpha=0.5, label='fit')
        xlabel('Frequency [MHz]')
        ylabel('S21 * const [dB]')
        grid()
        legend(loc='best')
        ax2 = subplot(132)
        ax2.set_title('I vs Q')
        axis('equal')
        plot(swp.i, swp.q, label='Sweep')
        plot(tod.i, tod.q, '.', alpha=0.2, label='TOD')
        plot(real(fitted), imag(fitted), 'y', lw=3, alpha=0.5, label='fit')
        xlabel('I')
        ylabel('Q')
        grid()
        axhline(color='k')
        axvline(color='k')
        legend(loc='best')
        ax3 = subplot(133)
        ax3.set_title('rewind')
        axis('equal')
        rw_s = r.rewind(swp.x, swp.iq)
        rw_t = r.rewind(tod.frequency, tod.iq)
        rw_f = r.rewind(swp.x, r.fitted(swp.x))
        md.plot_iq(rw_s, label='Sweep')
        md.plot_iq(rw_t, '.', alpha=0.2, label='TOD')
        md.plot_iq(rw_f, 'y', lw=3, alpha=0.5, label='fit')
        grid()
        xlabel('Re')
        ylabel('Im')
        axhline(color='k')
        axvline(color='k')
        legend(loc='best')
        show()
