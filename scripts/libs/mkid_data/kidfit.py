import numpy as np
import warnings
import json
import pickle
import collections

import matplotlib.pyplot as plt

import fit
import fit.fitresult

from .misc import MKIDDataException
from . import fitters
from .peak_search import search_peaks
from . import data

class KidFitResult(object):
    def __init__(self, fitresult=None, data = None, fitrange=slice(None)):
        if fitresult is None:
            self._result = fit.fitresult.FitResult()
        else:
            self._result = fitresult
        self.bgfunc      = None
        self.rewindfunc  = None
        self.data        = data
        self.fitrange    = fitrange
        self.info        = dict()
    def fitted(self, x):
        return self.eval(x)
    def add_functions(self, funcdict):
        if 'bgfunc' in funcdict:
            self.bgfunc = funcdict['bgfunc']
        if 'rewindfunc' in funcdict:
            self.rewindfunc = funcdict['rewindfunc']
    def rewind(self, x, iq):
        return self.rewindfunc(x, iq, **self.values())
    def rewind_data(self, d):
        if hasattr(d, 'info'):
            info = d.info
        else:
            info = dict()
        if isinstance(d, data.FixedData):
            ampl, phase = self.rewound_ampl_phase(d.frequency, d.iq)
            return data.FixedFitData(d.frequency, d.t, ampl, phase, info=info)
        if isinstance(d, data.SweepData):
            ampl, phase = self.rewound_ampl_phase(d.x, d.iq)
            return data.SweepFitData(d.x, ampl, phase, info=info)
        else:
            raise RuntimeError('not implemented for %s' % type(d))
    def rewound_ampl_phase(self, x, iq):
        """
        :return: (amplitude, phase) of rewinded x, iq
        """
        rw = self.rewind(x, iq)
        ampl  = 2*np.abs(rw)
        phase = -np.angle(-rw)
        ##### to avoid jump b.w. pi and -pi
        index_ar0 = np.where( abs(phase)>3.1 )
        if len(index_ar0[0])>0:
            index0 = index_ar0[0][0]
            initial = np.mean(phase[0:100])
            if initial>0.:
                index_ar1 = np.where( phase<-3.1 )
                factor = 2.*np.pi
            else:
                index_ar1 = np.where( phase>3.1 )
                factor = -2.*np.pi
            for index in index_ar1[0]:
                if index>index0: phase[index] += factor
        #####
        return ampl, phase
    def bg(self, x):
        return self.bgfunc(x, **self.values())

    def get_x(self, data=None):
        if data is None:
            if self.data is None:
                raise RuntimeError('no sweep data is supplied: you may use get_x(swpdata).')
            data = self.data
        return data.x[self.fitrange()]
    x = property(get_x) # backward compatibility

    def plot(self, data=None, ax1=None, ax2=None, opt=None):
        if data is None:
            if self.data is None:
                raise RuntimeError('no sweep data is supplied: you may use get_x(swpdata).')
            data = self.data
        s = self.fitrange
        p = data
        x = p.x
        I  = p.i
        Q  = p.q
        IQ = p.iq

        # get fitted resonant frequency
        if 'f0' in self.params:
            fitx0 = self.params['f0'].value
        elif 'fr' in self.params:
            fitx0 = self.params['fr'].value
        else:
            fitx0 = None

        fity  = self.fitted(x[s])
        if fitx0:
            fity0 = self.fitted(fitx0)

        ## figure 1: plot amplitude, I, Q vs frequency
        if ax1 is None:
            fig1 = plt.figure()
            ax1  = fig1.add_subplot(111)
        if opt is None:
            ax1.plot(x, abs(IQ),'.g', label='Ampl.')
            ax1.plot(x, I,'.b', label='I')
            ax1.plot(x, Q,'.r', label='Q')
            ax1.plot(x[s], abs(fity),'-y', lw=3)
            ax1.plot(x[s], np.real(fity),'-c', lw=3)
            ax1.plot(x[s], np.imag(fity),'-m', lw=3)
            if fitx0:
                ax1.plot(fitx0, np.abs(fity0), 'r*', ms=15)
            if self.bgfunc:
                ax1.plot(x, abs(self.bg(x)), '-', color='gray', lw=2, label='bg')
        elif opt=='sub':
            alpha = 0.5
            ax1.plot(x, abs(IQ),'g', alpha=alpha)
            ax1.plot(x, I,'b', alpha=alpha)
            ax1.plot(x, Q,'r', alpha=alpha)
            ax1.plot(x[s], abs(fity),'--y')
            ax1.plot(x[s], np.real(fity),'--c')
            ax1.plot(x[s], np.imag(fity),'--m')
            if fitx0:
                ax1.plot(fitx0, np.abs(fity0), 'r*')
            if self.bgfunc:
                ax1.plot(x, abs(self.bg(x)), '--', color='gray')
        if opt is None:
            ax1.set_xlabel('Frequency [GHz]')
            ax1.set_ylabel('Amplitude')
            ax1.grid()
            ax1.axhline(color='k')
            ax1.legend(loc='best')

        ## ax 2: plot on IQ plane
        if ax2 is None:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, aspect='equal')
        if opt is None:
            ax2.plot(I, Q, '.b', label='data')
            ax2.plot(np.real(fity), np.imag(fity), '-c', lw=3, label='fit')
            # ax2.plot(np.real(inity), np.imag(inity), '-y', label='initial guess')
            if fitx0:
                ax2.plot(np.real(fity0), np.imag(fity0), 'r*', ms=15)
            if self.bgfunc:
                ax2.plot(np.real(self.bg(x)), np.imag(self.bg(x)), '-', color='gray', lw=2, label='bg')
        elif opt=='sub':
            alpha = 0.5
            ax2.plot(I, Q, 'b', alpha=alpha)
            ax2.plot(np.real(fity), np.imag(fity), '--c')
            if fitx0:
                ax2.plot(np.real(fity0), np.imag(fity0), 'r*')
            if self.bgfunc:
                ax2.plot(np.real(self.bg(x)), np.imag(self.bg(x)), '--', color='gray')
        if opt is None:
            ax2.set_xlabel('I')
            ax2.set_ylabel('Q')
            ax2.axhline(color='k')
            ax2.axvline(color='k')
            ax2.grid()

        # Put a legend below current axis
        ax2.legend(loc='best')
        return ax1, ax2

    @property
    def fitparamdict(self):
        return self.params.valuesdict()

    def dumps(self, header='', **kws):
        from pprint import pformat
        out = dict()
        out['result']     = json.loads(self._result.dumps())
        out['rewindfunc'] = pickle.dumps(self.rewindfunc)
        out['bgfunc']     = pickle.dumps(self.bgfunc)
        out['fitrange']   = pickle.dumps(self.fitrange)
        out['info']       = pickle.dumps(self.info)

        s = header + '\n\n'
        s += pformat(self.info, **kws) + '\n\n'
        s += self._result.report_str()

        buf = ''
        for l in s.split('\n'):
            buf += ('# ' + l + '\n')
        buf += pformat(out, **kws)
        return buf
        # return json.dumps(out, **kws)

    @classmethod
    def loads(cls, s, **kws):
        r = KidFitResult()
        # dic = json.loads(s, **kws)
        ns = globals()
        ns['inf'] = np.inf
        ns['nan'] = np.nan
        dic = eval(s, ns)
        for k, v in dic.items():
            if k == 'result':
                r._result = r._result.loads(json.dumps(v))
            elif k == 'rewindfunc':
                r.rewindfunc = pickle.loads(v)
            elif k == 'bgfunc':
                r.bgfunc = pickle.loads(v)
            elif k == 'fitrange':
                r.fitrange = pickle.loads(v)
            elif k == 'info':
                r.info = pickle.loads(v)
            else:
                raise RuntimeError('unknown key: %s' % k)
        return r
    def dump(self, fp, **kws):
        return fp.write(self.dumps(**kws))
    @classmethod
    def load(cls, fp, **kws):
        return cls.loads(fp.read(), **kws)

    ## delegate all other methods
    def __getattr__(self, name):
        if name == '_result':
            raise AttributeError()
        return getattr(self._result, name)

    def __getstate__(self):
        return self.dumps()

    def __setstate__(self, s):
        self.__dict__ = self.loads(s).__dict__

def _fitsweep_complex(func, freqs, iqs, dataerror, params):
    """
    Fit a complex-valued function to IQ plane sweep.

    :param func: a complex-valued function of form f(x, param1, param2, ..., paramN) i.e. with the first argument is a independent variable `x`, and the rest are parameters.
    :param freqs: a 1-D array of float, of frequency [GHz]
    :param iqs: a 1-D array of complex data of IQ.
    :param dataerror: two-element tuple (Ierr, Qerr) corresponding to error of I and Q, respectively
    :param params: a lmfit.Parameters object to fit

    :return: a FitResult object.
    """

    convert = complex_to_cartesian2darray

    if dataerror is None:
        errIQ = None
    else:
        err_I = dataerror[0]
        err_Q = dataerror[1]
        errIQ = convert(np.broadcast_to(err_I, iqs.shape) +
                        1j*np.broadcast_to(err_Q, iqs.shape))
    r = fit.fit(func, freqs, convert(iqs), errIQ, params, silent=True, convert=convert)
    return r


def fit_from_params(data, fitrange, dataerror, params, func, names, residue_fun = None):
    """
    Fit IQ data. Fit data using given data, fitrange, dataerror, params.

    :param data: a SweepData to fit
    :param fitrange: a slice or tuple, to specify fit range in `data`
    :param dataerror: two-element tuple (Ierr, Qerr) corresponding to error of I and Q, respectively
    :param params: a lmfit.Parameters object to describe parameter.
    :param func: a complex-valued function of form f(x, param1, param2, ..., paramN) i.e. with the first argument is a independent variable `x`, and the rest are parameters.
    :param names: a list of parameter names of `func`, like ['param1', ..., 'paramN']
    :param resudue_fun: no effect (for backward compativbility)

    :return: a FitResult object.
    """

    if residue_fun:
        warnings.warn('residue_fun is deprecated. if you want, refer fitsweep_complex() or contact author')
    # if 'FWHM' in params:
    #     init_fwhm = params['FWHM'].value

    # cast `s` to slice if it's tuple
    if isinstance(fitrange, slice):
        s = fitrange
    else:
        if isinstance(fitrange[0], np.ndarray): # xrange
            s = slice(fitrange[0][0], fitrange[0][-1]+1)
        else:
            s = slice(*fitrange)
    fitresult = _fitsweep_complex(func, data.x[s], data.iq[s], dataerror, params)
    kidfitresult = KidFitResult(fitresult, data, s)
    return kidfitresult


#### utility functions
def fit_onepeak(data, fc, errors = None, nfwhm = 3, fitter = 'gaolinbg', Q_search=10000, plot=False):
    """
    Fit data with fitter.

    :param data: sweep data to fit
    :param fc: carrier frequency
    :param error: error in data (data format depends on fitter)
    :param nfwhm: fit range [FWHM]. if negative, use all data as fit range
    :param fitter: fitter to use.
    :param Q_search:
    :param plot: 'all'/True/False

    **fitter** supports:

    - **gao**: a complex function from Jiansong Gao's D thesis (default)
    - **gaolinbg**: a complex function from Jiansong Gao's D thesis
    - **mazinrev**: a complex function from Ben Mazin's D thesis
    - **blank**: a complex function only with cable delay effect or a Fitter object or a tuple (refer fitters.py)
    """
    if type(fitter) == str:
        if fitter in fitters.all_fitters:
            fitter = getattr(fitters, 'fitter_' + fitter)
        else:
            raise RuntimeError('if you specify fitter by string, it must be one of %s!' % all_fitters)
    func, guess, names, others = fitter
    params = guess(data)
    if nfwhm < 0:
        s = slice(None)
    else:
        peaks = search_peaks(data, fc, Q_search=Q_search, plot=plot)
        if len(peaks) == 1:
            pdict = peaks[0]
        else:
            raise MKIDDataException("number of peak not 1: %d" % len(peaks))
        s = adjust_fitrange(nfwhm, len(data), len(names), pdict)

    ## do prefitting and fix some parameters
    if 'prefit_and_fix' in fitter[-1]:
        r_prefit = fit_from_params(data, slice(None), errors, params, func, names)
        params   = r_prefit.params
        for k in fitter[-1]['prefit_and_fix']:
            params[k].vary = False

    ## add additional exprs if exsit
    if 'additional_expr' in fitter[-1]:
        for k, v in fitter[-1]['additional_expr'].items():
            params[k] = fit.Parameter(name=k, expr=v)

    r = fit_from_params(data, s, errors, params, func, names)
    r.add_functions(others)
    return r

def adjust_fitrange(nfwhm, ndata, nparams, peakparams, factor=1):
    """
    Adjust fit range to make sure :math:`N_{free} \geq nparam`

    :param nfwhm: a float, minimum fit length in unit of FWHM
    :param ndata: a integer, length of data
    :param nparams: a integer, number of parameter
    :param peakparams: a dict of peak information
    :param factor: a integer

    :Note: *specify 2* as **factor** if you fit complex variable (as 2d-vector)

    **peakparams** are:

    - **f0ind**: a integer, index of peak center
    - **dl**: a integer, index count to reach the left half-maximum
    - **dr**: a integer, index count to reach the right half-maximum

    :return: (rbegin, rend)

    - **rbegin** is a integer of fit range begin
    - **rend** is a integer of fit range end
    """
    f0ind = peakparams['f0ind']
    l, c, r = f0ind-peakparams['dl'], f0ind, f0ind+peakparams['dr']
    if (r - l + 1)*factor >= nparams:
        rbegin, rend = int(c-nfwhm*(c-l)), int(c+nfwhm*(r-c))
    else:
        n = nparams/(float(r - l + 1)*factor)
        rbegin, rend = int(f0ind-nfwhm*n*(c-l)), int(c+nfwhm*n*(r-c))
    if rbegin < 0:
        if rend + (-rbegin) >= ndata:
            rend = ndata-1
            rbegin = 0
        else:
            rend = rend + (-rbegin)
            rbegin = 0
    if rend >= ndata:
        if rbegin - (rend - ndata) < 0:
           rbegin = 0
           rend = ndata-1
        else:
           rbegin = rbegin - (rend - ndata)
           rend = ndata-1
    if (rend - rbegin + 1)*factor < 11:
        raise MKIDDataException("Fit range guess error")
    return rbegin, rend

## converter to fit complex data as 2d vector
def complex_to_cartesian2darray(x):
    x = np.atleast_1d(x)
    shape = x.shape
    return np.concatenate([np.real(x), np.imag(x)], axis=len(shape)-1)

def cartesian2darray_to_complex(x):
    assert x.shape[-1] % 2 == 0
    size = x.shape[-1] / 2
    return x[...,:size] + 1j*x[...,size:]
