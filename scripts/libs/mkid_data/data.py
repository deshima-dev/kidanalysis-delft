# -*- coding: utf-8 -*-
#### data class
"""
Provides MKID data object.

all data class object d (should) provides:
- d.fields() : attribute names of raw data arrays
- d.unpack() : return list of arrays in the same order with fields()
- d[slice_like] : return a new object with all arrays sliced as specified
- d.down_sample(nsample) : average each nsample
- len(d) : length of array (all array inside d must be of the same length)
"""
import sys
import re
import datetime
import collections
import warnings
import gzip
import fnmatch
import copy
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from functools import reduce

from .calibrate_tod import calibrate_with_blind_tones

from .misc import MKIDDataException, down_sample

class Drawer(object):
    """a mix-in class for plot own data."""
    def draw(self, ax=None, xname=None, yname=None, *args, **kws):
        """
        plot data.

        if xname are omitted, data for x is searched from ['x', 't', 'f'].
        if yname are omitted, data for y is searched from ['db', 'amplitude'].
        """
        x_defaults = ['x', 't', 'f']
        y_defaults = ['db', 'amplitude']

        if xname:
            x_data = getattr(self, xname)
        else:
            for n in x_defaults:
                if hasattr(self, n):
                    xname = n
                    x_data = getattr(self, n)
                    break
            else:
                raise RuntimeError('no default data found for x')

        if yname:
            y_data = getattr(self, yname)
        else:
            for n in y_defaults:
                if hasattr(self, n):
                    yname = n
                    y_data = getattr(self, n)
                    break
            else:
                raise RuntimeError('no default data found for y')

        if ax is None:
            ax = plt.gca()

        ax.plot(x_data, y_data, *args, **kws)
        if getattr(self.__class__, xname).__doc__:
            xdoc = getattr(self.__class__, xname).__doc__
            xlabel = ax.get_xlabel()
            if not xlabel:
                ax.set_xlabel(xdoc)
            elif xlabel != xdoc:
                warnings.warn('xlabel "%s" and "%s" not match' % (xdoc, xlabel))
        if getattr(self.__class__, yname).__doc__:
            ydoc = getattr(self.__class__, yname).__doc__
            ylabel = ax.get_ylabel()
            if ylabel:
                ax.set_ylabel(ylabel + ', ' + ydoc)
            else:
                ax.set_ylabel(ydoc)

        return ax

################################################################
# data to be indexed
################################################################

# should have __len__, __getitem__

class KidGlobalResponse(object):
    """
    class to represent response.

    provides `i`, `q`, `iq`, `deg`, `amplitude`, `db`.
    """
    def __init__(self, kind, data):
        """
        :param kind: one of ['dB-DEG', 'I-Q', 'dB']
        :param data: tuple of data, corresponding to kind
        """
        self._kind    = kind
        self._rawdata = data
        if kind == 'dB-DEG':
            self._data = _KidGlobalResponse_dBDEGData(data)
        elif kind == 'I-Q':
            self._data = _KidGlobalResponse_IQData(data)
        elif kind == 'I-Q-Gain':
            self._data = _KidGlobalResponse_IQGainData(data)
        elif kind == 'dB':
            self._data = _KidGlobalResponse_dBData(data)
        else:
            raise MKIDDataException('data format not implemented')
    def fields(self):
        return self._data.fields()
    def unpack(self):
        return [getattr(self, k) for k in self.fields()]
    @property
    def i(self):
        "I"
        return self._data.i()
    @property
    def q(self):
        "Q"
        return self._data.q()
    @property
    def iq(self):
        "IQ"
        return self._data.iq()
    @property
    def deg(self):
        "phase [deg]"
        return self._data.deg()
    @property
    def amplitude(self):
        "S21 in amplitude"
        return self._data.amplitude()
    @property
    def db(self):
        "S21 [dB]"
        return self._data.db()
    def __len__(self):
        return len(self._rawdata[0])
    def xrange(self, beg, end):
        return np.where((beg <= self.x) * (self.x <= end))
    def __getitem__(self, key):
        return KidGlobalResponse(self._kind, [d[key] for d in self._rawdata])

## データフォーマット違いの吸収用クラスたち
class _KidGlobalResponse_IQData:
    def __init__(self, data):
        self._data = dict()
        self._data['I'] = np.array(data[0])
        self._data['Q'] = np.array(data[1])
    def i(self):
        return self._data['I']
    def q(self):
        return self._data['Q']
    def iq(self):
        return self.i() + 1j*self.q()
    def deg(self):
        return np.arctan2(self.q(), self.i()) * 180/np.pi
    def amplitude(self):
        return np.sqrt(self.i()**2 + self.q()**2)
    def db(self):
        return amplitude_to_dB(self.amplitude())
    def fields(self):
        return ('i', 'q')
class _KidGlobalResponse_IQGainData:
    def __init__(self, data):
        self._data = dict()
        self._data['I'] = data[0]
        self._data['Q'] = data[1]
        self._data['gain'] = data[2]
    def i(self):
        return self._data['I'] / self._data['gain']
    def q(self):
        return self._data['Q'] / self._data['gain']
    def iq(self):
        return self.i() + 1j*self.q()
    def deg(self):
        return np.arctan2(self.q(), self.i()) * 180/np.pi
    def amplitude(self):
        return np.sqrt(self.i()**2 + self.q()**2)
    def db(self):
        return amplitude_to_dB(self.amplitude())
    def fields(self):
        return ('i', 'q')
class _KidGlobalResponse_dBDEGData:
    def __init__(self, data):
        self._data = dict()
        self._data['dB']  = np.array(data[0])
        self._data['DEG'] = np.array(data[1])
    def i(self):
        if not 'I' in self._data:
            self._data['I'] = dB_to_amplitude(self.db())*np.cos(self.deg()/180.0*np.pi)
        return self._data['I']
    def q(self):
        if not 'Q' in self._data:
            self._data['Q'] = dB_to_amplitude(self.db())*np.sin(self.deg()/180.0*np.pi)
        return self._data['Q']
    def deg(self):
        return self._data['DEG']
    def iq(self):
        return self.i() + 1j*self.q()
    def amplitude(self):
        return dB_to_amplitude(self.db())
    def db(self):
        return self._data['dB']
    def fields(self):
        return ('db', 'deg')
class _KidGlobalResponse_dBData():
    def __init__(self, data):
        self._data = dict()
        self._data['dB']  = np.array(data[0])
    def i(self):
        raise MKIDDataException('no phase data!')
    def q(self):
        raise MKIDDataException('no phase data!')
    def iq(self):
        raise MKIDDataException('no phase data!')
    def deg(self):
        raise MKIDDataException('no phase data!')
    def amplitude(self):
        return dB_to_amplitude(self.db())
    def db(self):
        return self._data['dB']
    def fields(self):
        return ('db',)

################################################################
# named array
################################################################

_general_array_template = \
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

class {typename}(Drawer):
    '''
    1-D arrays of {fields}.
    '''
    def __init__(self, arr, info=None):
        '''
        :param arr: a 2-D array
        :param info: (optional) a dict for attaching information
        '''
        lens = list(map(len, arr))
        if not lens[1:] == lens[:-1]:
            raise RuntimeError('given arrays are not of equal length')
        self.__data = np.asarray(arr)
        self.info   = info
    def __len__(self):
        return len(self.__data[0])
    def __getitem__(self, key):
        c = copy.copy(self)
        c.__data = np.asarray(self.__data[:, key])
        return c
    def down_sample(self, nsample, *args, **kws):
        c = copy.copy(self)
        D = np.array([down_sample(d, nsample, *args, **kws) for d in self.__data])
        c.__data = np.asarray(D)
        return c
    def fields(self):
        return ({field_strs})
    def unpack(self):
        return [getattr(self, k) for k in self.fields()]
{field_defs}
"""
_field_template = """\
    @property
    def {name}(self):
        "{doc}"
        return self.__data[{index}]"""
def named_array(typename, field_names, docs=None, verbose=False):
    """
    define class for general 2-D array.

    :param typename: name for generated class
    :param field_name: a list of string. name for fields
    :param docs: a list of string. brief descriptions for fields.
    if not given, field_name is used.
    :param verbose: print class definition string just before it is actually defined.
    """
    if docs is None:
        field_defs = [_field_template.format(index=i,name=n, doc=n)
                  for i, n in enumerate(field_names)]
    else:
        field_defs = [_field_template.format(index=i,name=n, doc=d)
                  for i, (n, d) in enumerate(zip(field_names, docs))]
    s = _general_array_template.format(typename=typename,
                                       fields=', '.join(field_names),
                                       field_defs='\n'.join(field_defs),
                                       field_strs=('("'
                                                   + '", "'.join(field_names)
                                                   + '",)'))
    namespace = dict(__name__='namedarray_%s' % typename)
    namespace['down_sample'] = down_sample
    namespace['Drawer'] = Drawer
    exec(s, namespace)
    result = namespace[typename]
    result._source = s
    if verbose:
        print(result._source)

    try: # for pickling to work: copied from namedtuple definition
        import sys as _sys
        result.__module__ = _sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return result

PowerArray = named_array("PowerArray", ['power'], ['Power [W]'])
TemperatureArray = named_array("PowerArray", ['temperature'], ['Temperature [K]'])
KidFitResponse = named_array("KidFitResponse", ['amplitude', 'phase'])

################################################################
# data for indexing
################################################################

# should have x (as property), __len__
    
class TimeArray(object):
    """
    A class for representing array of time, with down sampling functionality.
    time stamp is either:

    - 2 arguments:

    :param index:
    :param samplerate: a 1-D array of integer samplerate: a float value of sampling rate in Hz.

    In this case, time is given by index * samplerate.

    - 1 argument:
    :param timestamp: a 1-D array of time in second.

    In this case, time is given by timestamp itself.
    """
    def __init__(self, *args):
        """
        Initialize either with:
        - 1 arg: timestamp
        - 2 args: index and framerate
        """
        if len(args) == 1:
            self.__data  = _TimeArray_Timestamp(*args)
        elif len(args) == 2:
            self.__data = _TimeArray_Index(*args)
        else:
            raise MKIDDataException('invalid initialization of TimeArray')
    def down_sample(self, nsample):
        return self.__data.down_sample(nsample)
    def __getitem__(self, key):
        return self.__data[key]
    @property
    def x(self):
        "Time [s]"
        return self.__data.value()
    @property
    def t(self):
        "Time [s]"
        return self.__data.value()
    def __len__(self):
        return len(self.__data)
    def down_sample(self, nsample):
        return self.__data.down_sample(nsample)
    def fields(self):
        return ('t',)
    def unpack(self):
        return [getattr(self, k) for k in self.fields()]

class _TimeArray_Index(object):
    def __init__(self, index, rate):
        self.__index = index
        self.__rate = rate
    def down_sample(self, nsample):
        return TimeArray(self.__index[::nsample], self.__rate__)
    def value(self):
        return self.__index / self.__rate
    def __getitem__(self, key):
        return TimeArray(self.__index[key], self.__rate__)
    def __len__(self):
        return len(self.__index)
class _TimeArray_Timestamp(object):
    def __init__(self, timestamp):
        self.__timestamp = timestamp
    def down_sample(self, nsample):
        return TimeArray(self.__timestamp[::nsample])
    def value(self):
        return self.__timestamp
    def __getitem__(self, key):
        return TimeArray(self.__timestamp[key])
    def __len__(self):
        return len(self.__timestamp)

class SweepFreqArray(object):
    """
    A class for representing array of frequency

    :param freqs: a 1-D array of frequency [GHz]
    """
    def __init__(self, freqs):
        self.__data = freqs * 1e9
    def __len__(self):
        return len(self.__data)
    @property
    def x(self):
        "Frequency [GHz]"
        return self.__data / 1e9
    def fields(self):
        return ('x',)
    def __getitem__(self, key):
        c = copy.copy(self)
        c.__data = self.__data[key]
        return c
    def unpack(self):
        return [getattr(self, k) for k in self.fields()]

class FFTFreqArray(object):
    """
    A class for representing array of frequency

    :param freqs: a 1-D array of frequency [Hz]
    """
    def __init__(self, freqs):
        self.__data = freqs
    def __len__(self):
        return len(self.__data)
    @property
    def x(self):
        "Frequency [Hz]"
        return self.__data
    def fields(self):
        return ('x',)
    def __getitem__(self, key):
        c = copy.copy(self)
        c.__data = self.__data[key]
        return c
    def unpack(self):
        return [getattr(self, k) for k in self.fields()]

################################################################
################################################################

class BaseMultiData(Drawer):
    """
    base class for combined data array with the same length.

    datalist is a list of object with
    - __len__
    - __getitem__
    - __down_sample__ (optional)
    """
    def __init__(self, datalist, info=None, *args, **kws):
        import operator
        if not (reduce(operator.eq, map(len, datalist))):
            raise RuntimeError('given arrays are not of equal length')
        self._datalist = datalist
        if info is None:
            self.info = dict()
        else:
            self.info = info.copy()

    def __getitem__(self, key):
        c = copy.copy(self)
        c._datalist = [d[key] for d in self._datalist]
        return c

    def __getattr__(self, attr):
        if attr in ['_datalist']:
            raise AttributeError()
        for d in self._datalist:
            if hasattr(d, attr):
                return getattr(d, attr)
        else:
            raise AttributeError()

    def down_sample(self, nsample, *args, **kws):
        c = copy.copy(self)
        assert self._xdata is c._xdata
        c._data = [d.down_sample(nsample, *args, **kws) for d in self._datalist]
        return c

    def fields(self):
        "data names inside object"
        import operator
        return reduce(operator.add, [d.fields() for d in self._datalist])

    def unpack(self):
        return [getattr(self, k) for k in self.fields()]

################################################################
################################################################

class FixedData(BaseMultiData, TimeArray, KidGlobalResponse):
    """
    a class for TOD data before fitting: TimeArray + KidGlobalResponse.

    Initialize with one of the following form:

    - **FixedData(frequency, index, I, Q, fsample)**:

    :param freuquency: a float of frequency in Hz
    :param index: a 1-D array of integer for timestamp
    :param I: a 1-D array of measured I
    :param Q: a 1-D array of measured Q
    :param fsample: a float of sampling frequency in Hz

    - **FixedData(frequency, timestamp, I, Q)**:

    :param freuquency: a float of frequency in Hz
    :param timestamp: a 1-D array of timestamps in second
    :param I: a 1-D array of measured I
    :param Q: a 1-D array of measured Q
    """
    def __init__(self, *args, **kws):
        if args:
            if 'info' in kws:
                info = kws['info']
            else:
                info = dict()
            if isinstance(args[0], str):
                xdata = TimeArray(*args[1])
                ytype = args[0]
                ydata = KidGlobalResponse(ytype, args[2])
                super(FixedData, self).__init__((xdata, ydata), info)
                frequency = args[3]
                self._frequency = frequency
            elif len(args) == 5:
                frequency, index, I, Q, fsample = args
                ydata = KidGlobalResponse('I-Q', (I, Q))
                xdata = TimeArray(index, fsample)
                super(FixedData, self).__init__((xdata, ydata), info)
                self._frequency = frequency
            elif len(args) == 4:
                frequency, timestamp, I, Q = args
                ydata = KidGlobalResponse('I-Q', (I, Q))
                if type(timestamp) == TimeArray:
                    xdata = timestamp
                else:
                    xdata = TimeArray(timestamp)
                super(FixedData, self).__init__((xdata, ydata), info)
                self._frequency = frequency
    def calibrate_with_blind_tones(self, blind_left, blind_right):
        """
        do blind tone calibration.

        :param left: left blind tone TOD.
        :param right: right blind tone TOD.

        each blind tone is either a FixedData or a tuple (freq, IQ)
          where freq is a frequency [Hz],
                IQ is a 1-D array of IQ data as complex value.

        :return: a new FixedData object.
        """
        calibrated = calibrate_with_blind_tones(self, blind_left, blind_right)
        d = FixedData(self.frequency*1e+9, self.t,
                      np.real(calibrated), np.imag(calibrated),
                      info=self.info)
        return d


    @property
    def frequency(self):
        return self._frequency / 1e9

class FixedFitData(BaseMultiData, TimeArray, KidFitResponse):
    """
    a class for TOD data after fitting: TimeArray + KidFitResponse.

    Initialize with one of the following form:

    - **FixedFitData(frequency, index, I, Q, fsample)**:

    :param freuquency: a float of frequency in Hz
    :param index: a 1-D array of integer for timestamp
    :param amplitude: a 1-D array of amplitude
    :param phase: a 1-D array of phase
    :param fsample: a float of sampling frequency in Hz

    - **FixedFitData(frequency, timestamp, I, Q)**:

    :param freuquency: a float of frequency in Hz
    :param timestamp: a 1-D array of timestamps in second
    :param amplitude: a 1-D array of amplitude
    :param phase: a 1-D array of phase
    """
    def __init__(self, *args, **kws):
        if args:
            if 'info' in kws:
                info = kws['info']
            else:
                info = dict()
            if len(args) == 5:
                frequency, index, amplitude, phase, fsample = args
                ydata = KidFitResponse((amplitude, phase))
                xdata = TimeArray(index, fsample)
                super(FixedFitData, self).__init__((xdata, ydata), info)
                self._frequency = frequency
            elif len(args) == 4:
                frequency, timestamp, amplitude, phase = args
                ydata = KidFitResponse((amplitude, phase))
                if type(timestamp) == TimeArray:
                    xdata = timestamp
                else:
                    xdata = TimeArray(timestamp)
                super(FixedFitData, self).__init__((xdata, ydata), info)
                self._frequency = frequency

    @property
    def frequency(self):
        return self._frequency / 1e9

    #def power_spectrum_density(self, ndivide=1, doplot=False, window=scipy.signal.hanning, overwrap_half=False, dt=None):
    def power_spectrum_density(self, ndivide=1, doplot=False, window=np.hanning, overwrap_half=False, dt=None):
        if dt == None:
            dt = self.t[1] - self.t[0]
        f_, ampl_ = power_spectrum_density(self.amplitude, dt, ndivide, doplot, window, overwrap_half)
        f_, phase_ = power_spectrum_density(self.phase, dt, ndivide, doplot, window, overwrap_half)
        return FixedFitPSDData(f_, ampl_, phase_)

class SweepFitData(BaseMultiData, SweepFreqArray, KidFitResponse):
    """
    a class for sweep data after fitting: SweepFreqArray + KidFitResponse.

    - **SweepFitData(freq, I, Q)**:

    :param freq: a 1-D array of frequency [GHz]
    :param amplitude: a 1-D array of amplitude
    :param phase: a 1-D array of phase
    """
    def __init__(self, freq, amplitude, phase, info=None):
        if info is None:
            info = dict()
        else:
            info = info.copy()

        ydata = KidFitResponse((amplitude, phase))
        xdata = SweepFreqArray(freq)
        super(SweepFitData, self).__init__((xdata, ydata), info)


################################################################
# PSD data
################################################################
class FixedFitPSDData(BaseMultiData, FFTFreqArray, KidFitResponse):
    """
    a class for PSD data after fitting: FFTFreqArray + KidFitResponse.

    - **FixedFitPSDData(frequency, index, I, Q, fsample)**:

    :param freuquency: a 1-D array of float of frequency in Hz
    :param amplitude: a 1-D array of amplitude
    :param phase: a 1-D array of phase
    """

    def __init__(self, *args, **kws):
        if args:
            frequency, amplitude, phase = args
            if 'info' in kws:
                info = kws['info']
            else:
                info = dict()
            ydata = KidFitResponse((amplitude, phase))
            xdata = FFTFreqArray(frequency)
            super(FixedFitPSDData, self).__init__((xdata, ydata), info)

    @property
    def f(self):
        return self.x

################################################################
# others
################################################################
class rikenData(KidGlobalResponse):
    """
    data vs freqs (as x())
    """
    @property
    def x(self):
        """
        Frequency [GHz]
        """
        return self._Hz/1e9
    def __getitem__(self, key):
        data = rikenData()
        super(rikenData, data).__init__(self._kind, [d[key] for d in self._rawdata])
        newhz = self._Hz[key]
        data._Hz = newhz
        return data
    def __init__(self, filename = None, origin = None):
        if not filename:
            pass
        elif fnmatch.fnmatch(filename, '*.csv'):
            self.info, data = _riken_vna_csv(filename)
            super(rikenData, self).__init__('dB-DEG', data[1:])
            self._Hz = data[0]

        else:
            raise MKIDDataException('not known file type: %s' % filename)
    def down_sample(self, nsample, method='IQ'):
        if method == 'IQ':
            I = []
            Q = []
            for i in range(int(np.ceil(len(self)/float(nsample)))):
                beg = i*nsample
                end = min(len(self)-1, (i+1)*nsample)
                I.append(np.average(self.i[beg:end]))
                Q.append(np.average(self.q[beg:end]))
            data = rikenData()
            super(rikenData, data).__init__('I-Q', (I, Q))
            newhz = self._Hz[::nsample]
            data._Hz = newhz
            return data

        else:
            raise MKIDDataException('method not implemented: %s' % method)

class SweepData(KidGlobalResponse, Drawer):
    """
    data vs freqs (as x())
    """
    @property
    def x(self):
        """
        Frequency [GHz]
        """
        return self._Hz/1e9
    def __getitem__(self, key):
        data = SweepData()
        super(SweepData, data).__init__(self._kind, [d[key] for d in self._rawdata])
        newhz = self._Hz[key]
        data._Hz = newhz
        return data
    def __init__(self, filename = None, origin = None, filetype=None, nch=32):
        if not filename:
            pass
        elif filetype == 'rohde':
            self.info, data = _parse_rohde_csv(filename)
            super(SweepData, self).__init__('dB-DEG', data[1:])
            self._Hz = data[0]
        elif filetype == 'riken':
            self.info, data = _parse_riken_csv(filename)
            super(SweepData, self).__init__('dB-DEG', data[1:])
            self._Hz = data[0]
        elif fnmatch.fnmatch(filename, '*.csv'):
            self.info, data = _parse_vna_csv(filename)
            super(SweepData, self).__init__('dB-DEG', data[1:])
            self._Hz = data[0]
        elif fnmatch.fnmatch(filename, '*.cti'):
            self.info, data = _parse_vna_citi(filename)
            super(SweepData, self).__init__('dB-DEG', data[1:])
            self._Hz = data[0]
        elif fnmatch.fnmatch(filename, '*Freq_IQ*.dat*'):
            if not origin and origin != 0:
                raise MKIDDataException('specify origin([Hz])!')
            self.info, (MHz, I, Q) = _parse_kibe_sweep_data(filename, origin)
            if nch == 32:
                super(SweepData, self).__init__('I-Q', (I, Q))
            else:
                super(SweepData, self).__init__('I-Q', (I*32.0/float(nch), Q*32.0/float(nch)))
            self._Hz = MHz * 1e6 + origin*1e9
        else:
            raise MKIDDataException('not known file type: %s' % filename)
    def down_sample(self, nsample, method='IQ'):
        if method == 'IQ':
            I = []
            Q = []
            for i in range(int(np.ceil(len(self)/float(nsample)))):
                beg = i*nsample
                end = min(len(self)-1, (i+1)*nsample)
                I.append(np.average(self.i[beg:end]))
                Q.append(np.average(self.q[beg:end]))
            data = SweepData()
            super(SweepData, data).__init__('I-Q', (I, Q))
            newhz = self._Hz[::nsample]
            data._Hz = newhz
            return data
            
        else:
            raise MKIDDataException('method not implemented: %s' % method)
    @property
    def f(self):
        """
        Frequency [GHz]
        """
        return self._Hz/1e9
    def fields(self):
        return ('x',) + super(SweepData, self).fields()

    def unpack(self):
        return [getattr(self, k) for k in self.fields()]

###
def read_sweep(filename = None, origin = None, **kwargs):
    return SweepData(filename, origin, **kwargs)

def read_riken(filename = None, origin = None):
    return rikenData(filename, origin)

def read_fixed(filename, origin, force_reload=False, count=None):
    max_n_ch = 32               # determined by fpga firmware
    if force_reload:
        warnings.warn('force_reload option is deprecated', DeprecationWarning)

    if fnmatch.fnmatch(filename, '*.gz'):
        open_ = gzip.open
    else:
        open_ = open
    ## read header
    freqs = collections.OrderedDict()
    nchs  = dict()
    info  = dict()
    with open_(filename) as f:
        for i, l in enumerate(f):
            if (not l) or (l[0] != '#'): break
            # if i <= 2: continue
            if l[2] == '#': continue
            if ':' in l:
                pos = l.index(':')
                key, value = l[2:pos], l[pos+1:]
                info[key.strip()] = value.strip()
            else:       # frequency comb
                try:
                    ch, freq = l[2:].split()
                    f = float(freq)
                    if f not in freqs.values():
                        freqs[int(ch)] = f
                        nchs[f] = 1
                    else:
                        nchs[f] += 1
                except ValueError:  # didn't use all 32ch
                    pass

    inds = [0]

    ## parse info
    if 'used channel' in info:
        used_n_ch = int(info['used channel'].split('=')[1].strip())
    else:
        used_n_ch = max_n_ch
    if 'down sampling count' in info:
        count_ = int(info['down sampling count'].split('=')[1].strip())
        if count and count_ != count:
            raise MKIDDataException("count given as argument (%d) and written in file (%d) don't match" % (count, count_))
        count = count_
    else:
        if not count:
            print >> sys.stderr, "warning: count not given nor written in file. using 20000 as default"
            count = 20000   # default
    channels, freqvals = zip(*[(c, f) for (c, f) in freqs.items() if c <= used_n_ch])
    for c in channels:
        inds.append(2*c-1)
        inds.append(2*c)
    ## read body
    rawdata = my_loadtxt(filename, usecols = inds, unpack=True)
    ##  => [index, I_f1, Q_f1, I_f2, Q_f2, ..., I_fn, Q_fn] , n is number of frequencies used
    data = []
    for i, f in enumerate(freqvals):
        I = rawdata[2*i+1]/float(count)/2**24/(nchs[f]/float(max_n_ch))
        Q = rawdata[2*i+2]/float(count)/2**24/(nchs[f]/float(max_n_ch))
        # I = rawdata[2*i+1]/float(count)/2**24/float(nchs[f])
        # Q = rawdata[2*i+2]/float(count)/2**24/float(nchs[f])
        fsample = 245.76e6 / count # Hz
        d = FixedData(f*1e6 + origin*1e9, rawdata[0], I, Q, fsample)
        d.info = info
        data.append(d)
    return data

def _riken_vna_csv(filename):
    state = 'HEADER'
    data = []
    info = dict()
    timefmt = '!Date: %A, %B %d, %Y %H:%M:%S'
    for i, line in enumerate(open(filename)):
        line = line.rstrip()
        if line == '':
            continue
        elif state == 'HEADER':
            info['header'] = ['Freq(Hz)', 'S21(DB)', 'S21(DEG)']
            state = 'BODY'
        elif state == 'BODY':
            if line[0:3] == 'END':
                state = 'END'
                break
            data.append([float(val) for val in line.split(',')])
        else:
            raise exceptions.RuntimeError('parse error, state == %s' % state)
    if tuple(info['header']) != ('Freq(Hz)', 'S21(DB)', 'S21(DEG)'):
        print >> sys.stderr, 'seems not %s data' % ('Freq(Hz)', 'S21(DB)', 'S21(DEG)')
    return info, np.array(data).T

def _parse_vna_csv(filename):
    state = 'START'
    data = []
    info = dict()
    timefmt = '!Date: %A, %B %d, %Y %H:%M:%S'
    for i, line in enumerate(open(filename)):
        line = line.rstrip()
        if line == '':
            continue
        if line[0] == '!':
            if re.match('!Date:', line):
                info['date'] = datetime.datetime.strptime(line, timefmt)
                continue
        if state == 'START':
            if line[0:5] == 'BEGIN':
                state = 'HEADER'
        elif state == 'HEADER':
            info['header'] = line.split(',')
            state = 'BODY'
        elif state == 'BODY':
            if line[0:3] == 'END':
                state = 'END'
                break
            data.append([float(val) for val in line.split(',')])
        else:
            raise exceptions.RuntimeError('parse error, state == %s' % state)
    if tuple(info['header']) != ('Freq(Hz)', 'S21(DB)', 'S21(DEG)'):
        print >> sys.stderr, 'seems not %s data' % ('Freq(Hz)', 'S21(DB)', 'S21(DEG)')
    return info, np.array(data).T


### file parsers

def _parse_rohde_csv(filename):
    f, db, ang = np.loadtxt(filename, skiprows=3, unpack=True)
    return None, (f, db, ang)

def _parse_riken_csv(filename):
    f, db, ang = np.loadtxt(filename, unpack=True, delimiter=',')
    return None, (f, db, ang)

def _parse_vna_citi(filename):
    state = 'HEADER'
    var = []
    data = []
    info = dict()
    timefmt = '!Date: %A, %B %d, %Y %H:%M:%S'
    for i, line in enumerate(open(filename)):
        line = line.rstrip()
        if line == '':
            continue
        if line[0] == '!':
            if re.match('!Date:', line):
                info['date'] = datetime.datetime.strptime(line, timefmt)
                continue

        elif state == 'HEADER':
            if re.match('CITIFILE\s', line):
                # print 'citi file version', line.split(' ')[1]
                pass
            elif re.match('NAME\s', line):
                info['NAME'] = line.split(' ')[1]
            elif re.match('VAR\s', line):
                info['VAR'] = line.split(' ')[1]
            elif re.match('DATA\s', line):
                info['DATA'] = line.split(' ')[1:]
            elif re.match('VAR_LIST_BEGIN', line):
                state = 'VARLIST'
            # info['header'] = line.split(',')
            elif re.match('BEGIN', line):
                state = 'BODY'
        elif state == 'VARLIST':
            if line == 'VAR_LIST_END':
                state = 'HEADER'
            else:
                var.append(float(line))
        elif state == 'BODY':
            if line[0:3] == 'END':
                state = 'END'
                break
            data.append([float(val) for val in line.split(',')])
        else:
            raise exceptions.RuntimeError('parse error, state == %s' % state)
    return info, np.concatenate([np.array([var]), np.array(data).T])

def _parse_kibe_sweep_data(filename, origin):
    """
    Read kibe-readout file
    """
    state = 'START'
    data = []
    info = dict()
    info['origin'] = origin
    data = np.loadtxt(filename, unpack=True)
    return info, data
def clear_fixed_cache():
    warnings.warn('Call to deprecated function %s.' % func.__name__, DeprecationWarning)

from contextlib import contextmanager
@contextmanager
def empty_fixed_cache():
    warnings.warn('Call to deprecated function %s.' % func.__name__, DeprecationWarning)
    yield

def my_loadtxt(filename, usecols=None, unpack=None, skiprows=0, delimiter=' '):
    comment_char = '#'
    if filename[-3:] == '.gz':
        compression='gzip'
    elif filename[-4:] == '.bz2':
        compression='bz2'
    else:
        compression=None
    ar = np.array(pd.read_csv(filename, compression=compression, delimiter=delimiter,
                              comment='#', header=None, usecols=usecols,
                              skiprows=skiprows))
    if unpack:
        return ar.T
    else:
        return ar

def amplitude_to_dB(x):
    return 20.0*np.log(x)/np.log(10.0)
def dB_to_amplitude(x):
    return 10.0**(x/20.0)    


class FitTOD(object):
    """
    Class for amplitude/phase TOD.
    """
    def __init__(self, time, amplitude, phase):
        self._t       = time
        self._kind    = kind
        self._rawdata = data
        if kind == 'dB-DEG':
            self._data = self.dBDEGData(data)
        elif kind == 'I-Q':
            self._data = self.IQData(data)
        elif kind == 'dB':
            self._data = self.dBData(data)
        else:
            raise MKIDDataException('data format not implemented')
    @property
    def amplitude(self):
        "S21, amplitude ratio"
        return self._data.amplitude()
    @property
    def phase(self):
        "S21, power in dB"
        return self._data.db()
    def __len__(self):
        return len(self._rawdata[0])
