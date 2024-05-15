# -*- coding: utf-8 -*-
#### temperature
import sys
import re
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from .misc import MKIDDataException
from .fitting import *

def read_templog(filename, timedelta=0, doplot=True):
    """read temperature log file.
    parameters:
        filename:  temperature log filename.
                   if filename includes 'he10', assumed to be taken in gb.
                   if filename includes 'he10', assumed to be taken in clean room refrigerator.
        timedelta: time offset in seconds
                   this is added to the time written in the log file.
        doplot:    plot read timperature log.
    return:
        (times, temperatures)
    """
    dt = datetime.timedelta(seconds=timedelta)
    if re.match('.*csv$', filename): # assuming clean room
        print 'assuming clean room...'
        def _safe_float(s):     # sometimes there's garvage
            try:
                return float(s)
            except:
                return float('nan')

        templog = np.loadtxt(filename, # ex. '20140822/temp_change_-55dBm_with_eart_retry/20140822.csv'
                             usecols = (0, 8),
                             # usecols = (0, 8),
                             delimiter = ',',
                             converters =
                             {0: lambda s:datetime.datetime.strptime(s, '%m/%d/%Y %H:%M:%S') + dt,
                              8: _safe_float},
                             dtype=np.dtype('O4, f8'),
                             unpack=True)
    elif re.match('.*he10', filename): # assuming he10 temperature log
        print 'assuming he10 log...'
        templog = np.loadtxt(filename, # ex. 'vnadata/20140716/he10_2014_0716_020950.dat'
                             usecols = (0, 8),
                             converters =
                             {0: lambda s:datetime.datetime.strptime(s, '%Y-%m%d-%H%M%S') + dt,},
                             dtype=np.dtype('O4, f8'),
                             unpack=True)
    else:
        raise MKIDDataException("not known temperature log file type")

    if doplot:
        import matplotlib.dates as mdates
        fig, ax = plt.subplots()
        ax.plot(templog[0], templog[1], '-+', label='log')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
    return templog


import time
@np.vectorize
def to_ts(t):
    return time.mktime(t.timetuple())
@np.vectorize
def from_ts(ts):
    return datetime.datetime.fromtimestamp(ts)

def interp_datetime(t, tp, fp):
    """
    same as numpy.interp, except that x-axis is datetime.datetime
    """
    t_ts = [to_ts(t_) for t_ in t]
    tp_ts = [to_ts(t_) for t_ in tp]
    return np.interp(t_ts, tp_ts, fp)

def add_temperature(sweeps, templog, vnatimedelta):
    """
    append temperature information (sweep.info['temperature']) to each data

    parameters:
        sweep:        list of sweepdata
        templog:      temperature log read with read_templog
        vnatimedelta: time delta of vna
    """
    dt = datetime.timedelta(seconds=vnatimedelta)
    dates = [s.info['date'] + dt for s in sweeps]
    temps = interp_datetime(dates, templog[0], templog[1])
    for t, s in zip(temps, sweeps):
        s.info['temperature'] = t
    return dates, temps

def fit_files_with_temps(vnafiles, templogfile, vnadt, templogdt, fitter = fitter_gaolinbg):
    """
    fit sweeps and calculate temperature for each sweep, with some message and temperature plot

    Keyword arguments:
    vnafiles -- list of sweep data filenames
    templogfile -- name of temperature log file
    vnadt -- VNA's clock time minus actual time [s]
    templogdt -- temperature logger's clock time minus actual time [s]
    fitter -- fitter for fitting each sweeps

    Returns: tuple (dates, ps, rs, temps)
    dates -- datetime objects corresponding to each sweep
    ps -- each sweep data object
    rs -- fit results
    temps -- interpolated temperatures
    """

    func, guess, names, others = fitter
    ps = []                     # read sweep data
    rs = []                     # fit results
    print 'reading temperature log...'
    templog = read_templog(templogfile, templogdt, True)
    print 'reading vna files...'
    for f in vnafiles:
        print os.path.basename(f),
        p = read_sweep(f)
        peaks = search_peaks(p)
        if len(peaks) == 1:
            try:
                errors = get_error_sweep_iq(p[-100:])
                params = dict_to_params(guess(p))
                s = adjust_fitrange(3, len(p), len(names), peaks[0])
                r = fit_from_params(p, s, errors, params, func, names)
                c = ErrorCalculator(params)
                ps.append(p)
                rs.append(r)
                print 'o fit success, fr =', r.fitparams['fr'].value
            except:
                print 'x fit failed'
        else:
            if peaks:
                print 'x peak ambiguous (%d peaks found)' % len(peaks)
            else:
                print 'x peak not found'
    print 'fitted %d / total %d' % (len(ps), len(vnafiles))
    print 'interpolating temperature log...'
    dates, temps = np.array(add_temperature(ps, templog, vnadt))
    datemin, datemax = min(dates), max(dates)
    datedelta = datemax - datemin
    s = np.where((to_ts(datemin - datedelta/10) <= to_ts(dates)) * (to_ts(dates) <= to_ts(datemax + datedelta/10)))
    plt.plot(dates[s], temps[s], 'r|', label='interpolation points')
    plt.legend()
    plt.show()
    return (dates, ps, rs, temps)
