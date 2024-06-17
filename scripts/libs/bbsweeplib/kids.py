import warnings
import os
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from astropy.io import fits as pyfits
import h5py
import matplotlib.pyplot as plt

import mkid_data as md

from .files import read_kidslist, read_localsweep, read_fits, read_fits_single
from .Prad import read_filterfile, dPrad
from .TODcalib import kids_with_both_blinds
from .arrays import AveragedPSDData

from .cache import Cache, do_cache

## todo: nep should be class?

class fits_tods_fits(Mapping):
    chunksize = int(1e6)
    kSampleRate = 2e9
    def __init__(self, infile):
        self.infile  = infile
        self.open()

    def open(self):
        self.hud = pyfits.open(self.infile)
        bintable = self.hud[1]
        self.fftgain  = bintable.header['fftgain']
        self.framert  = bintable.header['framert']
        self.nbins    = bintable.header['nbins']
        self.npoints  = bintable.header['npoints']
        self.lofreq   = bintable.header['lofreq']
        self.bins     = [_to_nbit_signed(bintable.header['BIN%d' % i], self.npoints)
                         for i in range(self.nbins)]
        self.if_freq      = self.kSampleRate * np.array(self.bins) / 2**self.npoints
        self.carrier_freq = self.if_freq + self.lofreq


        bindata = bintable.data
        timestamp_ = bindata.field('timestamp')
        framenr_ = bindata.field('framenr')
        self.timestamp = timestamp_
        self.framenr = framenr_
        self._read_bins = {}

        self.offset = -100 # remove last part of data (sometimes they behave bad)

    def close(self):
        self.hud.close()

    def __len__(self):
        return len(self.bins)
    def __contains__(self, key):
        return key in self.bins
    def __iter__(self):
        for key in sorted(self.bins):
            yield key
    def __getitem__(self, key):
        #print( 'get %s' % key )
        if key in self._read_bins:
            # return self.h5.root.
            return self._read_bins[key]
        else:
            ind = self.bins.index(key)
            chunksize=int(1e4)
            name = 'kid_%04d' % ind
            rawdata = self.hud[1].data['data']
            #read_I_ = rawdata[:, 2*ind    ]
            #read_Q_ = rawdata[:, 2*ind + 1]
            #offset = -100 # remove last part of data (sometimes they behave bad)
            read_I_ = rawdata[:self.offset, 2*ind    ]
            read_Q_ = rawdata[:self.offset, 2*ind + 1]

            # IQ = md.KidGlobalResponse('I-Q-Gain', (read_I_, read_Q_, self.fftgain))
            # time = md.TimeArray(self.timestamp)
            # d = md.BaseMultiData((IQ, time))
            info = dict()
            info['bins']   = self.bins
            info['freqs']  = self.carrier_freq[ind]
            info['header'] = self.hud[1].header
            #d = md.FixedData('I-Q-Gain', (self.timestamp,), (read_I_, read_Q_, self.fftgain),
            #                 self.carrier_freq[ind], info=info)
            d = md.FixedData('I-Q-Gain', (self.timestamp[:self.offset],), (read_I_, read_Q_, self.fftgain),
                             self.carrier_freq[ind], info=info)
            self._read_bins[key] = d
            return d
    def __getstate__(self):
        if hasattr(self, 'hud'):
            dic = self.__dict__.copy()
            dic['_read_bins'] = {}
            dic['hud'] = None
            return dic
        else:
            return self.__dict__

    def __setstate__(self, dic):
        self.__dict__ = dic.copy()
        if 'hud' in dic:
            self.open()


class hdf5_tods_fits(Mapping):
    chunksize = int(1e6)
    kSampleRate = 2e9
    def __init__(self, infile, Readout, Channel):
        self.infile  = infile
        self.R = Readout
        self.Ch = Channel
        self.header = dict()

        self.open()

    def read_header(self):
        #sysinfo = dict( self.hud.attrs )
        #chinfo = dict( self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch].attrs )
        datainfo = dict( self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['data'].attrs )

        self.header['lofreq']  = datainfo['lo_frequency']*1e+6 # MHz to Hz
        self.header['npoints'] = datainfo['frame_length']
        self.header['fftgain'] = datainfo['fft_gain']
        self.header['nbins']   = datainfo['num_bins']
        self.header['framert'] = datainfo['frame_rate']

    def open(self):
        self.hud = h5py.File(self.infile, 'r')
        self.read_header()

        self.fftgain  = self.header['fftgain']
        self.framert  = self.header['framert']
        self.nbins    = self.header['nbins']
        self.npoints  = self.header['npoints']
        self.lofreq   = self.header['lofreq']
        lbins = self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['bins']
        assert( len(lbins)==self.nbins )
        for i in range(self.nbins):
            self.header['BIN%d' %i] = lbins[i]
        self.bins     = [_to_nbit_signed(self.header['BIN%d' %i], self.npoints)
                         for i in range(self.nbins)]
        self.if_freq      = self.kSampleRate * np.array(self.bins) / 2**self.npoints
        self.carrier_freq = self.if_freq + self.lofreq

        bindata = self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['data']
        timestamp_ = bindata['timestamp'] * 1e-3 # msec to sec
        framenr_ = bindata['frame_number']
        self.timestamp = timestamp_
        self.framenr = framenr_
        self._read_bins = {}

        self.offset = -100 # remove last part of data (sometimes they behave bad)

    def close(self):
        self.hud.close()

    def __len__(self):
        return len(self.bins)
    def __contains__(self, key):
        return key in self.bins
    def __iter__(self):
        for key in sorted(self.bins):
            yield key
    def __getitem__(self, key):
        #print( 'get %s' % key )
        if key in self._read_bins:
            # return self.h5.root.
            return self._read_bins[key]
        else:
            ind = self.bins.index(key)
            chunksize=int(1e4)
            name = 'kid_%04d' % ind
            rawdata = self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['data']['frame_data']
            #read_I_ = rawdata[:, 2*ind    ]
            #read_Q_ = rawdata[:, 2*ind + 1]
            #offset = -100 # remove last part of data (sometimes they behave bad)
            read_I_ = rawdata[:self.offset, 2*ind    ]
            read_Q_ = rawdata[:self.offset, 2*ind + 1]

            info = dict()
            info['bins']   = self.bins
            info['freqs']  = self.carrier_freq[ind]
            info['header'] = self.header
            #d = md.FixedData('I-Q-Gain', (self.timestamp,), (read_I_, read_Q_, self.fftgain),
            #                 self.carrier_freq[ind], info=info)
            d = md.FixedData('I-Q-Gain', (self.timestamp[:self.offset],), (read_I_, read_Q_, self.fftgain),
                             self.carrier_freq[ind], info=info)
            self._read_bins[key] = d
            return d
    def __getstate__(self):
        if hasattr(self, 'hud'):
            dic = self.__dict__.copy()
            dic['_read_bins'] = {}
            dic['hud'] = None
            return dic
        else:
            return self.__dict__

    def __setstate__(self, dic):
        self.__dict__ = dic.copy()
        if 'hud' in dic:
            self.open()


class KIDs(Cache, Mapping):
    """
    hold all kids
    """
    def __init__(self, path, kidslist, sweeps_path, tods_path, **kws):
        kidslist = read_kidslist(kidslist)
        os.mkdir(path)
        self.path         = path
        self._kidslist    = kidslist
        self.framelen     = kidslist[0].get('framelen')
        if not self.framelen:
            if kws.get('framelen'):
                self.framelen = kws['framelen']
            else:
                warnings.warn('framelen info not in kidslist: please set KIDs.framelen manually')
        self._kids        = [KID(self, i, k, l, r, self.powers()[k], tods_path) for i, (k, l, r) in
                             enumerate(kids_with_both_blinds(kidslist, allow_without_blind=True))]
        self._sweeps_path = sweeps_path
        self._tods_path   = tods_path

        ext = os.path.splitext(tods_path)[-1]
        if ext=='.h5':
            self.TODHDF5 = True
        else:
            self.TODHDF5 = False
        if self.TODHDF5:
            #print( kws, type(kws) )
            #if kws.has_key('Readout'):
            if 'Readout' in kws:
                self.Readout = kws['Readout']
            else:
                warnings.warn('please set Readout number')
                raise RuntimeError('error')
            #if kws.has_key('Channel'):
            if 'Channel' in kws:
                self.Channel = kws['Channel']
            else:
                warnings.warn('please set Channel number')
                raise RuntimeError('error')

        self.hasRoomChopperLOsweep = False
        if kws.get('sweeps_path_roomchopper'):
            self._sweeps_roomchopper_path = kws['sweeps_path_roomchopper']
            self.hasRoomChopperLOsweep = True
        else:
            print('There is no local sweep with room chopper closed')
            self._sweeps_roomchopper_path = None

    def bins_kid(self):
        return self._kidslist[1]

    def bins_blind(self):
        return self._kidslist[2]

    def powers(self):
        return self._kidslist[3]

    @property
    def header(self):
        """
        header of bintable inside KID TOD fits file.
        """
        key = list(self.raw_tods().keys())[0]
        return self.raw_tods.cache[key].info['header']

    @do_cache
    def raw_sweeps(self):
        """read sweep file"""
        return read_localsweep(self._sweeps_path, framelen=self.framelen)

    @do_cache
    def raw_sweeps_roomchopper(self):
        """read sweep file"""
        if self.hasRoomChopperLOsweep:
            return read_localsweep(self._sweeps_roomchopper_path, framelen=self.framelen)

    @do_cache
    def raw_tods(self):
        # """read tod fits file"""
        # return read_fits(self._tods_path)
        if self.TODHDF5:
            return hdf5_tods_fits(self._tods_path, self.Readout, self.Channel)
        else:
            return fits_tods_fits(self._tods_path)

    @do_cache
    def find_glitch(self, baseline_thresh = 6.0, glitch_thresh = 5.0, clusterize_thresh = 2):
        bad_for_any = None
        for i, k in enumerate(self._kids):
            if not k.enabled: continue
            if k.has_cache('find_glitch'):
                bad = k.find_glitch.cache
            else:
                bad = k.find_glitch(baseline_thresh, glitch_thresh, clusterize_thresh)

            if bad_for_any is None:
                #bad_for_any = bad
                bad_for_any = bad.copy()
            else:
                bad_for_any |= bad

            #if i % 20 == 0:
            #    self.save(clear_memory=True)

        return bad_for_any

    def deglitch_all_enabled(self):
        bad = self.get_cache('find_glitch')
        for i, k in enumerate(self._kids):
            if not k.enabled: continue
            k.deglitch(bad)
            if i % 20 == 0:
                self.save(clear_memory=True)

    @do_cache
    def set_slice(self, slices):
        return slices

    @do_cache
    def set_slice_resp(self, slices):
        return slices

    @do_cache
    def calc_slice_powers(self):
        """calculate power for each slice, by taking
        center of power and its width from slice and power tod"""
        ts, prad = self.other_tods['power'].unpack()
        slices = self.get_cache('set_slice')
        prad_centers = []
        dprad_centers = []
        for j, s in enumerate(slices):
            maxprad = max(prad[s])
            minprad = min(prad[s])
            prad_centers.append((maxprad + minprad)/2)
            dprad_centers.append((maxprad - minprad)/2)
        return prad_centers, dprad_centers

    @do_cache
    def set_average_range(self, f_beg, f_end):
        return f_beg, f_end

    @do_cache
    def set_calcNEP(self, calcPrad, calctemp, NEP_tot, NEP_R, NEP_P, NEP_w):
        return calcPrad, calctemp, NEP_tot, NEP_R, NEP_P, NEP_w

    @property
    def other_tods(self):
        """
        a dict for other tod data
        """
        if not self.has_cache('other_tods'):
            self.set_cache('other_tods', dict())
        return self.get_cache('other_tods')

    ## methods to be like dict(), with inheritance of Mapping
    def __len__(self):
        return len(self._kidslist[1])
    def __getitem__(self, index):
        return self._kids[index]
    def __iter__(self):
        for i in range(len(self)):
            yield i
    def __contains__(self, item):
        return i in self._kids

def _to_nbit_signed(x, n):
    if x > 2**(n-1):
        return -((~x & (2**n-1))+1)
    else:
        return x

class KID(Cache):
    def __init__(self, parent, index, freqbin, leftbin, rightbin, readpower, fits_path):
        self._parent   = parent
        self._index    = index
        self.bin       = freqbin
        self.bin_l     = leftbin
        self.bin_r     = rightbin
        self.enabled   = leftbin and rightbin
        self.readpower = readpower
        self.fitresult = None
        self.others    = dict()
        self.path      = parent.path + '/%04d' % self._index
        self.fits_path = fits_path
        os.mkdir(self.path)

    ## functions to retrieve raw data
    @do_cache
    def raw_sweep(self):
        swps = self._parent.get_cache('raw_sweeps')
        if swps:
            return swps[self.bin]

    @do_cache
    def raw_sweep_roomchopper(self):
        swps = self._parent.get_cache('raw_sweeps_roomchopper')
        if swps:
            return swps[self.bin]

    @do_cache
    def find_glitch(self,
                    baseline_thresh = 6.0, glitch_thresh = 5.0, clusterize_thresh = 2,
                    interp_offset = 0):
        #ampl = self.get_cache('calibrate_with_blind_tone').amplitude
        #bad = md.find_glitch([ampl], baseline_thresh, glitch_thresh, clusterize_thresh)
        #calibrated = self.get_cache('calibrate_with_blind_tone')
        #(ts, ampl, phase) = calibrated.unpack()
        rewind = self.get_cache('rewind_tod')
        phase = rewind.phase
        bad = md.find_glitch([phase],
                             baseline_thresh, glitch_thresh, clusterize_thresh,
                             interp_offset)
        return bad

    @do_cache
    def blind_tone_left(self):
        # raw_tods = self._parent.get_cache('raw_tods')
        # if raw_tods:
        #     return raw_tods[self.bin]
        # return read_fits_single(self.fits_path, self.bin_l)
        return self._parent.raw_tods.cache[self.bin_l]

    @do_cache
    def blind_tone_right(self):
        # raw_tods = self._parent.get_cache('raw_tods')
        # if raw_tods:
        #     return raw_tods[self.bin]
        # return read_fits_single(self.fits_path, self.bin_r)
        return self._parent.raw_tods.cache[self.bin_r]

    @do_cache
    def raw_tod(self):
        # raw_tods = self._parent.get_cache('raw_tods')
        # if raw_tods:
        #     return raw_tods[self.bin]
        return self._parent.raw_tods.cache[self.bin]
      #;read_fits_single(self.fits_path, self.bin)

    ## fit raw sweep
    @do_cache
    def fit(self, nfwhm=5, fitter='gaolinbg', Q_search=1e+4):
        # err = md.get_error_sweep_iq(self.sweep()[:10])
        swp = self.get_cache('raw_sweep')
        #fc = self.get_fcarrier() # GHz
        fc = self.get_fcenter() # GHz
        err = None
        r = md.fit_onepeak(swp, fc, err, nfwhm, fitter=fitter, Q_search=Q_search)
        return r

    @do_cache
    def fit_roomchopper(self, nfwhm=5, fitter='gaolinbg', Q_search=1e+4):
        swp = self.get_cache('raw_sweep_roomchopper')
        #fc = self.get_fcarrier() # GHz
        fc = self.get_fcenter() # GHz
        err = None
        r = md.fit_onepeak(swp, fc, err, nfwhm, fitter=fitter, Q_search=Q_search)
        return r

    ## functions to calibrate raw tod
    @do_cache
    def calibrate_with_blind_tone(self):
        raw_tods = self._parent.get_cache('raw_tods')
        sig = raw_tods[self.bin]
        cal = sig.calibrate_with_blind_tones(self.blind_tone_left.cache,
                                             self.blind_tone_right.cache)
        return cal

    @do_cache
    def rewind_tod(self):
        r   = self.get_cache('fit')
        #tod = self.get_cache('raw_tod')
        tod = self.get_cache('calibrate_with_blind_tone')
        return r.rewind_data(tod)

    @do_cache
    def deglitch(self, bad):
        caltod = self.get_cache('rewind_tod')
        ampl = md.interpolate_bad(caltod.amplitude, bad)
        phase = md.interpolate_bad(caltod.phase, bad)
        d = md.FixedFitData(caltod.frequency*1e9, caltod.t, ampl, phase)
        return d

    ## functions to slice tod
    #@do_cache
    #def slice_tod(self):
    #    slices = self._parent.get_cache('set_slice')
    #    d = self.get_cache('deglitch')
    #    ts, ampl, phase = d.unpack()
    #    # linearize phase
    #    linphase = self.convert_to_fshift(phase)
    #
    #    d = md.FixedFitData(d.frequency*1e+9, ts, ampl, linphase)
    #    return [d[s] for s in slices]

    @do_cache
    def calc_slice_psds(self, ndivide=7, window=None, overwrap_half=True, dt=None):
        #ds = self.get_cache('slice_tod')
        ds = self.slice_tod(opt='phase')
        ds_ = []
        for j, d in enumerate(ds):
            d_ = d.power_spectrum_density(ndivide, window=window, overwrap_half=overwrap_half, dt=dt)
            ds_.append(d_)
        return ds_

    @do_cache
    def average_slice_tod(self):
        #ds = self.get_cache('slice_tod')
        ds = self.slice_tod(opt='fshift')
        ampls, phases, dampls, dphases = [], [], [], []
        for j, d in enumerate(ds):
            ts, ampl, phase = d.unpack() # phase = fshift
            ampls.append( np.average(ampl) )
            phases.append( np.average(phase) )
            #dampls.append( np.std(ampl)/np.sqrt( len(ampl)-1 ) )
            #dphases.append( np.std(phase)/np.sqrt( len(phase)-1 ) )
            dampls.append( np.std(ampl) )
            dphases.append( np.std(phase) )

        return np.array([ampls, dampls, phases, dphases])

    @do_cache
    def average_slice_psds(self):
        ds = self.get_cache('calc_slice_psds')
        f_beg, f_end = self._parent.get_cache('set_average_range')

        ampls, phases, dampls, dphases, fs = [], [], [], [], []

        for d in ds:
            avgrange = np.logical_and(f_beg <= d.f, d.f < f_end)
            ampls.append(np.average(d.amplitude[avgrange]))
            phases.append(np.average(d.phase[avgrange])) # phase = fshift
            #ampls.append( np.amin(d.amplitude[avgrange]) )
            #phases.append( np.amin(d.phase[avgrange]) ) # phase = fshift
            dampls.append(np.std(d.amplitude[avgrange])/np.sqrt(len(d.amplitude[avgrange])-1))
            dphases.append(np.std(d.phase[avgrange])/np.sqrt(len(d.amplitude[avgrange])-1))
            fs.append(d.f)

        return AveragedPSDData([ampls, phases, dampls, dphases])

    @do_cache
    def set_filter_params(self, masterid, kind, F0, dF0, Q, dQ):
        return masterid, kind, F0, dF0, Q, dQ

    @do_cache
    def set_dPdT(self, pval, perr):
        return pval, perr

    @do_cache
    def set_Rfitresults(self, pval_ampl, perr_ampl, pval_phase, perr_phase):
        return pval_ampl, perr_ampl, pval_phase, perr_phase

    @do_cache
    def set_responsivities(self, power, ampl, phase, df, dfcalc,
                           dpower, dampl, dphase, ddf, ddfcalc):
        return power, ampl, phase, df, dfcalc, dpower, dampl, dphase, ddf, ddfcalc

    @do_cache
    def set_position(self, x, y):
        return x, y

    def calc_responsivity(self, power): ##### this should be removed
        """
        calculate responsivity by interpolating responsivity data.

        interpolation is done assuming responsivity is proportional to sqrt(power).
        """
        power_, ampl, phase, dpower, dampl,dphase = self.get_cache('set_responsivities')
        interp_ampl2 = np.interp(power, power_, np.array(ampl)**2)
        interp_phase2 = np.interp(power, power_, np.array(phase)**2)
        return np.sqrt(interp_ampl2), np.sqrt(interp_phase2)

    @do_cache
    def set_calcNEP(self, calcPrad, calctemp, NEP_tot, NEP_R, NEP_P, NEP_w):
        return calcPrad, calctemp, NEP_tot, NEP_R, NEP_P, NEP_w

    @do_cache
    def set_NEPs(self, power, ampl, phase, dpower, dampl, dphase):
        return np.array([power, ampl, phase, dpower, dampl,dphase])

    @do_cache
    def set_NEPdark(self, freq, phase):
        #return freq, ampl, phase
        return freq, phase

    @do_cache
    def set_etaopt(self, etaopt, detaopt):
        return etaopt, detaopt

    @do_cache
    def set_geometries(self, Allength, Alwidth, Althickness):
        return Allength, Alwidth, Althickness # um

    @property
    def other_tods(self):
        """
        a dict for other tod data
        """
        if not self.has_cache('other_tods'):
            self.set_cache('other_tods', dict())
        return self.get_cache('other_tods')

    def get_fcarrier(self):
        fc = self._parent.raw_tods.cache[self.bin].frequency # GHz
        return fc

    def get_fcenter(self):
        swp = self.get_cache('raw_sweep')
        fc = (swp.x[0] + swp.x[-1])/2. # GHz
        return fc

    def convert_to_fshift(self, phase, opt='phase'):
        ##### responsivity from df/fr
        r   = self.get_cache('fit')
        swp = self.get_cache('raw_sweep')
        fr = r.params['fr'].value # GHz
        fc = self.get_fcarrier() # GHz
        rw_f = r.rewind(swp.x, r.fitted(swp.x))
        phase_f = -np.angle(-rw_f)
        ## spline interpolation
        import scipy.interpolate
        tck = scipy.interpolate.splrep(phase_f, swp.x, s=0)
        f = scipy.interpolate.splev(phase, tck, der=0)

        if opt=='phase':
            return phase
        elif opt=='fshift':
            #return (f-fc)/fr
            return (f-fr)/fr
        elif opt=='linphase':
            Qr = r.params['Qr'].value
            #return 4*Qr * (f-fc)/fr
            return 4*Qr * (f-fr)/fr
        else:
            print( '>>> KID::convert_to_fshift: Not supported option!!' )
            print( '>>> return phase' )
            return phase

    def slice_tod(self, opt='phase'):
        slices = self._parent.get_cache('set_slice')
        d = self.get_cache('deglitch')
        ts, ampl, phase = d.unpack()

        ## linearize phase
        fshift = self.convert_to_fshift(phase, opt)

        d = md.FixedFitData(d.frequency*1e+9, ts, ampl, fshift)
        return [d[s] for s in slices]

    ## draw
    def draw(self, kind='freq', ax=None):
        """
        kind: one of 'freq', 'IQ'
        """
        if ax is None:
            ax = plt.gca()
        swp = self.get_cache('raw_sweep')
        tod = self.get_cache('raw_tod')

        if kind == 'freq':
            ax.plot(swp.x*1e3, swp.db)
            if tod:
                ax.plot(tod.frequency*1e3 + np.zeros_like(tod.db), tod.db, ',')
            ax.set_xlabel('Frequency [MHz]')
            ax.set_ylabel('S21 [dB]')
            ax.grid(True)

        elif kind == 'IQ':
            ax.set_aspect('equal')
            ax.plot(swp.i, swp.q)
            if tod:
                ax.plot(tod.i, tod.q, ',')
            ax.set_xlabel('I')
            ax.set_ylabel('Q')
            ax.grid(True)

        else:
            raise RuntimeError('unknown kind to plot for KID: %s' % kind)

        return ax

    def responsivity(self):
        pass
