#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Make TOD data of temperature


input files
 - (kidslist)
 - (temperaturefile)
 - out/CalibTOD_TOD_%04d.png

output files:
 - out/TemperatureTOD.npy
verbose output files:
 - out/TemperatureTOD_%04d.png
"""
import os
import sys
import shutil
import gc
import copy

import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits

def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=myHelpFormatter, description=__doc__)
    parser.add_argument('--skiprows', type=int, default=1,
                        help='number of rows to skip frequency datafile header')
    parser.add_argument('--verbose', type=int, default=default_verbose_level,
                        help='verbose level. 0: no plot, 1: plot')
    parser.add_argument('--refvalue', type=float, default=3.0,
                        help='Value that related to KIDs to be written to reference.dat. The larger this value, the fewer KIDs are written to reference.dat.')
    parser.add_argument('--mode', choices=['both', 'plot', 'calc'], default='calc',
                        help="""select do calulation only,
                        do plotting data (calculated before), or both""")
    parser.add_argument('--tcut', type=float, default=0.5,
                        help="""tcut for fft calculation (sec),
                        need to be shorter than integration time in the measurement""")
    parser.add_argument('--force', action='store_true') # for calc & plot
    parser.add_argument('--test', action='store_true') # for plot
    parser.add_argument('--ncpu', type=int, default=1,
                        help="")

    args = parser.parse_args(argv)
    if args.mode == 'both':
        do_calc = True
        do_plot = True
    elif args.mode == 'calc':
        do_calc = True
        do_plot = False
    elif args.mode == 'plot':
        do_calc = False
        do_plot = True

    THzsweepfreqfile  = config.get('datafiles', 'THzsweep_freq')
    #THzsweepfitsfile  = config.get('datafiles', 'THzsweep_fits')
    # assume time duration at one frequency is more than tcut (sec)
#    tcut = 0.5 # sec
#    tcut = 4 # sec
    tcut = args.tcut
    print( 'tcut: %f sec' %tcut )
    
    REFVALUE = args.refvalue
    
    if do_calc:
        Calc(kids, THzsweepfreqfile, tcut, args.skiprows, args.force, REFVALUE)

    if do_plot:
        NCPU = args.ncpu
        Plot(kids, THzsweepfreqfile, tcut, args.skiprows, args.force, args.test, NCPU)

def calcfft(signal, dt):
    nfftpoint = 1024
    size = len(signal)

    x = scipy.fftpack.fftfreq(nfftpoint, dt)
    x_ = x[np.where(x>=0)]
    #y = 2.0/size * np.abs( scipy.fftpack.fft(signal,n=nfftpoint) )
    y = 2.0/size * np.abs( scipy.fftpack.fft(signal-np.average(signal),n=nfftpoint) )
    y_ = y[:len(x_)]

    #return x_, y_
    return x_, 2.*y_


def Calc(kids, THzsweepfile, tcut, skiprows, force, REFVALUE):
    #tempts, temp, cnt = my_loadtxt(bathsweepfile, unpack=True, skiprows=skiprows, delimiter=' ')
    readtxt = my_loadtxt(THzsweepfile, unpack=True, skiprows=skiprows, delimiter=' ')
    tempts = readtxt[0]
    tempfreq   = readtxt[2]
    photocurrent = readtxt[3]**2
    if len(readtxt)>4:
        cnt = readtxt[4]

    ###### read reduced fits file
    ifname = os.path.join(outdir, 'reduced_' + os.path.basename(kids._tods_path))
    hdus = fits.open(ifname)

    ##### make slice
    idx = 0
    for i, kid in kids.items():
        if not kid.enabled: continue
        idx = i; break
    #ts, ampl, phase = kids[idx].get_cache('deglitch').unpack()
    #ts, ampl, phase = kids[idx].get_cache('rewind_tod').unpack()
    ts = hdus['READOUT'].data['timestamp']
    ampl, phase, linphase = hdus['READOUT'].data['Amp, Ph, linPh %d' %idx].T # same as rewind_tod
    slices = []
    beg = 0
    interpfreq = np.zeros(len(ts))
    for i, t in enumerate(ts):
        idx = np.where(t>tempts)[0]
        if len(idx)>len(slices):
            slices.append(slice(beg, i))
            beg = i
        interpfreq[i] = tempfreq[len(slices)]
    print( len(slices), len(tempts), len(ts) )
    kids.other_tods['THzfrequency'] = bbs.TemperatureTODData(ts, interpfreq)

    with open(os.path.join(outdir, 'THzFrequencyTOD_slices.dat'), 'w') as f:
        print( "# slice# index_from index_below freq (GHz)", file=f )
        for i, s in enumerate(slices):
            print( i, s.start, s.stop, tempfreq[i], file=f )
    
    if force and kids.has_cache('set_slice'):
        del kids.__dict__['_path_set_slice']
    kids.set_slice(slices)
    kids.save()

    ##### calculate response
    from util import rebin_array
    #dt = 1./kids.header['framert']
    dt = 1./hdus['READOUT'].header['FRAMERT']
    kid_list, std_list = [], []
    num_peak_check_list = []
    for i, kid in kids.items():
        if not kid.enabled: continue
        
        #ts, ampl, phase = kid.get_cache('deglitch').unpack()
        #ts, ampl, phase = kid.get_cache('rewind_tod').unpack()
        ampl, phase, linphase = hdus['READOUT'].data['Amp, Ph, linPh %d' %i].T # same as rewind_tod

        ##### responsivity from df/fr
        r = kid.get_cache('fit')
        Qr = r.params['Qr'].value
        df = kid.convert_to_fshift(phase, opt='fshift')

        if force and kid.has_cache('other_tods'):
            del kid.__dict__['_path_other_tods']
        #kid.other_tods['df/f'] = bbs.PowerTODData(ts, df)
        
        THzfreq, resp, resp_df = [], [], []
        for j, s in enumerate(slices):
            # assume time duration at one frequency is more than tcut (sec)
            # throw away first part of data for waiting for frequency stabilization
            idx = np.where((ts[s]-ts[s][0])>tcut)[0]
            if len(idx)>0:
                x_, y_ = calcfft(phase[s][idx], dt)
                xdf_, ydf_ = calcfft(df[s][idx], dt)
            else:
                print( i, j, s, len(phase[s]), len(df[s]) )
                continue
            
            # assume THz bias modulation of 11.92 Hz
            idx = np.where( (x_>10.)&(x_<14.) )[0]
            if len(idx)>0:
                THzfreq.append( tempfreq[j] )
                resp.append( np.amax(y_[idx]) )
                resp_df.append( np.amax(ydf_[idx]) )
            else:
                print( i, s, len(phase[s]), len(df[s]) )
                print( x_, y_ )
                print( xdf_, ydf_ )
        THzfreq = np.array(THzfreq)
        resp = np.array(resp)
        resp_df = np.array(resp_df)
        
        ## make lists for the reference.dat
        ind_leftside = np.argmin(np.abs(THzfreq-200.0))
        ind_rightside = np.argmin(np.abs(THzfreq-450.0))
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(resp[ind_leftside:ind_rightside].reshape(-1, 1))
        scaler = MinMaxScaler()
        normalized_data_df = scaler.fit_transform(resp_df[ind_leftside:ind_rightside].reshape(-1, 1))
        peaks, _ = find_peaks(normalized_data_df.ravel(), height=0.4, distance=30)
        median = np.nanmedian(normalized_data)
        normalized_data[normalized_data>median] = np.nan
        std = np.nanstd(normalized_data)
        if len(peaks)>1 or len(peaks)==0:
            num_peak_check_list.append(True) # BAD
        else:
            num_peak_check_list.append(False) # GOOD
        #print("np.nanstd(normalized_data) = ", std)
        kid_list.append(i)
        std_list.append(std)
        ## rebin in frequency direction
        rebin = 1
        rebin_THzfreq = rebin_array(THzfreq, rebin)
        rebin_resp = rebin_array(resp, rebin)
        rebin_resp_df = rebin_array(resp_df, rebin)
        
        if i%10==0:
            print( 'done for kid[%d]' %i )
        kid.other_tods['THzfrequency'] = bbs.PowerTODData(rebin_THzfreq, rebin_resp)
        kid.other_tods['THzfrequency_df'] = bbs.PowerTODData(rebin_THzfreq, rebin_resp_df)
        kids.save()
        for tmp0, tmp1 in kid.__dict__.copy().items():
            if tmp0[:6] == '_data_':
                del kid.__dict__[tmp0]

    hdus.close()
    
    ## save npys for the reference.dat
    kid_list, std_list = np.array(kid_list), np.array(std_list)
    num_peak_check_list = np.array(num_peak_check_list)
    np.save(os.path.join(outdir, "kid_list_THzFrequencyTOD.npy"), kid_list)
    np.save(os.path.join(outdir, "std_list_THzFrequencyTOD.npy"), std_list)
    for i, s in zip(kid_list, std_list):
        print("kid, badness_x1e5 = ", i, s*1e5)
    #Q1 = np.percentile(std_list, 25)
    #Q3 = np.percentile(std_list, 75)
    #IQR = Q3 - Q1
    #lower_bound = Q1 - REFVALUE * IQR
    #upper_bound = Q3 + REFVALUE * IQR
    #print("Q1*1e5, Q3*1e5, IQR*1e5 = ", Q1*1e5, Q3*1e5, IQR*1e5)
    std_list_median = np.nanmedian(std_list)
    std_list_std = np.nanstd(std_list)
    upper_bound_1st = std_list_median+std_list_std*REFVALUE
    bad_mask_1st = std_list > upper_bound_1st
    std_list_median = np.nanmedian(std_list[~bad_mask_1st])
    std_list_std = np.nanstd(std_list[~bad_mask_1st])
    upper_bound_2nd = std_list_median+std_list_std*REFVALUE
    bad_mask_2nd = std_list > upper_bound_2nd
    bad_mask, upper_bound = bad_mask_2nd, upper_bound_2nd
    print("REFVALUE = ", REFVALUE)
    print("upper_bound*1e5 = ", upper_bound*1e5)
    bad_mask = std_list > upper_bound
    for i in range(10):
        if bad_mask[-10+i]==False:
            continue
        else:
            bad_mask[-10+i:] = True
            break
    bad_mask = bad_mask | num_peak_check_list
    outliers = std_list[bad_mask]
    outliers_id = kid_list[bad_mask]    
    #top_indices = np.argsort(std_list)[-NUM_REFERENCE:][::-1]
    #top_x = sorted(kid_list[top_indices])
    f = open(os.path.join(outdir, "reference.dat"), "w")
    f.write("# reference\n")
    for kid_num in outliers_id:
        f.write(str(kid_num)+"\n")
    f.close()
    f = open(os.path.join(outdir, "badness_x100000.csv"), "w")
    for i in range(len(kid_list)):
        f.write(str(kid_list[i])+"\t"+str(round(std_list[i]*1e5, 3))+"\n")
    f.close()
    fig = plt.figure(figsize=(16, 8))
    plt.scatter(kid_list, std_list)
    plt.scatter(outliers_id, outliers)
    plt.scatter(kid_list[num_peak_check_list], std_list[num_peak_check_list], marker="x", c="lime")
    plt.plot([np.nanmin(kid_list), np.nanmax(kid_list)], [upper_bound, upper_bound], linestyle="-", c="r")
    plt.xlabel("ID")
    plt.ylabel("BADNESS")
    plt.savefig(os.path.join(outdir, "reference.pdf"))
    plt.savefig(os.path.join(outdir, "reference.png"), dpi=200)
    plt.clf()

def Plot(kids, THzsweepfile, tcut, skiprows, force, test, NCPU):
    ## plot
    print( 'THzFrequencyTOD::Plot()...' )

    plotdir = outdir + '/figTHzFrequencyTOD'
    if force:
        try:
            shutil.rmtree(plotdir)
        except OSError:
            pass
    #### make plot output directory
    try:
        os.mkdir(plotdir)
    except:
        pass

    ###### read reduced fits file
    ifname = os.path.join(outdir, 'reduced_' + os.path.basename(kids._tods_path))
    hdus = fits.open(ifname)

    ts, interpfreq = kids.other_tods['THzfrequency'].unpack()
    slices = kids.get_cache('set_slice')
    #dt = 1./kids.header['framert']
    dt = 1./hdus['READOUT'].header['FRAMERT']
    from matplotlib.backends.backend_pdf import PdfPages
    
    from multiprocessing import Pool
    p = Pool(NCPU)
    params_list = []
    for i, kid in kids.items():
        if not kid.enabled: continue
        ampl, phase, linphase = copy.deepcopy(hdus['READOUT'].data['Amp, Ph, linPh %d' %i].T)
        params_list.append({"i": i, "kid": kid, "plotdir": plotdir,"ampl": ampl, "phase": phase, "linphase": linphase, "ts": ts,"interpfreq": interpfreq})
    res = p.map(plot_multi, params_list)
    
    
    
    #####
    print( 'plotting all KIDs...' )
    import matplotlib
    colormap = matplotlib.cm.nipy_spectral
    fig = plt.figure(figsize=(16,8))
    for i, kid in kids.items():
        if not kid.enabled: continue
        freq, resp = kid.other_tods['THzfrequency'].unpack()   
        freq_, resp_ = kid.other_tods['THzfrequency_df'].unpack()

        plt.subplot(221)
        plt.plot(freq, resp, color=colormap(float(i)/len(kids)))
        plt.subplot(222)
        plt.plot(freq, resp-resp[0]+i*1., color=colormap(float(i)/len(kids)))
        #
        plt.subplot(223)
        plt.plot(freq_, resp_, color=colormap(float(i)/len(kids)))
        plt.subplot(224)
        plt.plot(freq_, resp_-resp_[0]+i*1e-5, color=colormap(float(i)/len(kids)))
        
        for tmp0, tmp1 in kid.__dict__.copy().items():
            if tmp0[:6] == '_data_':
                del kid.__dict__[tmp0]
    plt.subplot(221)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Raw Phase Response')
    plt.subplot(222)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Raw Phase Response + Offset')
    ax = plt.subplot(223)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Response from df/f')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = plt.subplot(224)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Response from df/f + Offset')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.savefig(os.path.join(plotdir, 'THzFrequencyTOD_all.png'))
    plt.close(fig)

    #####
    print( 'plotting photo current...' )
    readtxt = my_loadtxt(THzsweepfile, unpack=True, skiprows=skiprows, delimiter=' ')
    tempts = readtxt[0]
    tempfreq   = readtxt[2]
    photocurrent = readtxt[3]**2
    if len(readtxt)>4:
        cnt = readtxt[4]

    dt = tempts[1:] - tempts[:-1]
    print( np.average(dt), tempts[1]-tempts[0] )

    fig = plt.figure(figsize=(8,10))
    plt.subplot(311)
    if np.average(photocurrent)>0.:
        plt.semilogy(tempfreq, photocurrent)
    else:
        plt.plot(tempfreq, photocurrent)
    plt.ylim(ymin=1e-4)
    plt.xlabel('THz frequency (GHz)')
    plt.grid()

    plt.subplot(312)
    if np.average(photocurrent)>0.:
        plt.semilogy(tempts-tempts[0], photocurrent)
    else:
        plt.plot(tempts-tempts[0], photocurrent)
    plt.ylim(ymin=1e-4)
    plt.xlabel('sec')
    plt.grid()

    if np.average(photocurrent)>0.:
        size = 2**int( np.floor( np.log2(len(tempts)) ) )
        #f_, v_ = calcfft(photocurrent[:size], np.average(dt))
        f_, v_ = md.power_spectrum_density(photocurrent[:size], np.average(dt), 7, window=None, overwrap_half=True)
        plt.subplot(313)
        plt.semilogx(f_, np.log10(v_)*10.0)
        plt.xlabel('Hz')
        plt.grid()

    plt.savefig(os.path.join(plotdir, 'check_photocurrent.png'))
    plt.close(fig)

    
def plot_multi(params):
    i = params["i"]
    kid = params["kid"]
    plotdir = params["plotdir"]
    #hdus = params["hdus"]
#for i, kid in kids.items():
    #if not kid.enabled: continue

    #if test and i>10: break
    print( 'plotting KID[%d]' % i )
    #ts, ampl, phase = kid.get_cache('deglitch').unpack()
    #ts, ampl, phase = kid.get_cache('rewind_tod').unpack()
    ts = params["ts"]
    interpfreq = params["interpfreq"]
    #ampl, phase, linphase = hdus['READOUT'].data['Amp, Ph, linPh %d' %i].T # same as rewind_tod
    ampl, phase, linphase = params["ampl"], params["phase"], params["linphase"]
    ##### responsivity from df/fr
    r = kid.get_cache('fit')
    Qr = r.params['Qr'].value
    df = kid.convert_to_fshift(phase, opt='fshift')

    #interval = 10
    interval = 1
    fig = plt.figure(figsize=(10,14))
    plt.subplot(411)
    ax1 = plt.gca()
    plt.title('KID[%d]' % i)
    plt.plot(ts[::interval], ampl[::interval], label='<-Amplitude')
    plt.plot(ts[::interval], phase[::interval], label='<-Phase')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Response')
    ax2 = ax1.twinx()
    plt.plot(ts[::interval], interpfreq[::interval], '+-k', label='THzFrequency->')
    plt.ylabel('Frequency')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.grid()
    ax1.legend(h1+h2, l1+l2, loc='best')

    plt.subplot(412)
    ax1 = plt.gca()
    plt.plot(ts[::interval], df[::interval], label='<-From df/f')
    plt.xlabel('Time [s]')
    plt.ylabel('Response')
    ax2 = ax1.twinx()
    plt.plot(ts[::interval], interpfreq[::interval], '+-k', label='THzFrequency->')
    plt.ylabel('Frequency')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.grid()
    ax1.legend(h1+h2, l1+l2, loc='best')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.subplot(413)
    #plt.plot(interpfreq[::interval], ampl[::interval], 'b', label='Amplitude')
    #plt.plot(interpfreq[::interval], phase[::interval], 'g+', label='Phase')
    freq, resp = kid.other_tods['THzfrequency'].unpack()
    plt.plot(freq, resp, 'r', label='From phase')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Response')
    plt.grid()
    plt.legend(loc='best')
    #scaler = MinMaxScaler()
    #normalized_data = scaler.fit_transform(resp.reshape(-1, 1))
    #median = np.nanmedian(normalized_data)
    #normalized_data[normalized_data>median] = np.nan
    #std = np.nanstd(normalized_data)
    #print("np.nanstd(normalized_data) = ", std)
    #kid_list.append(i)
    #std_list.append(std)
    
    ax = plt.subplot(414)
    freq, resp = kid.other_tods['THzfrequency_df'].unpack()
    plt.plot(freq, resp, 'r', label='From df/f')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Response')
    plt.grid()
    plt.legend(loc='best')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.savefig(os.path.join(plotdir, 'THzFrequencyTOD_%04d.png' % i))
    #plt.close(fig)
    plt.clf()
    plt.close()

    ##
    '''
    pp = PdfPages(os.path.join(plotdir, 'checkresponse_%04d.pdf' %i))
    #displaystep = int(len(slices)/10.)
    displaystep = int(len(slices)/5.)
    display = 1
    fig = plt.figure(1,figsize=(16,10))
    for j, s in enumerate(slices):
        if j==display or j==display-1 or j==display+1:
            # assume time duration at one frequency is more than tcut (sec)
            # throw away first part of data for waiting for frequency stabilization
            idx = np.where((ts[s]-ts[s][0])>tcut)[0]
            x_, y_ = calcfft(phase[s][idx], dt)
            xdf_, ydf_ = calcfft(df[s][idx], dt)

            plt.figure(1)
            plt.subplot(221)
            plt.plot(ts[s]-ts[0], phase[s], 'x-')
            plt.subplot(222)
            plt.plot(x_, y_, 'x-')
            plt.subplot(223)
            plt.plot(ts[s]-ts[0], df[s], 'x-')
            plt.subplot(224)
            plt.plot(xdf_, ydf_, 'x-')
            
            fig_ = plt.figure(figsize=(16,10))
            plt.subplot(221)
            plt.title('raw phase vs time')
            #plt.plot(ts[s]-ts[0], phase[s]-np.average(phase[s]), 'x-')
            plt.plot(ts[s]-ts[0], phase[s], 'x-')
            plt.plot(ts[s][idx]-ts[0], phase[s][idx], 'o-', label='Used for FFT')
            plt.legend(loc='best')
            plt.subplot(222)
            plt.title('FFT of raw phase')
            plt.plot(x_, y_, 'x-')
            ax = plt.subplot(223)
            plt.title('df/f vs time')
            plt.plot(ts[s]-ts[0], df[s], 'x-')
            plt.plot(ts[s][idx]-ts[0], df[s][idx], 'o-', label='Used for FFT')
            plt.legend(loc='best')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax = plt.subplot(224)
            plt.title('FFT of df/f')
            plt.plot(xdf_, ydf_, 'x-')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            pp.savefig(fig_)
            plt.close(fig_)
            
            if j==display+1:
                display += displaystep
    plt.figure(1)
    plt.subplot(221)
    plt.title('raw phase vs time')
    plt.subplot(222)
    plt.title('FFT of raw phase')
    ax = plt.subplot(223)
    plt.title('df/f vs time')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = plt.subplot(224)
    plt.title('FFT of df/f')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    pp.savefig(fig)
    pp.close()
    #plt.close(fig)
    plt.clf()
    plt.close()
    '''
        
    for tmp0, tmp1 in kid.__dict__.copy().items():
        if tmp0[:6] == '_data_':
            del kid.__dict__[tmp0]

    del ampl, phase, linphase, df, freq, resp, fig
    gc.collect()
    
    #hdus.close()
    
    #kid_list, std_list = np.array(kid_list), np.array(std_list)
    #np.save(os.path.join(plotdir, 'kid_list_THzFrequencyTOD.npy'), kid_list)
    #np.save(os.path.join(plotdir, 'std_list_THzFrequencyTOD.npy'), std_list)



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)
    import mkid_data as md
    
    from common import *
    import scipy.interpolate
    import scipy.fftpack
    from scipy.signal import find_peaks
    from astropy.io import fits
    from sklearn.preprocessing import MinMaxScaler

    main()

