#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import shutil
import os

def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=myHelpFormatter, description=__doc__)
    parser.add_argument('--mode', choices=['1', '2', 'all'], default='1',
                        help="""select analysis mode, or all""") # for analysis
    parser.add_argument('--rebin', type=int, default=1,
                        help='rebin for 1st and 2nd analysis')
    parser.add_argument('--refid', type=int, default=None,
                        help='reference kidid for 2nd analysis')    
    parser.add_argument('--force', action='store_true') # for plot
    args = parser.parse_args(argv)

    #rebin = 5
    rebin = args.rebin
    reftxtlist = outdir + '/reference.dat'
    if args.mode=='1' or args.mode=='all':
        first_ana(kids, reftxtlist, rebin, args.force)
    elif args.mode=='2' or args.mode=='all':
        second_ana(kids, reftxtlist, args.refid, rebin, args.force)

#############################################################
def lorentzian(x,v):
    numerator =  v['hwhm']**2
    #numerator =  v['hwhm']
    denominator = (x-v['peak_center'])**2 + v['hwhm']**2
    y = v['intensity']*(numerator/denominator) + v['offset']
    #y = v['intensity']/np.pi*(numerator/denominator) + v['offset']
    return y

def deriv_lorentzian(x, v, i):
    p0 = v['hwhm']
    p1 = v['peak_center']
    p2 = v['intensity']
    p3 = v['offset']
    A = (x-p1)**2/(p0**2)

    if i==0:
        pderiv = 2.*A/p0 * p2/(A+1)**2
    elif i==1:
        pderiv = 2.*A/(x-p1) * p2/(A+1)**2
    elif i==2:
        pderiv = 1./(A+1)
    elif i==3:
        pderiv = 1.*np.ones(len(x))

    return pderiv

def residuals(params, x, data=None, eps=None):
    v = params.valuesdict()
    err = data - lorentzian(x,v)

    return err

def fit_lorentzian(x, y):
    # initial values: hwhm, peak center, intensity, offset
    params = lmfit.Parameters()
    params.add('hwhm', value=1., min=0)
    params.add('peak_center', value=x[np.argmax(y)])
    params.add('intensity', value=np.amax(y))
    #params.add('offset', value=x[np.argmin(y)], min=0)
    params.add('offset', value=x[np.argmin(y)])

    #minimized = lmfit.minimize(residuals, params, method='leastsq', args=(x,), kws={'data':y})
    mini = lmfit.Minimizer(residuals, params, fcn_args=(x,), fcn_kws={'data':y})
    result = mini.minimize(method='leastsq')
    #ci = lmfit.conf_interval(mini, result)

    #return result, ci
    return result

def GetBestParArray(kidid, result_fit):
    result_FWHM, result_height, result_offset, result_Q, result_RF = [], [], [], [], []
    for i,kid in enumerate(kidid):
        best_parameters = result_fit[i]

        # fit result
        F0 = best_parameters['peak_center'].value
        dF0 = best_parameters['peak_center'].stderr
        FWHM = 2*best_parameters['hwhm'].value
        dFWHM = 2*best_parameters['hwhm'].stderr
        Q = F0/FWHM
        dQ = Q*np.sqrt( (dF0/F0)**2 + (dFWHM/FWHM)**2 )
        height = best_parameters['intensity'].value
        dheight = best_parameters['intensity'].stderr
        offset = best_parameters['offset'].value
        doffset = best_parameters['offset'].stderr
        rejection = height/offset
        drejection = rejection*np.sqrt( (dheight/height)**2 + (doffset/offset)**2 )

        result_FWHM.append( (FWHM,dFWHM) )
        result_height.append( (height,dheight) )
        result_offset.append( (offset,doffset) )
        result_Q.append( (Q,dQ) )
        result_RF.append( (rejection,drejection) )

    return np.array([result_FWHM, result_height, result_offset, result_Q, result_RF])

def fit_and_plot(freq, resp, refresp, rebin, ikid, refidx, display, opt):
    from util import rebin_array
    ## rebin in frequency direction and fit with Lorentzian
    freq_ = rebin_array(freq, rebin)
    resp_ = rebin_array(resp, rebin)
    if refidx<0:
        signal = resp_
    else:
        refresp_ = rebin_array(refresp, rebin)
        signal = resp_/refresp_

    result = fit_lorentzian(freq_, signal) #hwhm, peak center, intensity, offset
    best_parameters = result.params

    ## check fit result
    F0 = best_parameters['peak_center'].value
    dF0 = best_parameters['peak_center'].stderr
    FWHM = 2*best_parameters['hwhm'].value
    Q = F0/FWHM
    offset = best_parameters['offset'].value

    check = 0
    #if FWHM>0 and Q<5000. and offset>0.:
    if FWHM>0 and Q<5000.:
        check = 1

    if display>=0:
        # partial derivatives df/dp of lorentzian with respect to each parameter
        dfdp = [ deriv_lorentzian(freq_, best_parameters.valuesdict(), j) for j in range(len(best_parameters)) ]
        confprob = 0.95
        #fit = lorentzian(freq_, best_parameters.valuesdict())
        upper, lower = None, None
        if result.covar is not None:
            fit, upper, lower = confidence_band(freq_, dfdp, confprob, result, lorentzian)
        else:
            fit = lorentzian(freq_, best_parameters.valuesdict())
            
        ## plot
        plt.figure(1)
        ax = plt.subplot(4,1,display%4+1)
        plt.plot(freq_, signal, '-')
        plt.plot(freq_, fit, 'r-', alpha=0.5, lw=3, label='F0=%.2lf GHz\nQ=%.2lf' %(F0,Q))
        if upper is not None:
            plt.fill_between(freq_, lower, upper, color='g', alpha=0.2, label=r'%.0f %% CL band' %(confprob*100))
        plt.legend(loc='best')
        plt.ylabel('%s Response' %opt)
        if refidx<0 and display%4==0:
            plt.title('KID[%d], No Reference' %ikid)
        elif refidx<0:
            plt.title('No Reference')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        elif display%4==0:
            plt.title('KID[%d], Ref: KID[%d]' %(ikid,refidx))
        else:
            plt.title('Ref: KID[%d]' %refidx)
            
    return (check, F0, dF0, best_parameters)

#############################################################

def first_ana(kids, reftxtlist, rebin, force):
    if os.path.exists(reftxtlist):
        print( 'reading: %s...' %reftxtlist )
    else:
        print( '%s does not exist!' %reftxtlist )
        print( 'quit' )
        sys.exit()
    readtxt = my_loadtxt(reftxtlist, unpack=True, skiprows=1)
    reflist = readtxt[0]
    #print reflist

    ## plot
    plotdir = outdir + '/figAnaSpectrum1st'
    if force:
        try:
            shutil.rmtree(plotdir)
        except OSError:
            pass
    #### make plot output directory
    os.mkdir(plotdir)

    for i, kid in kids.items():
        if not kid.enabled: continue
                
        print( 'plotting KID[%d]... with different references' %i )
        pp = PdfPages(os.path.join(plotdir, 'AnaSpectrum_%04d.pdf' %i))
        display = 0
        kidid, result_F0, result_fit = [], [], []
        kidid_df, result_F0_df, result_fit_df = [], [], []

        fig = plt.figure(1, figsize=(10,10))
        ##### no reference
        freq, resp = kid.other_tods['THzfrequency'].unpack()
        check, F0, dF0, best_parameters = fit_and_plot(freq, resp, 0, rebin, i, -1, display, 'Raw Phase')
        ## check fit result
        if check>0:
            kidid.append( -1 )
            result_F0.append( (F0,dF0) )
            result_fit.append( best_parameters )
        display += 1

        ##
        freq_df, resp_df = kid.other_tods['THzfrequency_df'].unpack()
        check, F0, dF0, best_parameters = fit_and_plot(freq_df, resp_df, 0, rebin, i, -1, display, 'df/f')
        if check>0:
            kidid_df.append( -1 )
            result_F0_df.append( (F0,dF0) )
            result_fit_df.append( best_parameters )
        display += 1

        ##### with reference
        for refidx in reflist:
            if refidx<0: continue
            refkid = kids[refidx]
            if not refkid.enabled: continue
            if i==refidx: continue
            
            if display%4==0:
                fig = plt.figure(1, figsize=(10,10))

            reffreq, refresp = refkid.other_tods['THzfrequency'].unpack()
            check, F0, dF0, best_parameters = fit_and_plot(freq, resp, refresp, rebin, i, refidx, display, 'Raw Phase')
            if check>0:
                kidid.append( refidx )
                result_F0.append( (F0,dF0) )
                result_fit.append( best_parameters )
            display += 1

            ##
            reffreq_df, refresp_df = refkid.other_tods['THzfrequency_df'].unpack()
            check, F0, dF0, best_parameters = fit_and_plot(freq_df, resp_df, refresp_df, rebin, i, refidx, display, 'df/f')
            if check>0:
                kidid_df.append( refidx )
                result_F0_df.append( (F0,dF0) )
                result_fit_df.append( best_parameters )

            if display%4==3:
                plt.xlabel('Frequency [GHz]')
                pp.savefig(fig)
                #plt.close(fig)
                plt.clf()
                plt.close()
            display += 1

            for tmp0, tmp1 in refkid.__dict__.copy().items():
                if tmp0[:6] == '_data_':
                    del refkid.__dict__[tmp0]
        if display%4>0:
            plt.figure(1)
            plt.xlabel('Frequency [GHz]')
            pp.savefig(fig)
            #plt.close(fig)
            plt.clf()
            plt.close()

        ##### plot refid vs results
        kidid = np.array(kidid)
        result_F0 = np.array(result_F0)
#        Plot(1, kidid, result_F0, result_fit, pp, 'Raw Phase')
        #
        kidid_df = np.array(kidid_df)
        result_F0_df = np.array(result_F0_df)
#        Plot(1, kidid_df, result_F0_df, result_fit_df, pp, 'df/f')
        pp.close()
                
        for tmp0, tmp1 in kid.__dict__.copy().items():
            if tmp0[:6] == '_data_':
                del kid.__dict__[tmp0]


def second_ana(kids, reftxtlist, refidx, rebin, force):
    if os.path.exists(reftxtlist):
        print( 'reading: %s...' %reftxtlist )
    else:
        print( '%s does not exist!' %reftxtlist )
        print( 'quit' )
        sys.exit()
    readtxt = my_loadtxt(reftxtlist, unpack=True, skiprows=1)
    reflist = readtxt[0]
    #print reflist

    ## plot
    plotdir = outdir + '/figAnaSpectrum2nd'
    if force:
        try:
            shutil.rmtree(plotdir)
        except OSError:
            pass
    #### make plot output directory
    os.mkdir(plotdir)

    #### reference if exist
    if refidx is not None:
        refkid = kids[refidx]
        if not refkid.enabled:
            print( 'Refid %d is disabled' %refidx )
        _, refresp = refkid.other_tods['THzfrequency'].unpack()
        _, refresp_df = refkid.other_tods['THzfrequency_df'].unpack()

    ##### fit
    from util import rebin_array
    kidid, result_F0, result_fit, result_F0_df, result_fit_df = [], [], [], [], []
    Fsubmm, Spectrum, Spectrum_df = [], [], []
    for i, kid in kids.items():
        if not kid.enabled: continue

        if refidx is None:
            print( 'fitting KID[%d]... with no reference' %i )
            ##### no reference
            freq, resp = kid.other_tods['THzfrequency'].unpack()
            check, F0, dF0, best_parameters = fit_and_plot(freq, resp, 0, rebin, i, -1, -1, 'Raw Phase')
            
            freq_df, resp_df = kid.other_tods['THzfrequency_df'].unpack()
            check_df, F0_df, dF0_df, best_parameters_df = fit_and_plot(freq_df, resp_df, 0, rebin, i, -1, -1, 'df/f')
        else:
            print( 'fitting KID[%d]... with reference %d' %(i,refidx) )
            ##### with reference
            freq, resp = kid.other_tods['THzfrequency'].unpack()
            check, F0, dF0, best_parameters = fit_and_plot(freq, resp, refresp, rebin, i, refidx, -1, 'Raw Phase')

            freq_df, resp_df = kid.other_tods['THzfrequency_df'].unpack()
            check_df, F0_df, dF0_df, best_parameters_df = fit_and_plot(freq_df, resp_df, refresp_df, rebin, i, refidx, -1, 'df/f')

        ## check fit result
        #if True:
        if check>0 and check_df>0:
            kidid.append( i )
            result_F0.append( (F0,dF0) )
            result_fit.append( best_parameters )
            result_F0_df.append( (F0_df,dF0_df) )
            result_fit_df.append( best_parameters_df )

            rebin_freq = rebin_array(freq, rebin)
            rebin_resp = rebin_array(resp, rebin)
            rebin_resp_df = rebin_array(resp_df, rebin)
            Fsubmm.append(rebin_freq)
            Spectrum.append(rebin_resp)
            Spectrum_df.append(rebin_resp_df)

        for tmp0, tmp1 in kid.__dict__.copy().items():
            if tmp0[:6] == '_data_':
                del kid.__dict__[tmp0]

    ##### write results
    kidid = np.array(kidid)
    result_F0, result_F0_df = np.array([result_F0, result_F0_df])
    """
    print(type(Fsubmm[0]))
    print(type(Spectrum[0]))
    print(type(result_fit[0]))
    print(type(Spectrum_df[0]))
    print(type(result_fit_df[0]))
    Fsubmm = np.array(Fsubmm)
    Spectrum = np.array(Spectrum)
    result_fit = np.array(result_fit).astype(object)
    Spectrum_df = np.array(Spectrum_df)
    result_fit_df = np.array(result_fit_df).astype(object)
    print(Fsubmm.shape)
    print(Spectrum.shape)
    print(result_fit.shape)
    print(Spectrum_df.shape)
    print(result_fit_df.shape)
    
    print(type(Fsubmm[0]))
    print(type(Spectrum[0]))
    print(type(result_fit[0]))
    print(type(Spectrum_df[0]))
    print(type(result_fit_df[0]))
    """
    """
    #Fsubmm, Spectrum, result_fit, Spectrum_df, result_fit_df = np.array([Fsubmm, Spectrum, result_fit, Spectrum_df, result_fit_df])
    np.save(outdir+'/AnaSpectrum2nd_fit_all.npy',
            np.array([kidid, Fsubmm.tolist(), Spectrum.tolist(), result_fit.tolist(), Spectrum_df.tolist(), result_fit_df.tolist()], dtype=object))
#            np.array([kidid, Fsubmm.tolist(), Spectrum.tolist(), result_fit.tolist(), Spectrum_df.tolist(), result_fit_df.tolist()]))
    """

    ##### delete references
    from util import delete_elements
    lrefidx = []
    for i,kid in enumerate(kidid):
        if kid in reflist: lrefidx.append(i)
    print( 'delete references...:', kidid[lrefidx] )
    kidid, result_F0, result_fit, result_F0_df, result_fit_df, Fsubmm, Spectrum, Spectrum_df\
        = delete_elements([kidid, result_F0, result_fit, result_F0_df, result_fit_df, Fsubmm, Spectrum, Spectrum_df], lrefidx)
    """
    np.save(outdir+'/AnaSpectrum2nd_fit.npy',
            np.array([kidid, Fsubmm.tolist(), Spectrum.tolist(), result_fit.tolist(), Spectrum_df.tolist(), result_fit_df.tolist()], dtype=object))
#            np.array([kidid, Fsubmm.tolist(), Spectrum.tolist(), result_fit.tolist(), Spectrum_df.tolist(), result_fit_df.tolist()]))
    """
    ##### plot
    result_index = np.argsort(result_F0_df.T[0])
    print()
    print( 'Sorted index:' )
    print( len(result_index), result_index )
    ##
    import matplotlib
    colormap = matplotlib.cm.nipy_spectral
    #colormap = matplotlib.cm.prism
    figall = plt.figure(1, figsize=(8,10))
#    figallzoom = plt.figure(2, figsize=(8,10))
    for i,kid in enumerate(kidid):
        #ii = len(kidid)-1 - i
        ii = result_index[i]
        coli = len(kidid)-1 - i
        freq = Fsubmm[ii]
        if refidx is None:
            signal = Spectrum_df[ii]
        else:
            #signal = Spectrum_df[ii]/refresp_df
            signal = Spectrum_df[ii]/rebin_array(refresp_df, rebin)
        
        #hwhm, peak center, intensity, offset
        best_parameters = result_fit_df[ii]
        fit = lorentzian(freq, best_parameters.valuesdict())
        
        # fit result
        height = best_parameters['intensity'].value
        offset = best_parameters['offset'].value

        #norm = offset
        #norm = height
        norm = np.amax(signal)
        plotoffset = 0.1
        #norm = offset*maxheight
        if offset<0.: print( 'Negative offset:', i, kid, height, offset )
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(freq, signal/norm, '--', color=colormap(float(coli)/len(kidid)), lw=0.5)
        plt.plot(freq, fit/norm, '-', color=colormap(float(coli)/len(kidid)), lw=1.5)
        plt.subplot(212)
        #plt.plot(freq, signal/norm+ii, '-', color=colormap(float(ii)/len(kidid)), lw=1.5)
        #plt.plot(freq, fit/norm+ii, '-k', lw=1.5)
        plt.plot(freq, signal/norm+coli*plotoffset, '-', color=colormap(float(coli)/len(kidid)), lw=1.5)
        plt.plot(freq, fit/norm+coli*plotoffset, '-k', lw=1.5)

        ##### zoomed
        plt.figure(2)
        plt.subplot(211)
        plt.plot(freq, signal/norm, '--', color=colormap(float(coli)/len(kidid)), lw=0.5)
        plt.plot(freq, fit/norm, '-', color=colormap(float(coli)/len(kidid)), lw=1.5)
        plt.subplot(212)
        plt.plot(freq, signal/norm+coli*plotoffset, '-', color=colormap(float(coli)/len(kidid)), lw=1.5)
        plt.plot(freq, fit/norm+coli*plotoffset, '-k', lw=1.5)

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Response (a.u.)')
    plt.subplot(212)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Response (a.u.) + Offset')
    plt.savefig(os.path.join(plotdir, 'AnaSpectrum_df_all.png'))
    plt.close(figall)

    ##### zoomed
#    plt.figure(2)
#    plt.subplot(211)
#    plt.xlabel('Frequency [GHz]')
#    plt.ylabel('Response (a.u.)')
#    plt.xlim(310., 380.)
#    #plt.xlim(570.,730.)
#    plt.subplot(212)
#    plt.xlabel('Frequency [GHz]')
#    plt.ylabel('Response (a.u.) + Offset')
#    plt.xlim(310., 380.)
#    #plt.xlim(570.,730.)
#    plt.savefig(os.path.join(plotdir, 'AnaSpectrum_df_all.zoomed.png'))
#    plt.close(figallzoom)

    #####
    ##### plot kidid vs results
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(os.path.join(plotdir, 'AnaSpectrum_fit.pdf'))
    Plot(2, kidid, result_F0, result_fit, pp, 'Raw Phase')
    Plot(2, kidid, result_F0_df, result_fit_df, pp, 'df/f')
    pp.close()

    print()
    print( '%d KIDs were plotted' %len(kidid) )
    dF = Fsubmm[0][1]-Fsubmm[0][0]
    print( 'dF = %lf [GHz]' %dF )
    ## check overlap in F0: should be 0.65-0.66 GHz interval
    print()
    print( 'Overlapped in submm frequencies (from df/f reponse):' )
    for i, F0 in enumerate(result_F0_df.T[0]):
        idx = np.where( abs(result_F0_df.T[0]-F0)<0.66 )[0]
        if len(idx)>1:
            print( kidid[i], F0, ':', kidid[idx], result_F0_df[idx] )


def Plot(mode, kidid, result_F0, result_fit, pp, opt):
    result_FWHM, result_height, result_offset, result_Q, result_RF = GetBestParArray(kidid, result_fit)

    ## F0
    if mode==1:
        fig = plt.figure(2, figsize=(8,10))
        plt.figure(2)
        plt.subplot(4,1,1)
        plt.title('Results of %s response' %opt)
    elif mode==2:
        fig = plt.figure()
    plt.errorbar(kidid, result_F0.T[0], yerr=result_F0.T[1], fmt='o', label='F0 [GHz]')
    #plt.errorbar(kidid, result_F0.T[0], yerr=result_F0.T[1], fmt='.', label='F0 [GHz]')
    plt.ylabel('Center Frequency [GHz]')
#    plt.xlim(-2, 100)
#    plt.ylim(300., 400.)
    #plt.ylim(500.,700.)
    if mode==2:
        plt.title('Results of %s response' %opt)
        plt.xlabel('KIDID')
        pp.savefig(fig)
        plt.close(fig)
    
    ## Q
    if mode==1:
        plt.subplot(4,1,2)
    elif mode==2:
        fig = plt.figure()
    plt.errorbar(kidid, result_Q.T[0], yerr=result_Q.T[1], fmt='o', label='Q')
    plt.yscale('log')
    plt.ylabel('Q value')
#    plt.xlim(-2, 100)
    if mode==2:
        plt.title('Results of %s response' %opt)
        plt.xlabel('KIDID')
        pp.savefig(fig)
        plt.close(fig)

    ## height
    if mode==1:
        plt.subplot(4,1,3)
    elif mode==2:
        fig = plt.figure()
    plt.errorbar(kidid, result_height.T[0], yerr=result_height.T[1], fmt='o', label='height')
    plt.yscale('log')
#    plt.xlim(-2, 100)
    if mode==1:
        plt.errorbar(kidid, result_offset.T[0], yerr=result_offset.T[1], fmt='o', label='offset')
        plt.ylabel('Height and Offset')
    elif mode==2:
        plt.title('Results of %s response' %opt)
        plt.ylabel('Height')
        plt.xlabel('KIDID')
        pp.savefig(fig)
        plt.close(fig)

    ## offset
    if mode==2:
        fig = plt.figure()
        plt.errorbar(kidid, result_offset.T[0], yerr=result_offset.T[1], fmt='o', label='offset')
        plt.title('Results of %s response' %opt)
        plt.xlabel('KIDID')
        plt.ylabel('Offset')
#        plt.xlim(-2, 100)
        pp.savefig(fig)
        plt.close(fig)

    ## RF
    if mode==1:
        plt.subplot(4,1,4)
    elif mode==2:
        fig = plt.figure()
    yerr_lower = result_RF.T[0]/(result_RF.T[0]-result_RF.T[1])
    idx = np.where(yerr_lower<=0)[0]
    yerr_lower[idx] = result_RF.T[0][idx]
    yerr_lower = 10.*np.log10(yerr_lower)
    yerr_upper = (result_RF.T[0]+result_RF.T[1])/result_RF.T[0]
    yerr_upper = 10.*np.log10(yerr_upper)
    plt.errorbar(kidid, 10.*np.log10(result_RF.T[0]), yerr=[yerr_lower, yerr_upper], fmt='o', label='rejection')
    plt.ylabel('Rejection [dB]')
#    plt.xlim(-2, 100)
    plt.ylim(ymin=0.)
    if mode==1:
        plt.xlabel('Reference KID')
        pp.savefig(fig)
        plt.close(fig)
    elif mode==2:
        plt.title('Results of %s response' %opt)
        plt.xlabel('KIDID')
        pp.savefig(fig)
        plt.close(fig)

    ## additional plots for mode==2
    if mode==2:
        fig = plt.figure(figsize=(8,10))

        plt.subplot(2,1,1)
        plt.errorbar(result_F0.T[0], result_Q.T[0], yerr=result_Q.T[1], xerr=result_F0.T[1], fmt='o', label='Q')
        plt.title('Results of %s response' %opt)
        plt.yscale('log')
        plt.xlabel('Center Frequency [GHz]')
        plt.ylabel('Q value')
        plt.grid()

        plt.subplot(2,1,2)
        plt.errorbar(result_F0.T[0], 10.*np.log10(result_RF.T[0]), yerr=[yerr_lower, yerr_upper], xerr=result_F0.T[1],
                     fmt='o', label='rejection')
        plt.xlabel('Center Frequency [GHz]')
        plt.ylabel('Rejection [dB]')
        plt.ylim(ymin=0.)
        plt.grid()
        pp.savefig(fig)
        plt.close(fig)
        #
        ## check F0 interval
        fig = plt.figure()
        F0sort = np.sort(result_F0.T[0])
        interval = F0sort[1:] - F0sort[:-1]
        #print
        #print 'Intervals:'
        #print interval
        #print
        #plt.hist(interval)
        plt.hist(interval, bins=50, range=(0.,5.))
        plt.title('Results of %s response' %opt)
        plt.xlabel('Interval of Center Frequnecy [GHz]')
        plt.xscale('log')
        pp.savefig(fig)
        plt.close(fig)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)

    from common import *
    import pandas as pd
    import scipy.optimize
    import lmfit
    from fit.confidence_band import confidence_band
    from matplotlib.backends.backend_pdf import PdfPages

    main()

