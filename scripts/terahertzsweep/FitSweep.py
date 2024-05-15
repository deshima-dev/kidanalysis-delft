#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Fit local sweeps and output fitting result.

input files:
 - (kids list file)
 - (local sweep file)
verbose input files:
 - (tod file)

output files:
 - out/FitSweep_result_%04d.dat
verbose output files:
 - out/FitSweep_%04d.png
"""

import os
import sys
import shutil

from collections import OrderedDict

import matplotlib.pyplot as plt
import mkid_data as md
import numpy as np

def get_kids_from_db(runid):
    import sqlite3
    dbname = '/Users/spacekids/database/kid_test.db'
    #dbname = '/Users/spacekids/database/kid_test.telesto.db'
    conn = sqlite3.connect(dbname,
                           detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    sqlite3.dbapi2.converters['DATETIME'] = sqlite3.dbapi2.converters['TIMESTAMP']
    c = conn.cursor()

    c.execute('''SELECT cache_path 
                 FROM analysis 
                 WHERE runid = ?
              ''',(runid,))
    d = c.fetchall()
    if len(d)==1:
        tmp_c_path = d[0][0]
        print( 'Teacher cache path:', tmp_c_path )
        kids = bbs.Cache.load(tmp_c_path)
    elif len(d)==0:
        raise Exception('No analysis for RUNID: {0}'.format(runid))
    else:
        raise Exception('Multiple analysis for RUNID: {0}'.format(runid))
    return kids

def get_fitter_name(fit_result):
    for fitter_name in md.fitters.all_fitters:
        tmpewa = getattr(md.fitters, 'fitter_' + fitter_name)
        if tmpewa.func.expr == fit_result._result.function.expr:
            return fitter_name
    raise Exception('Fitter not found for the expression {0}.'.format(fit_result._result.function.expr))

def fit_w_teach(data, teacher, errors=None):
    """
    Fit data with fitter.

    :param data: sweep data to fit
    :param teacher: fit result used as a teacher
    :param errors: error in data (data format depends on fitter)
    """
    import fit
    from fit.expr_with_args import EWA_to_func
    
    fitter = getattr(md.fitters, 'fitter_' + get_fitter_name(teacher))
    func, guess, names, others = fitter
    
    params = OrderedDict()
    for name in names:
        params[name] = teacher.fitparamdict[name]

    # Fix arga and absa
    tmpfunc = EWA_to_func(func)
    tmp_res = tmpfunc(data.f, *params.values())
    params['absa'] = data.amplitude.mean()/np.abs(tmp_res).mean()*params['absa']
    tmp_res = tmpfunc(data.f, *params.values())
    angle_diff = np.angle(data.iq.mean()) - np.angle(tmp_res.mean()) 
    params['arga'] = params['arga'] + angle_diff
    
    s = teacher.fitrange
    
    if 'additional_expr' in fitter[-1]:
        for k, v in fitter[-1]['additional_expr'].items():
            params[k] = fit.Parameter(name=k, expr=v)

    r = md.fit_from_params(data, s, errors, params, func, names)
    r.add_functions(others)
    return r

def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=myHelpFormatter, description=__doc__)
    parser.add_argument('KIDs', type=int, nargs='*',
                        help='KID index to fit. If not given, fit all KIDs')
    parser.add_argument('--nfwhm', type=int, default=5,
                        help='Fit range in [FWHM] (roughly guessed FWHM is used).')
    parser.add_argument('--fitter', type=str, default='gaolinbg',
                        help=('function to be used in fit. supported are: %s'
                              % md.fitters.all_fitters))
    parser.add_argument('--verbose', type=int, default=default_verbose_level,
                        help="""
                        verbose level. 0: won't plot result.
                        1: plot result.""")
    parser.add_argument('--ncpu', type=int, default=1,
                        help="")
    parser.add_argument('--mode', choices=['both', 'plot', 'plot', 'calc'], default='calc',
                        help="""select do calulation only,
                        do plotting data (calculated before), or both""")
    parser.add_argument('--force', action='store_true') # for analysis and plot
    parser.add_argument('--test', action='store_true') # for plot
    #parser.add_argument('--teacher', type=str, default=None,
    #                    help='Path to the cache that is used as a teacher.')
    parser.add_argument('--teacher', type=int, default=None,
                        help='runid used as a teacher.')

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

    if do_calc:
        Calc(kids, args.KIDs, args.nfwhm, args.fitter, args.verbose, args.force, teacher=args.teacher)
        #NCPU = args.ncpu
        #Plot(kids, args.force, args.test, NCPU
        #Calc_multi(kids, args.KIDs, args.nfwhm, args.fitter, args.verbose, args.force, teacher=args.teacher)
    
    if do_plot:
        NCPU = args.ncpu
        Plot(kids, args.force, args.test, NCPU)

def Calc(kids, kidinds, nfwhm=5, fitter='gaolinbg', verbose=0, force=False, teacher=None):
    failed = []
    kidid = []
    fr, dfr = [], []
    Qr, dQr = [], []
    Qc, dQc = [], []
    Qi, dQi = [], []
    if teacher is not None:
        #teacher_kids = bbs.Cache.load(teacher)
        teacher_kids = get_kids_from_db(teacher)
    for i, kid in kids.items():
        if kid.enabled:
            if force:
                if kid.has_cache('fit'):
                    del kid.__dict__['_path_fit']

            print( 'fitting KID[%d]' % i )
            try:
                if teacher is None:
                    r = kid.fit(nfwhm, fitter, Q_search=100)
                else:
                    t_kid = teacher_kids[i]
                    r = fit_w_teach(kid.get_cache('raw_sweep'), t_kid.get_cache('fit'))
                    kid.set_cache('fit', r)
                # kid.rewind_tod()
                fr_ = r.params['fr'].value # GHz
                dfr_ = r.params['fr'].stderr # GHz
                Qr_ = r.params['Qr'].value
                dQr_ = r.params['Qr'].stderr
                Qc_ = r.params['Qc'].value
                dQc_ = r.params['Qc'].stderr
                Qi_ = r.params['Qi'].value
                dQi_ = r.params['Qi'].stderr

                #if dfr_/fr_<0. or dQr_/Qr_<0. or dQc_/Qc_<0. or dQi_/Qi_<0.:
                if dfr_==None or fr_==None or dQr_==None or Qr_==None or dQc_==None or Qc_==None:# reject None
                    failed.append(i)
                elif dfr_/fr_<0. or dQr_/Qr_<0. or dQc_/Qc_<0.:
                    failed.append(i)
                elif fr_!=fr_ or dfr_!=dfr_ or Qr_!=Qr_ or dQr_!=dQr_: # reject Nan
                    failed.append(i)
                elif abs(Qr_)==float('inf') or abs(dQr_)==float('inf'): # reject +/-inf
                    failed.append(i)
                else:
                    kidid.append(i)
                    fr.append(fr_)
                    dfr.append(dfr_)
                    Qr.append(Qr_)
                    dQr.append(dQr_)
                    Qc.append(Qc_)
                    dQc.append(dQc_)
                    Qi.append(Qi_)
                    dQi.append(dQi_)

            except md.MKIDDataException as err:
                print( 'KID[%d] fit failed: %s' %(i, err.value) )
                failed.append(i)
                
            try:
                kids.save(clear_memory=True)
            except:
                print("kids.save(clear_memory=True):: Failed")
                pass
            for tmp0, tmp1 in kid.__dict__.copy().items():
                if tmp0[:6] == '_data_':
                    del kid.__dict__[tmp0]

    print( 'appended kid indices failing in fit, to %s:' % disabled_kid_file )
    print( failed )
    if len(failed)>0:
        with open(disabled_kid_file, 'a') as f:
            print( '# below are append by FitSweep', file=f )
            for i in failed:
                print( i, file=f )

#    ## reject KIDs with large error in Qr, Qc, Qi
#    from util import bad_index_by_fraction, delete_elements
#    idx_ar = bad_index_by_fraction(Qr, dQr, thre0=6., thre1=5.)
#    print 'by Qr error fraction:'
#    print np.array(kidid)[idx_ar]
#    if len(idx_ar)>0:
#        with open(disabled_kid_file, 'a') as f:
#            print >>f, '# below are append by FitSweep'
#            print >>f, '# by Qr error fraction'
#            for idx in idx_ar:
#                print >>f, kidid[idx]
#        kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi\
#            = delete_elements([kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi], idx_ar)
#
#    idx_ar = bad_index_by_fraction(Qc, dQc, thre0=6., thre1=5.)
#    print 'by Qc error fraction:'
#    print np.array(kidid)[idx_ar]
#    if len(idx_ar)>0:
#        with open(disabled_kid_file, 'a') as f:
#            print >>f, '# below are append by FitSweep'
#            print >>f, '# by Qc error fraction'
#            for idx in idx_ar:
#                print >>f, kidid[idx]
#        kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi\
#            = delete_elements([kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi], idx_ar)
#
#    idx_ar = bad_index_by_fraction(Qi, dQi, thre0=6., thre1=5.)
#    print 'by Qi error fraction:'
#    print np.array(kidid)[idx_ar]
#    if len(idx_ar)>0:
#        with open(disabled_kid_file, 'a') as f:
#            print >>f, '# below are append by FitSweep'
#            print >>f, '# by Qi error fraction'
#            for idx in idx_ar:
#                print >>f, kidid[idx]
#        kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi\
#            = delete_elements([kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi], idx_ar)
    ##
    np.save(outdir+'/FitSweep_fit.npy',
            np.array([kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi]))
            

def Plot(kids, force, test, NCPU):
    plotdir = outdir + '/figFitSweep'
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
    
    params_list = []
    for i, kid in kids.items():
        if not kid.enabled: continue
        params_list.append({"i": i, "kid": kid, "plotdir":plotdir})
    from multiprocessing import Pool
    print(md)
    p = Pool(NCPU)
    result = p.map(plot_multi, params_list)

    """
    for i, kid in kids.items():
        if not kid.enabled: continue
        if test and i>10: break
        print( 'plotting KID[%d]' % i )
        r   = kid.get_cache('fit')
        swp = kid.get_cache('raw_sweep')
        tod = kid.get_cache('raw_tod')
        fitted = r.fitted(swp.x)
        #fig = plt.figure(figsize=(16,4))
        fig = plt.figure(figsize=(16,10))
        #plt.title('KID[%d]' % i)
        ax1 = plt.subplot(231)
        ax1.set_title('Amplitude vs Freq')
        ax2 = plt.subplot(232)
        ax2.set_title('KID[%d], I vs Q' %i)
        plt.axis('equal')
        r.plot(ax1=ax1, ax2=ax2, data=swp)
        ax3 = plt.subplot(233)
        ax3.set_title('rewind')
        plt.axis('equal')
        rw_s = r.rewind(swp.x, swp.iq)
        #rw_t = r.rewind(tod.frequency, tod.iq)
        rw_t = r.rewind(tod.frequency, tod.iq[::100])
        rw_f = r.rewind(swp.x, r.fitted(swp.x))
        md.plot_iq(rw_s, label='Sweep')
        md.plot_iq(rw_t, '.', alpha=0.2, label='TOD')
        md.plot_iq(rw_f, 'y', lw=3, alpha=0.5, label='fit')
        plt.grid()
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.axhline(color='k')
        plt.axvline(color='k')
        plt.legend(loc='best')

        plt.subplot(212)
        fc = tod.frequency # GHz
        fr = r.params['fr'].value # GHz
        phase_s = -np.angle(-rw_s)
        phase_t = -np.angle(-rw_t)
        phase_f = -np.angle(-rw_f)
        plt.plot(swp.x, phase_s, label='Sweep')
        plt.plot(swp.x, phase_f, 'y', lw=3, alpha=0.5, label='fit')
        ## spline interpolation
        from scipy import interpolate
        try:
            tck = interpolate.splrep(phase_f, swp.x, s=0)
            x = interpolate.splev(phase_t, tck, der=0)
            plt.plot(x, phase_t, '.', alpha=0.2, label='TOD')
        except ValueError:
            print( 'KID[%d] TOD interpolation failed..' %i )
        #
        tck = interpolate.splrep(swp.x, phase_s, s=0)
        y = interpolate.splev(fc, tck, der=0)
        plt.plot(fc, y, 'go', label='carrier f')
        y = interpolate.splev(fr, tck, der=0)
        plt.plot(fr, y, 'r*', label='resonance f')
        ##
        plt.grid()
        plt.title('KID[%d], Phase vs Freq' %i)
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Phase [rad]')
        plt.xlim(min(swp.x), max(swp.x))
        plt.legend(loc='best')

        plt.savefig(os.path.join(plotdir, 'FitSweep_%04d.png' % i))
        #plt.close(fig)
        plt.clf()
        plt.close()

        for tmp0, tmp1 in kid.__dict__.copy().items():
            if tmp0[:6] == '_data_':
                del kid.__dict__[tmp0]
    """
    
def plot_multi(params):
    i = params["i"]
    kid = params["kid"]
    plotdir = params["plotdir"]
    #if not kid.enabled: continue
    #if test and i>10: break
    print( 'plotting KID[%d]' % i )
    r   = kid.get_cache('fit')
    swp = kid.get_cache('raw_sweep')
    tod = kid.get_cache('raw_tod')
    fitted = r.fitted(swp.x)
    #fig = plt.figure(figsize=(16,4))
    fig = plt.figure(figsize=(16,10))
    #plt.title('KID[%d]' % i)
    ax1 = plt.subplot(231)
    ax1.set_title('Amplitude vs Freq')
    ax2 = plt.subplot(232)
    ax2.set_title('KID[%d], I vs Q' %i)
    plt.axis('equal')
    r.plot(ax1=ax1, ax2=ax2, data=swp)
    ax3 = plt.subplot(233)
    ax3.set_title('rewind')
    plt.axis('equal')
    rw_s = r.rewind(swp.x, swp.iq)
    #rw_t = r.rewind(tod.frequency, tod.iq)
    rw_t = r.rewind(tod.frequency, tod.iq[::100])
    rw_f = r.rewind(swp.x, r.fitted(swp.x))
    md.plot_iq(rw_s, label='Sweep')
    md.plot_iq(rw_t, '.', alpha=0.2, label='TOD')
    md.plot_iq(rw_f, 'y', lw=3, alpha=0.5, label='fit')
    plt.grid()
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.axhline(color='k')
    plt.axvline(color='k')
    plt.legend(loc='best')

    plt.subplot(212)
    fc = tod.frequency # GHz
    fr = r.params['fr'].value # GHz
    phase_s = -np.angle(-rw_s)
    phase_t = -np.angle(-rw_t)
    phase_f = -np.angle(-rw_f)
    plt.plot(swp.x, phase_s, label='Sweep')
    plt.plot(swp.x, phase_f, 'y', lw=3, alpha=0.5, label='fit')
    ## spline interpolation
    from scipy import interpolate
    try:
        tck = interpolate.splrep(phase_f, swp.x, s=0)
        x = interpolate.splev(phase_t, tck, der=0)
        plt.plot(x, phase_t, '.', alpha=0.2, label='TOD')
    except ValueError:
        print( 'KID[%d] TOD interpolation failed..' %i )
        #
    tck = interpolate.splrep(swp.x, phase_s, s=0)
    y = interpolate.splev(fc, tck, der=0)
    plt.plot(fc, y, 'go', label='carrier f')
    y = interpolate.splev(fr, tck, der=0)
    plt.plot(fr, y, 'r*', label='resonance f')
    ##
    plt.grid()
    plt.title('KID[%d], Phase vs Freq' %i)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Phase [rad]')
    plt.xlim(min(swp.x), max(swp.x))
    plt.legend(loc='best')

    plt.savefig(os.path.join(plotdir, 'FitSweep_%04d.png' % i))
    #plt.close(fig)
    plt.clf()
    plt.close()

    for tmp0, tmp1 in kid.__dict__.copy().items():
        if tmp0[:6] == '_data_':
            del kid.__dict__[tmp0]
    pass

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)

    from common import *
    main()

