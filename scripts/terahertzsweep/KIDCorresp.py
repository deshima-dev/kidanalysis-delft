#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import shutil
import os

import sqlite3
import datetime
import numpy as np
dbname = '/Users/sfujita/Desktop/DESHIMA/toptica/kid_test.db'
conn = sqlite3.connect(dbname, 
                       detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

sqlite3.dbapi2.converters['DATETIME'] = sqlite3.dbapi2.converters['TIMESTAMP']
c = conn.cursor()

def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=myHelpFormatter, description=__doc__)
    '''
    parser.add_argument('--mode', choices=['1', '2', 'all'], default='1',
                        help="""select analysis mode, or all""") # for analysis
    parser.add_argument('--force', action='store_true') # for plot
    '''
    parser.add_argument('--threshold',
                        #type=float, default = 6.25)
                        type=float, default = 7.0)
#    parser.add_argument('--DB', action='store_true', help="""write to DB""", default = False)
    args = parser.parse_args(argv)

    kid_corresp(kids, args)

#############################################################
def kid_corresp(kids, args):
    ## plot
    plotdir = os.path.join(outdir, 'KIDCorresp')
    #### make plot output directory
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    #### load npy file 
    npypath = os.path.join(outdir, 'AnaSpectrum2nd_fit.npy')
    if not os.path.exists(npypath):
        raise Exception('NPy file not found.')
    kidid, Fsubmm, Spectrum, _result_fit, Spectrum_df, _result_fit_df = np.load(npypath, allow_pickle=True)
    with open(outdir+'/result_fit.pkl', 'rb') as f:
        result_fit = pickle.load(f)
    with open(outdir+'/result_fit_df.pkl', 'rb') as f:
        result_fit_df = pickle.load(f)

    #### load design value
    print( outdir )
#    c.execute('''
#    SELECT main.detector_version 
#    FROM analysis
#        INNER JOIN main 
#        ON analysis.runid = main.runid
#    WHERE analysis.analysis_dir_path = ?;
#    ''', (outdir, ))
#    detector_version = c.fetchall()[0][0]
    detector_version = 'LT263_FlightChip'
    #detector_version = 'LT317_W1_Chip4'

    c.execute('''
    SELECT master_id, F_filter_design, f_KID_design 
    FROM "{0}" 
    WHERE attribute = "filter";
    '''.format(detector_version))
    design = np.array([[m_id, F_f_d, f_K_d] for m_id, F_f_d, f_K_d in c.fetchall()])

    ########################## information
    master_ids = design[:,0]
    F_filter_design = design[:,1]
    f_KID_design = design[:,2]

    measured_ids_raw = kidid
    F_filter_measured_raw = np.array([ rfd['peak_center'].value for rfd in result_fit_df ])
    f_KID_measured_raw = np.array([ kid.get_cache('fit').params['fr'].value \
                       for i, kid in kids.items() if i in kidid ])
    hwhm_measured = np.array([ rfd['hwhm'].value for rfd in result_fit_df ])
    print( len(measured_ids_raw) )
    print( len(F_filter_measured_raw) )
    print( len(f_KID_measured_raw) )
    
    rejection = args.threshold # GHz (to get non-filter KIDs out)
    good_indices = np.where(f_KID_measured_raw < rejection)[0]
    measured_ids = measured_ids_raw[good_indices]
    F_filter_measured = F_filter_measured_raw[good_indices]
    f_KID_measured = f_KID_measured_raw[good_indices]

    ########################## raw plot
    fig, ax = plt.subplots()
    #ax.scatter(f_KID_measured_raw, F_filter_measured_raw,
    ax.scatter(f_KID_measured, F_filter_measured, marker = '.',
               label = 'Measured: f_KID mean=%.3f, std=%.3f,\nF_filter mean=%.1f, std=%.1f'\
               %(f_KID_measured.mean(), f_KID_measured.std(), F_filter_measured.mean(), F_filter_measured.std()))
    ax.scatter(f_KID_design, F_filter_design, color = 'red', marker = '.',
               label = 'Design: f_KID mean=%.3f, std=%.3f,\nF_filter mean=%.1f, std=%.1f'\
               %(f_KID_design.mean(), f_KID_design.std(), F_filter_design.mean(), F_filter_design.std()))
    l,r = ax.get_xlim()
    #ax.axvspan(rejection, r, color = 'red', alpha = 0.3)
    ax.grid()
    ax.legend(loc=0, fontsize=12)
    ax.set_ylim(ymax=max(np.amax(F_filter_measured), np.amax(F_filter_design))+50.)
    ax.set_xlabel('f_KID [GHz]')
    ax.set_ylabel('F_filter [GHz]')
    fig.savefig(os.path.join(plotdir, 'raw.pdf'))
    fig.savefig(os.path.join(plotdir, 'raw.png'), dpi = 300)
#    fig.savefig(os.path.join(plotdir, 'raw.design.pdf'))
#    fig.savefig(os.path.join(plotdir, 'raw.design.png'), dpi = 300)

    ########################## adjust
    def adjust(a, b):
        ''' supposing a and b are numpy array'''
        return a, (a.std()/b.std())*(b - b.mean()) + a.mean()

    print( f_KID_design, f_KID_measured )
    fKda, fKma = adjust(f_KID_design, f_KID_measured)

    Ffda, Ffma = adjust(F_filter_design, F_filter_measured)

    def rectify(a, b):
        return a - a.mean(), (b - b.mean())*a.std()/b.std()

    X_design, Y_design = rectify(fKda, Ffda)
    C_design = X_design + 1j*Y_design
    X_measured, Y_measured = rectify(fKma, Ffma)
    C_measured = X_measured + 1j*Y_measured

    ############# adj plot
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(X_measured, Y_measured, label = 'Measured')
    ax.scatter(X_design, Y_design, color = 'red', label = 'Design')
    ax.grid()
    ax.legend(loc = 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(os.path.join(plotdir, 'rect.pdf'))
    fig.savefig(os.path.join(plotdir, 'rect.png'), dpi = 300)

    ################################## find correspondence
    def search_nearest(point, design_list):
        distance = np.abs(design_list - point)
        min_index = distance.argmin()
        return min_index

    nearest_indices = np.array([search_nearest(p, C_design) for p in C_measured])

    #### Collision handler
    from collections import Counter

    def get_collision():
        return [item for item, count in Counter(nearest_indices).items() if count > 1]

    def collision_handler():
        print( nearest_indices )
        dups = get_collision()
        print( '{0} collision detected.'.format(len(dups)) )

        nokori = np.array([ i for i in np.arange(len(C_design)) if not i in nearest_indices ])
        for dup in dups:
            print( dup )
            dup_kids = np.where(nearest_indices == dup)[0]
            print( measured_ids[dup_kids] )
            nokori_nearest = np.array(
                [search_nearest(p, C_design[nokori]) for p in C_measured[dup_kids]]
                )
            # most farthest will get original 
            farthest = np.abs(C_design[nokori[nokori_nearest]] - C_measured[dup_kids]).argmax()
            print( farthest )
            for i in range(len(dup_kids)):
                if i == farthest:
                    pass
                else:
                    nearest_indices[dup_kids[i]] = nokori[nokori_nearest[i]]
        pass

    loop_limit = 3
    for i in range(loop_limit):
        if len(get_collision()) == 0:
            break
        collision_handler()
    
    if len(get_collision()) != 0:
        raise Exception('Could not avoid collision!')
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X_measured, Y_measured, label = 'Measured')
    ax.scatter(X_design, Y_design, color = 'red', label = 'Design')
    for i, index in enumerate(nearest_indices):
        line = plt.Line2D((X_design[index], X_measured[i]), (Y_design[index], Y_measured[i]))        
        ax.add_line(line)
        ax.text(X_design[index], Y_design[index], '%d' %int(master_ids[index]),
                ha='left', va='bottom', color='red')
        ax.text(X_measured[i], Y_measured[i], '%d' %measured_ids[i],
                ha='right', va='top', color='blue')
    ax.legend(loc = 0)
    ax.grid()
    fig.savefig(os.path.join(plotdir, 'correspondence.pdf'))
    fig.savefig(os.path.join(plotdir, 'correspondence.png'), dpi = 300)

    ########################### Write to json
    kc_path = os.path.join(outdir, 'kid_corresp.json')
    import json
    ## add wideband KID
    ref_list = []
    line_count = 0
    with open(os.path.join(outdir, "reference.dat")) as f:
        for line in f:
            if line_count != 0:
                ref_list.append(int(line.split("\n")[0]))
            line_count += 1
    print("reference.dat = ", ref_list)
    tmpdict = {}
    if len(ref_list)>=4:
        last_four = ref_list[-4:]
        length = 1
        if last_four[-1]==last_four[-2]+1:
            length += 1
            if last_four[-2]==last_four[-3]+1:
                length += 1
                if last_four[-3]==last_four[-4]+1:
                    length += 1
        last_four = last_four[4-length:]
        for i in range(length):
            tmpdict[str(i)] = last_four[i]
    #tmpdict  = { int(master_ids[nearest_indices[i]]):kidid for i, kidid in enumerate(measured_ids)}
    for i, kidid in enumerate(measured_ids):
        tmpdict[int(master_ids[nearest_indices[i]])] = kidid 
    with open(kc_path, 'w') as f:
        json.dump(tmpdict, f, indent = 2)
    
    ########################### Register to DB
#    if args.DB:
#        LO, framelen = get_LO_framelen(outdir)
#        detector_version = get_detector_version(outdir)
#        print( 'write to DB, detector version=%s' %detector_version )
#        c.execute('''
#    INSERT INTO master_correspondence
#    (detector_version, LO, framelen, json_path)
#    VALUES
#    (?,?,?,?)
#    ''', (detector_version, LO, framelen, kc_path))
#        conn.commit()
#    print( 'Bye.' )

def get_kidslist_path(outdir):
    dd_path_sql = '''
    SELECT main.data_dir_path 
    FROM analysis
        INNER JOIN 
         main
        ON main.runid = analysis.runid
    WHERE analysis.analysis_dir_path = ?;    
    '''
    c.execute(dd_path_sql, (outdir,))
    kidslist_path = [os.path.join(ddpath, 'kids.list') for (ddpath ,) in c.fetchall()]
    return kidslist_path[0]

def get_detector_version(outdir):
    dd_path_sql = '''
    SELECT main.detector_version
    FROM analysis
        INNER JOIN 
         main
        ON main.runid = analysis.runid
    WHERE analysis.analysis_dir_path = ?;    
    '''
    c.execute(dd_path_sql, (outdir,))
    dvs = [ dv for (dv ,) in c.fetchall()]
    return dvs[0]

def get_LO_framelen(outdir):
    kidslist_path = get_kidslist_path(outdir)
    with open(kidslist_path) as f:
        LO_line = f.readline()
        framelen_line = f.readline()
    '''
    #LO:6000.0000
    #framelen:19
    '''
    import re
    LO_re = re.search(r'#LO:(.*)', LO_line)
    LO = float(LO_re.group(1))
    framelen_re = re.search(r'#framelen:(.*)', framelen_line)
    framelen = int(framelen_re.group(1))
    return LO, framelen

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)

    from common import *                   
    main()

