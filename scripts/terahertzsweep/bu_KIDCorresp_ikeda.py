#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import shutil
import os

import sqlite3
import datetime
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
#dbname = '/Users/sfujita/Desktop/DESHIMA/toptica/kid_test.db'
dbname = '../kid_test.db'
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
    plotdir = os.path.join(outdir, 'KIDCorresp_ikeda')
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
    plt.close()
    plt.clf()
    plt.cla()

#    fig.savefig(os.path.join(plotdir, 'raw.design.pdf'))
#    fig.savefig(os.path.join(plotdir, 'raw.design.png'), dpi = 300)

    ########################## adjust
    #### Ikeda's adjustment
    np.save(os.path.join(plotdir, "f_KID_design.npy"), f_KID_design)
    np.save(os.path.join(plotdir, "F_filter_design.npy"), F_filter_design)
    np.save(os.path.join(plotdir, "f_KID_measured.npy"), f_KID_measured)
    np.save(os.path.join(plotdir, "F_filter_measured.npy"), F_filter_measured)


    tmpx = f_KID_design
    tmpy = F_filter_design

    num1_d = (np.argsort(-(tmpx[1:] - tmpx[0:-1]))[0] + 1)
    num2_d = tmpx.size - num1_d

    pnts1_d = np.hstack( ( np.reshape(tmpx[:num1_d],[num1_d,1]), np.reshape(tmpy[:num1_d],[num1_d,1]) ) )
    pnts2_d = np.hstack( ( np.reshape(tmpx[num1_d:],[num2_d,1]), np.reshape(tmpy[num1_d:],[num2_d,1]) ) )


    tmpx = f_KID_measured
    tmpy = F_filter_measured

    num1_m = (np.argsort(-(tmpx[1:] - tmpx[0:-1]))[0] + 1)
    num2_m = tmpx.size - num1_m

    pnts1_m = np.hstack( ( np.reshape(tmpx[:num1_m],[num1_m,1]), np.reshape(tmpy[:num1_m],[num1_m,1]) ) )
    pnts2_m = np.hstack( ( np.reshape(tmpx[num1_m:],[num2_m,1]), np.reshape(tmpy[num1_m:],[num2_m,1]) ) )
    plt.plot(range(tmpx.size),tmpx)
    plt.savefig(os.path.join(plotdir, "separate.png"))
    plt.clf()

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pnts1_d[:,0], pnts1_d[:,1], c ='r', s = 1)
    ax1.scatter(pnts1_m[:,0], pnts1_m[:,1], c ='b', s = 1)   
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pnts2_d[:,0], pnts2_d[:,1], c ='r', s = 1)
    ax2.scatter(pnts2_m[:,0], pnts2_m[:,1], c ='b', s = 1)
    fig.savefig(os.path.join(plotdir, "1_2_ori.png"))
    plt.clf()

    tmp  = ConvexHull(pnts1_d)
    hull1_d = tmp.points[tmp.vertices]
    tmp = ConvexHull(pnts2_d)
    hull2_d = tmp.points[tmp.vertices]
    tmp = ConvexHull(pnts1_m)
    hull1_m = tmp.points[tmp.vertices]
    tmp = ConvexHull(pnts2_m)
    hull2_m = tmp.points[tmp.vertices]

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pnts1_d[:,0], pnts1_d[:,1], c ='r', s = 1)
    tmp = np.vstack((hull1_d, hull1_d[0]))
    ax1.plot(tmp[:,0], tmp[:,1], c = 'b')
    ax1.scatter(pnts1_m[:,0], pnts1_m[:,1], c ='b', s = 1)
    tmp = np.vstack((hull1_m, hull1_m[0]))
    ax1.plot(tmp[:,0], tmp[:,1], c = 'r')
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pnts2_d[:,0], pnts2_d[:,1], c ='r', s = 1)
    tmp = np.vstack((hull2_d, hull2_d[0]))
    ax2.plot(tmp[:,0], tmp[:,1], c = 'b')
    ax2.scatter(pnts2_m[:,0], pnts2_m[:,1], c ='b', s = 1)
    tmp = np.vstack((hull2_m, hull2_m[0]))
    ax2.plot(tmp[:,0], tmp[:,1], c = 'r')
    fig.savefig(os.path.join(plotdir, "1_2_region_ori.png"))
    plt.clf()

    c_pnts1_m = bb_pnts(hull1_m)
    c_pnts2_m = bb_pnts(hull2_m)
    c_pnts1_d = bb_pnts(hull1_d)
    c_pnts2_d = bb_pnts(hull2_d)

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pnts1_d[:,0], pnts1_d[:,1], c ='r', s = 1)
    tmp = np.vstack((c_pnts1_d, c_pnts1_d[0]))
    ax1.plot(tmp[:,0], tmp[:,1], c = 'r')
    ax1.scatter(pnts1_m[:,0], pnts1_m[:,1], c ='b', s = 1)
    tmp = np.vstack((c_pnts1_m, c_pnts1_m[0]))
    ax1.plot(tmp[:,0], tmp[:,1], c = 'b')
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pnts2_d[:,0], pnts2_d[:,1], c ='r', s = 1)
    tmp = np.vstack((c_pnts2_d, c_pnts2_d[0]))
    ax2.plot(tmp[:,0], tmp[:,1], c = 'r')
    ax2.scatter(pnts2_m[:,0], pnts2_m[:,1], c ='b', s = 1)
    tmp = np.vstack((c_pnts2_m, c_pnts2_m[0]))
    ax2.plot(tmp[:,0], tmp[:,1], c = 'b')
    fig.savefig(os.path.join(plotdir, "1_2_region_parallelogram.png"))
    plt.clf()

    A1, b1 = affine_trans(c_pnts1_m, c_pnts1_d)
    A2, b2 = affine_trans(c_pnts2_m, c_pnts2_d)
    pnts1_m_adj      = np.matmul(pnts1_m, A1) + b1
    c_pnts1_m_adj = np.matmul(c_pnts1_m, A1) + b1
    pnts2_m_adj      = np.matmul(pnts2_m, A2) + b2
    c_pnts2_m_adj = np.matmul(c_pnts2_m, A2) + b2

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pnts1_d[:,0], pnts1_d[:,1], c ='r', s = 1)
    tmp = np.vstack((c_pnts1_d, c_pnts1_d[0]))
    ax1.plot(tmp[:,0], tmp[:,1], c = 'r')
    ax1.scatter(pnts1_m_adj[:,0], pnts1_m_adj[:,1], c ='b', s = 1)
    tmp = np.vstack((c_pnts1_m_adj, c_pnts1_m_adj[0]))
    ax1.plot(tmp[:,0], tmp[:,1], c = 'b')
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pnts2_d[:,0], pnts2_d[:,1], c ='r', s = 1)
    tmp = np.vstack((c_pnts2_d, c_pnts2_d[0]))
    ax2.plot(tmp[:,0], tmp[:,1], c = 'r')
    ax2.scatter(pnts2_m_adj[:,0], pnts2_m_adj[:,1], c ='b', s = 1)
    tmp = np.vstack((c_pnts2_m_adj, c_pnts2_m_adj[0]))
    ax2.plot(tmp[:,0], tmp[:,1], c = 'b')
    fig.savefig(os.path.join(plotdir, "1_2_overlap_with_parallelogram.png"))
    plt.clf()

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pnts1_d[:,0], pnts1_d[:,1], c ='r', s = 1)
    ax1.scatter(pnts1_m_adj[:,0], pnts1_m_adj[:,1], c ='b', s = 1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pnts2_d[:,0], pnts2_d[:,1], c ='r', s = 1)
    ax2.scatter(pnts2_m_adj[:,0], pnts2_m_adj[:,1], c ='b', s = 1)
    fig.savefig(os.path.join(plotdir, "1_2_overlap.png"))
    plt.clf()

    C1 = compute_C(pnts1_m_adj, pnts1_d)
    C2 = compute_C(pnts2_m_adj, pnts2_d)

    m1_ind, d1_ind = linear_sum_assignment(C1)
    m2_ind, d2_ind = linear_sum_assignment(C2)

    tmp1x = pnts1_m_adj[m1_ind,0]
    tmp1y = pnts1_m_adj[m1_ind,1]
    tmp1u = pnts1_d[d1_ind,0] - pnts1_m_adj[m1_ind,0]
    tmp1v = pnts1_d[d1_ind,1] - pnts1_m_adj[m1_ind,1]

    tmp2x = pnts2_m_adj[m2_ind,0]
    tmp2y = pnts2_m_adj[m2_ind,1]
    tmp2u = pnts2_d[d2_ind,0] - pnts2_m_adj[m2_ind,0]
    tmp2v = pnts2_d[d2_ind,1] - pnts2_m_adj[m2_ind,1]

    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pnts1_d[:,0], pnts1_d[:,1], c ='r', s = 1)
    ax1.scatter(pnts1_m_adj[:,0], pnts1_m_adj[:,1], c ='b', s = 1)
    ax1.plot([pnts1_m_adj[m1_ind,0], pnts1_d[d1_ind,0]], [pnts1_m_adj[m1_ind,1], pnts1_d[d1_ind,1]], color = 'y')
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pnts2_d[:,0], pnts2_d[:,1], c ='r', s = 1)
    ax2.scatter(pnts2_m_adj[:,0], pnts2_m_adj[:,1], c ='r', s = 1)
    ax2.plot([pnts2_m_adj[m2_ind,0], pnts2_d[d2_ind,0]], [pnts2_m_adj[m2_ind,1], pnts2_d[d2_ind,1]], color = 'y')
    fig.savefig(os.path.join(plotdir, "1_2_corresp.png"))

    nearest_indices = np.empty(len(measured_ids), dtype=int)
    nearest_indices[good_indices[m1_ind]] = d1_ind
    nearest_indices[good_indices[num1_m + m2_ind]] = num1_d + d2_ind

    pnts_m_adj = np.vstack((pnts1_m_adj, pnts2_m_adj))
    pnts_d = np.vstack((pnts1_d, pnts2_d))
    X_measured = pnts_m_adj[:, 0]
    Y_measured = pnts_m_adj[:, 1]
    X_design   = pnts_d[:, 0]
    Y_design   = pnts_d[:, 1]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X_measured, Y_measured, color = 'blue', label = 'Measured')
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
    fig.savefig(os.path.join(plotdir, 'correspondence_ikeda.pdf'))
    fig.savefig(os.path.join(plotdir, 'correspondence_ikeda.png'), dpi = 300)

    ########################### Write to json
    kc_path = os.path.join(outdir, 'kid_corresp_ikeda.json')
    import json
    ## add wideband KID
    ref_list = []
    with open(os.path.join(outdir, "reference.dat")) as f:
        for line in f:
            if line[0] == '#':
                continue
            elif len(line[:-1]) == 0:
                continue
            else:
                ref_list.append(int(line.split("\n")[0]))
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

def bb_pnts(hull_pnts):

    edge_angles = np.zeros(len(hull_pnts) - 1)

    for i in range(len(edge_angles)):
        edge_x = hull_pnts[i + 1, 0] - hull_pnts[i, 0]
        edge_y = hull_pnts[i + 1, 1] - hull_pnts[i, 1]
        edge_angles[i] = abs(np.arctan2(edge_y, edge_x) % (np.pi / 2))

    edge_angles = np.unique(edge_angles)

    min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0)

    for i in range(len(edge_angles)):
            # create Rotation matrix
        R = np.array([[np.cos(edge_angles[i]), np.cos(edge_angles[i] - (np.pi / 2))],
                              [np.cos(edge_angles[i] + (np.pi / 2)), np.cos(edge_angles[i])]])

        rot_pnts = np.dot(R, hull_pnts.T)

        # min max
        min_x = np.nanmin(rot_pnts[0], axis=0)
        max_x = np.nanmax(rot_pnts[0], axis=0)
        min_y = np.nanmin(rot_pnts[1], axis=0)
        max_y = np.nanmax(rot_pnts[1], axis=0)

        # width heigh
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        # store the smallest
        if (area < min_bbox[1]):
            min_bbox = (edge_angles[i], area, width, height, min_x, max_x, min_y, max_y)

    angle = min_bbox[0]

    R = np.array([[np.cos(angle), np.cos(angle - (np.pi / 2))],
                          [np.cos(angle + (np.pi / 2)), np.cos(angle)]])

    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    c_tmp = np.zeros((4, 2))

    c_tmp[0] = np.dot([max_x, min_y], R)
    c_tmp[1] = np.dot([min_x, min_y], R)
    c_tmp[2] = np.dot([min_x, max_y], R)
    c_tmp[3] = np.dot([max_x, max_y], R)

    corner_pnts = np.zeros((4, 2))

    corner_pnts[0] = c_tmp[np.argsort(np.sum(np.square(c_tmp), 1))[0],:]

    tmp_ind = np.argsort(np.sum(np.square(c_tmp - corner_pnts[0]), 1))

    corner_pnts[1] = c_tmp[tmp_ind[1],:]
    corner_pnts[2] = c_tmp[tmp_ind[3],:]
    corner_pnts[3] = c_tmp[tmp_ind[2],:]

    return corner_pnts    

def affine_trans(cpnts_f, cpnts_t):
    A = np.empty((2,2))
    Y = np.vstack(( cpnts_t[1]-cpnts_t[0], cpnts_t[2] - cpnts_t[1]))
    X = np.vstack(( cpnts_f[1]-cpnts_f[0], cpnts_f[2] - cpnts_f[1]))
    A = np.matmul(np.linalg.inv(X), Y)
    b = cpnts_t[0] - np.matmul(cpnts_f[0],A)
    return A, b

def compute_C(pnts_f, pnts_t):

    M = pnts_f.shape[0]
    N = pnts_t.shape[0]

    scale_0 = np.max(pnts_t[:,0]) - np.min(pnts_t[:,0])
    scale_1 = np.max(pnts_t[:,1]) - np.min(pnts_t[:,1])

    c0_tmp_f = np.matmul(np.reshape(pnts_f[:,0],[M,1]), np.ones((1,N),dtype = np.float64))/(scale_0)
    c0_tmp_t = np.matmul(np.ones((M,1),dtype = np.float64), np.reshape(pnts_t[:,0],[1,N]))/(scale_0)
    c0_tmp   = np.square((c0_tmp_t - c0_tmp_f))

    c1_tmp_f = np.matmul(np.reshape(pnts_f[:,1],[M,1]), np.ones((1,N),dtype = np.float64))/(scale_1)
    c1_tmp_t = np.matmul(np.ones((M,1),dtype = np.float64), np.reshape(pnts_t[:,1],[1,N]))/(scale_1)
    c1_tmp   = np.square((c1_tmp_t - c1_tmp_f))

    C = np.sqrt(c0_tmp + c1_tmp)

    return C






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

