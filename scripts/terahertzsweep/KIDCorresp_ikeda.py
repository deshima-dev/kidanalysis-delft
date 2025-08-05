#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import shutil
import os
from pathlib import Path

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
    parser.add_argument('--del_1st_or_2nd', type=str, default="2nd",
                        help='If there are overlapping KIDs, which one should be deleted')
#    parser.add_argument('--DB', action='store_true', help="""write to DB""", default = False)
    args = parser.parse_args(argv)

    kid_corresp(kids, args)


#############################################################
def kid_corresp(kids, args):
    del_1st_or_2nd = args.del_1st_or_2nd
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
    
    rejection = args.threshold # GHz (to get non-filter KIDs out)
    good_indices_ori = np.where(f_KID_measured_raw < rejection)[0]
    ## Remove lower left corner
    diff_1_0 = F_filter_measured_raw[good_indices_ori[1]] - F_filter_measured_raw[good_indices_ori[0]]
    if diff_1_0>100:
        good_indices = np.delete(good_indices_ori, 0)
    elif diff_1_0<-100:
        good_indices = np.delete(good_indices_ori, 1)
    else:
        good_indices = good_indices_ori[:]

    #print("good_indices", good_indices)
    #print("good_indices.shape", good_indices.shape)

    measured_ids = measured_ids_raw[good_indices]
    F_filter_measured = F_filter_measured_raw[good_indices]
    f_KID_measured = f_KID_measured_raw[good_indices]

    ## Remove too small distance (If more than 3 KIDs are overlapped it may not work well.)
    """
    diff_y = np.abs(np.diff(F_filter_measured))
    #print("F_filter_measured.shape", F_filter_measured.shape)
    flag_list = []
    for i in range(len(diff_y)):
        if diff_y[i]<0.5: #GHz
            if del_1st_or_2nd=="1st":
                good_indices = np.delete(good_indices_ori, i)
                flag_list.append(good_indices_ori[i])
            elif del_1st_or_2nd=="2nd":
                good_indices = np.delete(good_indices_ori, i+1)
                flag_list.append(good_indices_ori[i+1])
            else:
                print('"del_1st_or_2nd" should be 1st or 2nd')
                sys.exit()
    """
    from scipy.spatial.distance import pdist, squareform
    x_norm = (f_KID_measured - np.nanmean(f_KID_measured)) / np.nanstd(f_KID_measured)
    y_norm = (F_filter_measured - np.nanmean(F_filter_measured)) / np.nanstd(F_filter_measured)
    coords = np.vstack([x_norm, y_norm]).T
    dists = squareform(pdist(coords))
    i_upper, j_upper = np.triu_indices(len(coords), k=1)
    dist_values = dists[i_upper, j_upper]
    index_pairs = list(zip(i_upper, j_upper))
    sorted_pairs = sorted(zip(dist_values, index_pairs), key=lambda t: t[0])
    threshold = 0.03 ########
    close_kid_id_pairs = [
        (i, j) 
        for d, (i, j) in sorted_pairs 
        if d <= threshold]
    flag_list_ind, flag_list_kidid = [], []
    for p in close_kid_id_pairs:
        i, j = p
        print(f"close_kid_id_pairs = ({measured_ids[i]}, {measured_ids[j]})")
        if del_1st_or_2nd=="1st":
            flag_list_ind.append(i)
            flag_list_kidid.append(measured_ids[i])
        elif del_1st_or_2nd=="2nd":
            flag_list_ind.append(j)
            flag_list_kidid.append(measured_ids[j])
        else:
            print('"del_1st_or_2nd" should be 1st or 2nd')
            sys.exit()
    good_indices = np.delete(good_indices, flag_list_ind)
    f = open(os.path.join(outdir, "flag_overlap_delete_%s_F_measured.dat"%del_1st_or_2nd), "w")
    f.write("# flag_overlap_%s_F_measured\n"%del_1st_or_2nd)
    for i in flag_list_kidid:
        f.write("%i\n"%i)
    f.close()
    measured_ids = measured_ids_raw[good_indices]
    F_filter_measured = F_filter_measured_raw[good_indices]
    f_KID_measured = f_KID_measured_raw[good_indices]
    #print("measured_ids", measured_ids)

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
    fig.savefig(os.path.join(plotdir, 'raw_delete_%s.pdf'%del_1st_or_2nd))
    fig.savefig(os.path.join(plotdir, 'raw_delete_%s.png'%del_1st_or_2nd), dpi = 300)
    plt.close()
    plt.clf()
    plt.cla()

#    fig.savefig(os.path.join(plotdir, 'raw.design.pdf'))
#    fig.savefig(os.path.join(plotdir, 'raw.design.png'), dpi = 300)

    ########################## adjust
    #### Ikeda's adjustment
    np.save(os.path.join(plotdir, "f_KID_design_delete_%s.npy"%del_1st_or_2nd), f_KID_design)
    np.save(os.path.join(plotdir, "F_filter_design_delete_%s.npy"%del_1st_or_2nd), F_filter_design)
    np.save(os.path.join(plotdir, "f_KID_measured_delete_%s.npy"%del_1st_or_2nd), f_KID_measured)
    np.save(os.path.join(plotdir, "F_filter_measured_delete_%s.npy"%del_1st_or_2nd), F_filter_measured)

    Ncol = 7 ##########
    
    designed_1, designed_2, measured_1, measured_2, num1_d, num2_d, num1_m, num2_m = make_points(f_KID_design, F_filter_design, f_KID_measured, F_filter_measured)

    d1_ind, m1_ind, measured_1_trans, pnts1_d, pnts1_m, box1_d, box1_m = align_points(designed_1, measured_1, Ncol)
    d2_ind, m2_ind, measured_2_trans, pnts2_d, pnts2_m, box2_d, box2_m = align_points(designed_2, measured_2, Ncol)
	
    Path_outdir = Path(outdir)
    parent = (Path_outdir.parent).parent.name

    fig = plt.figure(figsize=(16,8))
    plot_pnts(fig, 1, designed_1, measured_1)#, ind_d = d1_ind, ind_m = m1_ind)
    plot_pnts(fig, 2, designed_2, measured_2)
    fig.savefig(os.path.join(plotdir, f"original_{parent}_delete_{del_1st_or_2nd}.png"))

    fig = plt.figure(figsize=(16,8))
    plot_pnts(fig, 1, designed_1, measured_1_trans, ind_d = d1_ind, ind_m = m1_ind)
    plot_pnts(fig, 2, designed_2, measured_2_trans, ind_d = d2_ind, ind_m = m2_ind)
    fig.savefig(os.path.join(plotdir, f"aligned_{parent}_delete_{del_1st_or_2nd}.png"))

    fig = plt.figure(figsize=(16,8))
    plot_pnts(fig, 1, pnts1_d, pnts1_m, box_d = box1_d, box_m = box1_m)#, ind_d = d1_ind, ind_m = m1_ind)
    plot_pnts(fig, 2, pnts2_d, pnts2_m, box_d = box2_d, box_m = box2_m)#, ind_d = d1_ind, ind_m = m1_ind)
    fig.savefig(os.path.join(plotdir, f"modified_{parent}_delete_{del_1st_or_2nd}.png"))

    #fig = plt.figure(figsize=(16,8))
    #plot_pnts(fig, 1, pnts1_d_adj, pnts1, equal = 'True', ind_d = d1_ind, ind_m = m1_ind)
    #plot_pnts(fig, 2, pnts2_d_adj, pnts2, equal = 'True', ind_d = d2_ind, ind_m = m2_ind)
    ##plot_pnts(fig, 2, designed_2, measured_2_trans)
    #fig.savefig(os.path.join(plotdir, f"KIDs_transform_{parent}.png"))

    nearest_indices = np.empty(len(measured_ids), dtype=int)
    nearest_indices[good_indices_ori[m1_ind]] = d1_ind
    nearest_indices[good_indices_ori[num1_m + m2_ind]] = num1_d + d2_ind

    pnts_m_adj = np.vstack((measured_1_trans, measured_2_trans))
    pnts_d = np.vstack((designed_1, designed_2))
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
    fig.savefig(os.path.join(plotdir, f'correspondence_{parent}_delete_{del_1st_or_2nd}_ikeda.pdf'))
    fig.savefig(os.path.join(plotdir, f'correspondence_{parent}_delete_{del_1st_or_2nd}_ikeda.png'), dpi = 300)




    ########################### Write to json
    kc_path = os.path.join(outdir, f'kid_corresp_delete_{del_1st_or_2nd}_ikeda.json')
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
    #print("reference.dat = ", ref_list)
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

def make_points(ndarray_d1, ndarray_d2, ndarray_m1, ndarray_m2):
    #d_x = np.load(fname_d1)
    #d_y = np.load(fname_d2)

    #m_x = np.load(fname_m1)
    #m_y = np.load(fname_m2)
    
    d_x = ndarray_d1
    d_y = ndarray_d2

    m_x = ndarray_m1
    m_y = ndarray_m2

    num1_d = (np.argsort(-(d_x[1:] - d_x[0:-1]))[0] + 1)
    num2_d = d_x.size - num1_d

    pnts1_d = np.hstack( ( np.reshape(d_x[:num1_d],[num1_d,1]), np.reshape(d_y[:num1_d],[num1_d,1]) ) )
    pnts2_d = np.hstack( ( np.reshape(d_x[num1_d:],[num2_d,1]), np.reshape(d_y[num1_d:],[num2_d,1]) ) )

    num1_m = (np.argsort(-(m_x[1:] - m_x[0:-1]))[0] + 1)
    num2_m = m_x.size - num1_m

    pnts1_m = np.hstack( ( np.reshape(m_x[:num1_m],[num1_m,1]), np.reshape(m_y[:num1_m],[num1_m,1]) ) )
    pnts2_m = np.hstack( ( np.reshape(m_x[num1_m:],[num2_m,1]), np.reshape(m_y[num1_m:],[num2_m,1]) ) )

    return pnts1_d, pnts2_d, pnts1_m, pnts2_m, num1_d, num2_d, num1_m, num2_m

def plot_pnts(fig, subplot_num, pnts_d, pnts_m, equal = 'False', 
              ind_d = np.empty(0), ind_m = np.empty(0), box_d = np.empty(0), box_m = np.empty(0)):
    ax = fig.add_subplot(1, 2, subplot_num)
    ax.scatter(pnts_d[:,0], pnts_d[:,1], c ='b', s = 5)
    ax.scatter(pnts_m[:,0], pnts_m[:,1], c ='r', s = 5)

    if equal == 'True':
        ax.axis('equal')

    if (ind_d.size > 0) & (ind_m.size > 0):
        ax.plot([pnts_m[ind_m,0], pnts_d[ind_d,0]], [pnts_m[ind_m,1], pnts_d[ind_d,1]], color = 'y')

    if (box_d.size >0):
        tmp = np.vstack((box_d, box_d[0]))
        ax.plot(tmp[:,0], tmp[:,1], c = 'b', linewidth = 2)

    if (box_m.size >0):
        tmp = np.vstack((box_m, box_m[0]))
        ax.plot(tmp[:,0], tmp[:,1], c = 'r', linewidth = 2)


def intersection_with_horizontal(pnt1, pnt2, y_min):
    x1, y1 = pnt1
    x2, y2 = pnt2

    if y1 == y2:
        raise ValueError("pnt1 and pnt2 are on a horizontal line; no intersection or infinite intersections.")

    t = (y_min - y1) / (y2 - y1) 
    x = x1 + t * (x2 - x1)

    return [x, y_min]

def parallelogram_with_x_axis(points):

    tmp  = ConvexHull(points)
    hull_pnts = tmp.points[tmp.vertices]

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

    y_min = np.min(points[:,1])
    y_max = np.max(points[:,1])

    c_pnts = np.zeros((4,2))

    c_pnts[1] = intersection_with_horizontal(corner_pnts[1], corner_pnts[2], y_min)
    c_pnts[2] = intersection_with_horizontal(corner_pnts[1], corner_pnts[2], y_max)
    
    c_pnts[0] = intersection_with_horizontal(corner_pnts[3], corner_pnts[0], y_min)
    c_pnts[3] = intersection_with_horizontal(corner_pnts[3], corner_pnts[0], y_max)

    return c_pnts
    
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

    c0_tmp_f = np.matmul(np.reshape(pnts_f[:,0],[M,1]), np.ones((1,N),dtype = np.float64))
    c0_tmp_t = np.matmul(np.ones((M,1),dtype = np.float64), np.reshape(pnts_t[:,0],[1,N]))
    c0_tmp   = np.square((c0_tmp_t - c0_tmp_f))

    c1_tmp_f = np.matmul(np.reshape(pnts_f[:,1],[M,1]), np.ones((1,N),dtype = np.float64))
    c1_tmp_t = np.matmul(np.ones((M,1),dtype = np.float64), np.reshape(pnts_t[:,1],[1,N]))
    c1_tmp   = np.square((c1_tmp_t - c1_tmp_f))

    C = np.sqrt(c0_tmp + c1_tmp)

    return C
    
def GMM_likelihood(pnts_f, mu_s, sigma):

    M = pnts_f.shape[0]
    N = mu_s.shape[0]

    x_tmp_f = np.matmul(np.reshape(pnts_f[:,0],[M,1]), np.ones((1,N),dtype = np.float64))/(sigma)
    x_tmp_t = np.matmul(np.ones((M,1),dtype = np.float64), np.reshape(mu_s[:,0],[1,N]))/(sigma)
    x_tmp   = np.square((x_tmp_t - x_tmp_f))

    y_tmp_f = np.matmul(np.reshape(pnts_f[:,1],[M,1]), np.ones((1,N),dtype = np.float64))/(sigma)
    y_tmp_t = np.matmul(np.ones((M,1),dtype = np.float64), np.reshape(mu_s[:,1],[1,N]))/(sigma)
    y_tmp   = np.square((y_tmp_t - y_tmp_f))

    gauss = np.exp(- (x_tmp + y_tmp)/2)

    prob = np.sum(gauss,1)/(2*np.pi*sigma*sigma)
    log_likelihood = np.sum(np.log(prob))

    return log_likelihood
    
def align_points(designed, measured, Ncol, x_range = 2, y_range = 2, step = 1/50, sigma = 0.2):

    # convert y-axis to log scale
    
    pnts_d = np.empty_like(designed)
    pnts_m = np.empty_like(measured)

    pnts_d[:,0] = designed[:,0]
    pnts_d[:,1] = np.log(designed[:,1])

    pnts_m[:,0] = measured[:,0]
    pnts_m[:,1] = np.log(measured[:,1])

    # set y_step of the designed and measured KID data

    tmp = np.sort(np.abs(pnts_d[1:,1]-pnts_d[0:-1,1]))
    y_step_d = np.mean(tmp[:-10])

    tmp = np.sort(np.abs(pnts_m[1:,1]-pnts_m[0:-1,1]))
    y_step_m = np.mean(tmp[(tmp >= 0.7 * y_step_d) & (tmp <= 1.2 * y_step_d)])

    # set boxes for the designed and measured KID data

    box_d = parallelogram_with_x_axis(pnts_d)
    box_m = parallelogram_with_x_axis(pnts_m)

    maxy_d = (box_d[2,1] - box_d[1,1])/y_step_d
    maxy_m = min((box_m[2,1] - box_m[1,1])/y_step_m, (box_d[2,1] - box_d[1,1])/y_step_d)

    template_d = np.empty((4,2))
    template_m = np.empty((4,2))

    template_d[0] = [0,0]
    template_d[1] = [np.float64(Ncol)-1.0, 0]
    template_d[2] = [np.float64(Ncol)-1.0, maxy_d]
    template_d[3] = [0, maxy_d]

    template_m[0] = [0,0]
    template_m[1] = [np.float64(Ncol)-1.0, 0]
    template_m[2] = [np.float64(Ncol)-1.0, maxy_m]
    template_m[3] = [0, maxy_m]

#   print(y_step_d, y_step_m)
#    print(template_d)
#    print(template_m)

    # affine transform data

    A_d, b_d = affine_trans(box_d, template_d)
    A_m, b_m = affine_trans(box_m, template_m)

    pnts_d_adj = np.matmul(pnts_d, A_d) + b_d
    pnts_m_adj = np.matmul(pnts_m, A_m) + b_m

    # check bias

    x_kizami = np.arange(-x_range, x_range, step)
    y_kizami = np.arange(-y_range, y_range, step)

    Nx = x_kizami.size
    Ny = y_kizami.size

    likelihood_mat = np.empty((Nx, Ny))

    pnts= pnts_m_adj

    print('Making test matrix for adjustment')    
    for i in range(Nx):
        for j in range(Ny):
            pnts = np.empty(pnts_m_adj.shape)
            pnts[:,0] = pnts_m_adj[:,0] + x_kizami[i]
            pnts[:,1] = pnts_m_adj[:,1] + y_kizami[j]
            likelihood_mat[i,j] = GMM_likelihood(pnts, pnts_d_adj, sigma)

    max_pos = np.unravel_index(np.argmax(likelihood_mat), likelihood_mat.shape)
    #    print(max_pos)

    x_bias = x_kizami[max_pos[0]]
    y_bias = y_kizami[max_pos[1]]

    #    print(x_bias, y_bias)    

    #    plot_points(pnts_d_adj, pnts_d_adj, pnts_m_adj, pnts_m_adj)

    # Finding KIDs correspondence

    pnts = np.empty(pnts_m_adj.shape)
    pnts[:,0] = pnts_m_adj[:,0] + x_bias
    pnts[:,1] = pnts_m_adj[:,1] + y_bias

    C = compute_C(pnts, pnts_d_adj)

    m_ind, d_ind = linear_sum_assignment(C)

    # return transposed data
    A_inv = np.linalg.inv(A_d)

    tmp = np.matmul((pnts-b_d), A_inv)

    pnts_m_trans = np.empty_like(measured)

    pnts_m_trans[:,0] = tmp[:,0]
    pnts_m_trans[:,1] = np.exp(tmp[:,1])

    return d_ind, m_ind, pnts_m_trans, pnts_d, pnts_m, box_d, box_m    
#    return d_ind, m_ind, pnts_m_trans, pnts_d, pnts_m, box_d, box_m    







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

