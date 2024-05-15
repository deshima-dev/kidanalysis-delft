#!/usr/bin/env python
# -*- coding:utf-8 -*-


def main(target_dirname, out_dirname):
    anapath_list = sorted(glob(os.path.join(target_dirname, "PreadScan_*", out_dirname)))
    print( anapath_list )
    ts_list = [anapath[anapath.find("PreadScan")+29:anapath.find("PreadScan")+35] for anapath in anapath_list]
    tmpcache_list = [bbs.Cache.load('%s/THzsweep/' %anapath) for anapath in anapath_list]
    P_list = [tmpcache[0].readpower for tmpcache  in tmpcache_list]   
    color_list = [get_color_from_cmap(1.0/(len(P_list)-1.0)*i) for i in range(len(P_list))]
    
    #fig, axarr = plt.subplots(len(tmpcache_list[0]), 1, figsize=(16, 96), facecolor='white')
    Qr_list_list, Qc_list_list, Qi_list_list, fr_list_list = [], [], [], []
    try:
        os.system("mkdir %s"%os.path.join(target_dirname, "fig_dips"))
    except:
        pass
    for kidid in range(len(tmpcache_list[0])):
        fig, axarr = plt.subplots(1, 1, figsize=(16, 8), facecolor='white')
        Qr_list, Qc_list, Qi_list, fr_list = [], [], [], []
        for i, tmpcache in enumerate(tmpcache_list):
            kid = tmpcache[kidid]
            swpdata = kid.raw_sweep()
            #axarr[kidid].plot(swpdata.f, np.log10(swpdata.amplitude), label=str(P_list[i]), color=color_list[i])
            #axarr[kidid].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            axarr.plot(swpdata.f, np.log10(swpdata.amplitude), label=str(P_list[i]), color=color_list[i])
            axarr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            try:
                fitr = kid.get_cache('fit')
                iq_rew = fitr.rewind(swpdata.f, swpdata.iq)
                #axarr[kidid].plot(swpdata.f, np.abs(fitr.fitted(swpdata.f)))      
                #axarr.plot(swpdata.f, np.abs(fitr.fitted(swpdata.f)))        
                #print( '+++++', kidid, fitr.params['Qr'].value, fitr.params['Qc'].value, fitr.params['Qi'].value )
                if fitr.params['Qi'].value<0 or fitr.params['Qc'].value<0 or fitr.params['Qr'].value<0:
                    print( kidid, 'Negative value!!' )
                Qr_list.append(fitr.params['Qr'].value)
                Qc_list.append(fitr.params['Qc'].value)
                Qi_list.append(fitr.params['Qi'].value)
                fr_list.append(fitr.params['fr'].value)
            except:
                print( 'Fit failed. KIDID: {0}'.format(kidid) )
                Qr_list.append(np.nan)
                Qc_list.append(np.nan)
                Qi_list.append(np.nan)
                fr_list.append(np.nan)
        #axarr[kidid].legend()
        #axarr[kidid].set_title("kidid: %s (log)"%str(kidid))
        #axarr[kidid].grid()
        axarr.legend()
        axarr.set_title("kidid: %s (log)"%str(kidid))
        axarr.grid()
        Qr_list_list.append(Qr_list)
        Qc_list_list.append(Qc_list)
        Qi_list_list.append(Qi_list)
        fr_list_list.append(fr_list)
        plt.savefig(os.path.join(target_dirname, "fig_dips", "dips_%s.png"%str(kidid).zfill(5)))
        plt.clf()
    #plt.savefig(os.path.join(target_dirname, "curves.png"))
    #plt.clf()

    plot_name_list = ["Qr", "Qc", "Qi", "fr"]
    plot_list = [Qr_list_list, Qc_list_list, Qi_list_list, fr_list_list]
    color_list_2 = ["r", "g", "b", "k"]
    #fig, axarr = plt.subplots(len(tmpcache_list[0]), 4, figsize=(16, 96), facecolor='white')
    try:
        os.system("mkdir %s"%os.path.join(target_dirname, "fig_Qr_Qc_Qi_fr"))
    except:
        pass
    for j in range(len(fr_list_list)):
        fig, axarr = plt.subplots(1, 4, figsize=(16, 8), facecolor='white')
        for i in range(4):
            plot_name = plot_name_list[i]
            plot = plot_list[i]
            #axarr[j, i].scatter(P_list, np.log10(plot[j]), color=color_list_2[i], label=plot_name)
            #axarr[j, i].plot(P_list, np.log10(plot[j]), color=color_list_2[i], label=plot_name)
            #axarr[j, i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            #axarr[j, i].legend()
            #axarr[j, i].set_title("kidid: %s (log)"%str(j), y=1.02)
            #axarr[j, i].grid()
            #axarr[j, i].set_xlim(P_list[0]-5, P_list[-1]+5)
            axarr[i].scatter(P_list, np.log10(plot[j]), color=color_list_2[i], label=plot_name)
            axarr[i].plot(P_list, np.log10(plot[j]), color=color_list_2[i], label=plot_name)
            axarr[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            axarr[i].legend()
            axarr[i].set_title("kidid: %s (log)"%str(j), y=1.02)
            axarr[i].grid()
            axarr[i].set_xlim(P_list[0]-5, P_list[-1]+5)
        plt.savefig(os.path.join(target_dirname, "fig_Qr_Qc_Qi_fr", "Qr_Qc_Qi_fr_%s.png"%str(j).zfill(5)))
        plt.clf()
    #plt.savefig(os.path.join(target_dirname, "Qr_Qc_Qi_fr.png"))
    #plt.clf()


    Amp_center_list_list = []
    Freq_center_list_list = []
    for kidid in range(len(tmpcache_list[0])):
        Amp_center_list, Freq_center_list = [], []
        for tmpcache in tmpcache_list:
            kid = tmpcache[kidid]
            swpdata = kid.raw_sweep()
            Amp_center_list.append(swpdata.amplitude[175])
            Freq_center_list.append(swpdata.f[175])
        Amp_center_list_list.append(Amp_center_list)
        Freq_center_list_list.append(Freq_center_list)

    threshold = 0.999999      ###################

    x_over_threshold_max_list = []
    #fig, axarr = plt.subplots(len(tmpcache_list[0]), 1, figsize=(16, 96), facecolor='white')
    try:
        os.system("mkdir %s"%os.path.join(target_dirname, "fig_Psweep"))
    except:
        pass
    np.save("%s"%os.path.join(target_dirname, "fr_list_list.npy"), np.array(fr_list_list))
    np.save("%s"%os.path.join(target_dirname, "P_list.npy"), np.array(P_list))
    np.save("%s"%os.path.join(target_dirname, "Freq_center_list_list.npy"), np.array(Freq_center_list_list))
    for j in range(len(fr_list_list)):
        fig, axarr = plt.subplots(1, 1, figsize=(16, 8), facecolor='white')
        x = np.array(P_list)
        y = np.array(Freq_center_list_list[j])*1e9
        y_max = np.nanmax(y)
        y_argmax = np.nanargmax(y)
        y_threshold = y_max*threshold
        x_over_threshold = x[y>y_threshold]
        y_over_threshold = y[y>y_threshold]
        x_over_threshold_max = x_over_threshold[-1]
        y_over_threshold_max = y_over_threshold[-1]
        
        #axarr[j].scatter(x, np.log10(y), color="k", label="Freq_center")
        #axarr[j].plot(x, np.log10(y), color="k", label="Freq_center")
        #axarr[j].scatter(x[y_argmax], np.log10(y_max), color="r")
        #axarr[j].plot([P_list[0]-5, P_list[-1]+5], [np.log10(y_threshold), np.log10(y_threshold)], color="r", linestyle="--")
        #axarr[j].scatter(x_over_threshold_max, np.log10(y_over_threshold_max), color="b", marker="*", s=500)
        #axarr[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #axarr[j].legend()
        #axarr[j].set_title("kidid: %s (log)"%str(j), y=1.02)
        #axarr[j].grid()
        #axarr[j].set_xlim(P_list[0]-5, P_list[-1]+5)
        axarr.scatter(x, np.log10(y), color="k", label="Freq_center")
        axarr.plot(x, np.log10(y), color="k", label="Freq_center")
        axarr.scatter(x[y_argmax], np.log10(y_max), color="r")
        axarr.plot([P_list[0]-5, P_list[-1]+5], [np.log10(y_threshold), np.log10(y_threshold)], color="r", linestyle="--")
        axarr.scatter(x_over_threshold_max, np.log10(y_over_threshold_max), color="b", marker="*", s=500)
        axarr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axarr.legend()
        axarr.set_title("kidid: %s (log)"%str(j), y=1.02)
        axarr.grid()
        axarr.set_xlim(P_list[0]-5, P_list[-1]+5)
        x_over_threshold_max_list.append(x_over_threshold_max)
        plt.savefig(os.path.join(target_dirname, "fig_Psweep", "Psweep_%s.png"%str(j).zfill(5))) 
        plt.clf()   
    #plt.savefig(os.path.join(target_dirname, "Psweep.png")) 
    #plt.clf()   

    x_over_threshold_max_array = np.array(x_over_threshold_max_list) - 4.0    ######
    x_over_threshold_max_array = np.where(x_over_threshold_max_array < P_list[0], P_list[0], x_over_threshold_max_array)
    print(x_over_threshold_max_array)

    now = datetime.datetime.now()
    ts_now = now.strftime("%Y%m%d%H%M%S")

    os.system("cp %s %s"%(os.path.join(target_dirname, "kids.list"), os.path.join(target_dirname, "bu_%s_kids.list"%ts_now)))
    replace_values(os.path.join(target_dirname, "kids.list"), x_over_threshold_max_array)
    
    """
    #hdu_dict = {}
    for i, P in enumerate(P_list):
        #hdu_dict[str(P)] = fits.open(os.path.join(anapath_list[i], "reduced_measurement.fits"))
        hdu = fits.open(os.path.join(anapath_list[i], "reduced_measurement.fits"))
        ts = copy.deepcopy(hdu['READOUT'].data['timestamp'])
        ampl, phase, linphase = copy.deepcopy(hdu['READOUT'].data['Amp, Ph, linPh %d' %i].T)
        plot_tod_ratio = 0.25
        ts = ts - ts[0]
        size = 2**int(np.floor(np.log2(len(ts)*plot_tod_ratio)))
        f_, ampl_ = md.power_spectrum_density(ampl[:size], dt, 7, window=None, overwrap_half=True)
        f_, phase_ = md.power_spectrum_density(phase[:size], dt, 7, window=None, overwrap_half=True)

    fig, axarr = plt.subplots(1, 1, figsize=(16, 8), facecolor='white')
    """


def replace_values(file_path, replacements):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_index = next(i for i, line in enumerate(lines) if '#KIDs' in line) + 1
    end_index = start_index + len(replacements)
    for i in range(start_index, end_index):
        parts = lines[i].split()
        parts[-1] = str(int(replacements[i - start_index]))
        lines[i] = ' '.join(parts) + '\n'
    with open(file_path, 'w') as file:
        file.writelines(lines)

def get_color_from_cmap(value, cmap_name='jet'):
    import matplotlib
    cmap = plt.get_cmap(cmap_name)
    rgba_color = cmap(value)
    hex_color = matplotlib.colors.rgb2hex(rgba_color)
    return hex_color               



if __name__ == '__main__':
    import os, sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)
    import mkid_data as md
    from mkid_data import fit_onepeak
    import matplotlib.pyplot as plt
    import numpy as np
    from glob import glob
    import bbsweeplib as bbs
    from common import *
    import datetime
    import copy
    
    args = sys.argv

    main(args[1], args[2])

