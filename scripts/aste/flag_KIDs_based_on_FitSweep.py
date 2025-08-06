#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import shutil

from collections import OrderedDict

import matplotlib.pyplot as plt
#import mkid_data as md
import numpy as np
from glob import glob

def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=myHelpFormatter, description=__doc__)
    parser.add_argument('--threshold_fr_diff', type=float, default=1.0e-4,
                        help='Threshold of difference between fr with neighboring KID. If it is less than this, one of them will be deleted.')
    parser.add_argument('--threshold_chi2', type=float, default=5.0e-8,
                        help='threshold of chi2 to be deleted.')
    parser.add_argument('--frac_FWHM', type=float, default=2.0,
                        help='calculation range of chi2 (+-FWHM*?)')
    parser.add_argument('--frac_fr', type=float, default=0.2,
                        help='check to see if fr is too far apart.')
    parser.add_argument('--del_1st_or_2nd', type=str, default="2nd",
                        help='If there are overlapping KIDs, which one should be deleted.')
    #parser.add_argument('--terahertzsweep_or_aste', type=str, default="obs",
    #                    help='for THzsweep [terahertzsweep] or observations (COSMOS dir) [aste].')

    args = parser.parse_args(argv)
    threshold_fr_diff = args.threshold_fr_diff
    threshold_chi2 = args.threshold_chi2
    frac_FWHM = args.frac_FWHM
    frac_fr = args.frac_fr
    del_1st_or_2nd = args.del_1st_or_2nd
    #terahertzsweep_or_aste = args.terahertzsweep_or_aste

    #if toptica_or_obs=="terahertzsweep":
    #    fit_res_dir = os.path.join(outdir, "THzsweep")
    #elif toptica_or_obs=="aste":
    #    fit_res_dir = os.path.join(outdir, "aste")
    #else:
    #    print('"terahertzsweep_or_aste" should be terahertzsweep or aste')
    #    sys.exit()

    #dir_list = sorted(glob(os.path.join(fit_res_dir, "[0-9]*[0-9]")))

    ##### save original disabled_kids.dat
    reftxtlist = outdir + '/disabled_kids.dat'
    os.system(f"cp -n {reftxtlist} {os.path.join(outdir, 'disabled_kids_ori.dat')}")

    #####
    def main_2(sky_or_room):
        if sky_or_room=="sky":
            rs = 'raw_sweep'
            fit_name = "fit"
        elif sky_or_room=="room":
            rs = 'raw_sweep_roomchopper'
            fit_name = "fit_roomchopper"
        raw_sweep_list = []
        fit_list = []
        for i, kid in kids.items():
            if kid.enabled:
                raw_sweep_list.append( kid.get_cache(rs) )
                fit_list.append( kid.get_cache(fit_name) )            
            else:
                raw_sweep_list.append(np.nan)
                fit_list.append(np.nan)

        #residual_pmFrac, good_list, bad_list = pmFrac(Frac, threshold, fr_offset_frac)
        residual_pmFrac, good_list, bad_list = pmFrac(fit_list, raw_sweep_list, frac_FWHM, threshold_chi2, frac_fr)
        print(f"bad_list from chi2 ({sky_or_room})", bad_list)

        reftxtlist = outdir + f'/disabled_kids.dat'
        os.system(f"cp {os.path.join(outdir, 'disabled_kids_ori.dat')} {reftxtlist}")

        with open(reftxtlist, 'a') as f:
            print( '# below are append due to bad chi2', file=f )
            for i in bad_list:
                print( i, file=f )

        bad_list = np.array(my_loadtxt(reftxtlist, unpack=True, skiprows=1)[0])
        good_list = np.setdiff1d(good_list, bad_list)
        bad_list = np.sort(bad_list)

        kid_ids = []
        frs = []
        for i in range(len(fit_list)):
          kid_ids.append(i)
          try:
            fr = fit_list[i]._result.params["fr"].value
            frs.append(fr)
          except:
            frs.append(np.nan)
        kid_ids = np.array(kid_ids)
        frs = np.array(frs)
        np.save(os.path.join(outdir,f"frs_{sky_or_room}.npy"), frs)

        #diff_next = np.abs(np.diff(frs))
        #ids_next = kid_ids[:-1]

        #threshold_fr_diff = 1e-4 #########
        mask_good = ~np.isin(kid_ids, bad_list)
        kid_ids_good = kid_ids[mask_good]
        frs_good = frs[mask_good]

        sorted_indices = np.argsort(kid_ids_good)
        kid_ids_good = kid_ids_good[sorted_indices]
        frs_good = frs_good[sorted_indices]

        fr_diffs = np.abs(np.diff(frs_good))
        adjacent_pairs = [(kid_ids_good[i], kid_ids_good[i+1])
                      for i in range(len(frs_good) - 1)
                      if fr_diffs[i] <= threshold_fr_diff]
        np.save(os.path.join(outdir,f"kid_ids_good_{sky_or_room}.npy"), kid_ids_good)
        np.save(os.path.join(outdir,f"fr_diffs_{sky_or_room}.npy"), fr_diffs)
        print(f"adjacent_pairs ({sky_or_room}) =", adjacent_pairs)

        bad_list_fr_diff = []
        for a, b in adjacent_pairs:
            if del_1st_or_2nd=="1st":
                bad_list_fr_diff.append(a)  
            elif del_1st_or_2nd=="2nd":
                bad_list_fr_diff.append(b)  
            else:
                print('"del_1st_or_2nd" should be 1st or 2nd')
                sys.exit()      

        print(f"bad_list_fr_diff ({sky_or_room})", bad_list_fr_diff)
        with open(reftxtlist, 'a') as f:
            print( '# below are append due to overlap %s' %del_1st_or_2nd, file=f )
            for i in bad_list_fr_diff:
                print( i, file=f )

        bad_candidates = np.array([i for i in bad_list_fr_diff if i in good_list]).astype("int")
        bad_list = np.union1d(bad_list, bad_candidates).astype("int")
        good_list = np.setdiff1d(good_list, bad_list).astype("int")
        bad_list = np.sort(bad_list).astype("int")
        print(f"bad_list ({sky_or_room})", bad_list)

        #kid_ids = np.array([i for i in range(len(raw_sweep_list))])

        plt.figure(figsize=(16, 8))
        plt.scatter(kid_ids[good_list], residual_pmFrac[good_list], marker="o", c="b", label="Good")
        plt.scatter(kid_ids[bad_list], residual_pmFrac[bad_list], marker="x", c="r", label="Bad")
        plt.grid()
        plt.xlabel("KID ID", size=20)
        plt.ylabel(r"$\chi^2$", size=20)
        plt.yscale("log")
        plt.legend()
        for i in bad_list:
            x = kid_ids[i]
            y = residual_pmFrac[i]
            if not np.isfinite(y):
                continue
            plt.text(x, y, str(int(x)), color="magenta", fontsize=10, ha="center", va="bottom")

        plt.savefig(os.path.join(outdir, f"flag_delete_{del_1st_or_2nd}_{sky_or_room}.png"))

    
        np.save(os.path.join(outdir,f"kid_ids_{sky_or_room}.npy"), kid_ids)
        np.save(os.path.join(outdir,f"residual_pmFrac_{sky_or_room}.npy"), residual_pmFrac)

        reftxtlist_new = outdir + f'/disabled_kids_{sky_or_room}.dat'
        com = f"mv {reftxtlist} {reftxtlist_new}"
        os.system(com)
        
        return bad_list, bad_list_fr_diff

    bad_list_sky, bad_list_fr_diff_sky = main_2("sky")
    bad_list_room, bad_list_fr_diff_room = main_2("room")

    bad_list_concat = np.concatenate([bad_list_sky, bad_list_room])
    bad_list_fr_diff_concat = np.concatenate([bad_list_fr_diff_sky, bad_list_fr_diff_room])

    reftxtlist = outdir + f'/disabled_kids.dat'
    os.system(f"cp {os.path.join(outdir, 'disabled_kids_ori.dat')} {reftxtlist}")

    unique_vals, counts = np.unique(bad_list_concat, return_counts=True)
    once_only = unique_vals[counts == 1]
    print(f"bad KID ID from chi2 (appeared once): {once_only}")
    unique_vals, counts = np.unique(bad_list_fr_diff_concat, return_counts=True)
    once_only = unique_vals[counts == 1]
    print(f"bad KID ID from fr (appeared once): {once_only}")

    with open(reftxtlist, 'a') as f:
        print( '# below are append due to bad chi2', file=f )
        for i in np.unique(bad_list_concat):
            print( str(int(i)), file=f )
    with open(reftxtlist, 'a') as f:
            print( '# below are append due to overlap %s' %del_1st_or_2nd, file=f )
            for i in np.unique(bad_list_fr_diff_concat):
                print( str(int(i)), file=f )



    """
    f = open(os.path.join(outdir, "flag_delete_%s.dat"%del_1st_or_2nd), "w")
    f.write("# flag\n")
    for i in bad_list:
        f.write("%i\n"%i)
    f.close()
   """

def gao_model(f, params):
    arga  = params['arga'].value
    absa  = params['absa'].value
    tau   = params['tau'].value
    fr    = params['fr'].value
    Qr    = params['Qr'].value
    Qc    = params['Qc'].value
    phi0  = params['phi0'].value
    c     = params['c'].value

    df = f - fr
    background = c * df + 1
    resonance = Qr * np.exp(1j * phi0) / (Qc * (1 + 2j * Qr * df / fr))
    system_phase = np.exp(1j * (arga - 2 * np.pi * tau * f))

    return absa * (background - resonance) * system_phase

def residual(params, f, data):
    model = gao_model(f, params)
    return np.concatenate([(np.real(model) - np.real(data)), (np.imag(model) - np.imag(data))])**2    

def pmFrac(fit_list, raw_sweep_list, Frac, threshold, fr_offset_frac):
    residual_sum_pmFrac_list = []
    good_list, bad_list = [], []
    is_first = True
    for i in range(len(raw_sweep_list)):
        try:
            params_test = fit_list[i]._result.params
            f_test = raw_sweep_list[i].f
            data_test = raw_sweep_list[i].iq

            residual_test = residual(params_test, f_test, data_test)

            fr = fit_list[i]._result.params["fr"].value
            Qr = fit_list[i]._result.params["Qr"].value
            FWHM = fr/Qr

            len_f = len(f_test)
            ind_fr = np.argmin(np.abs(f_test - fr))

            if is_first:
                # print(i)
                print("fr = ", fr, "GHz")
                print("Qr = ", Qr)
                print("FWHM = ", FWHM*1e3, "MHz")
                print("pmFrac = ", FWHM*Frac*1e3, "MHz")
                is_first = False

            ind_start = np.argmin(np.abs(f_test - (fr - FWHM*Frac)))
            ind_end = np.argmin(np.abs(f_test - (fr + FWHM*Frac)))
            residual_sum = float(np.sum(residual_test[ind_start:ind_end]))/(ind_end - ind_start)

            if ind_fr<0 or ind_fr>len_f-1:
                residual_sum_pmFrac_list.append(residual_sum)
                bad_list.append(i)
                continue

            if FWHM*1e3<0 or FWHM*1e3>1.0:
                residual_sum_pmFrac_list.append(residual_sum)
                bad_list.append(i)
                continue

            f_cen = np.nanmedian(f_test)
            ind_f_cen = np.argmin(np.abs(f_test - f_cen))
            if abs(ind_fr-ind_f_cen) > len_f*fr_offset_frac:
                residual_sum_pmFrac_list.append(residual_sum)
                bad_list.append(i)
                continue

            residual_sum_pmFrac_list.append(residual_sum)
            if residual_sum <= threshold:
                good_list.append(i)
            else:
                bad_list.append(i)
        except:
            residual_sum_pmFrac_list.append(np.nan)
            bad_list.append(i)
            continue
    return np.array(residual_sum_pmFrac_list), np.array(good_list), np.array(bad_list)



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)

    current_outdir = './current_outdir.conf'
    with open(current_outdir) as f:
        outdir = f.readline().strip()
    reftxtlist = outdir + '/disabled_kids.dat'
    if os.path.exists(os.path.join(outdir, 'disabled_kids_ori.dat')):
        com = f"cp {os.path.join(outdir, 'disabled_kids_ori.dat')} {reftxtlist}"
        print(com)
        os.system(com)

    from common import *
    main()


