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
    parser.add_argument('--del_1st_or_2nd', type=str, default="2nd",
                        help='If there are overlapping KIDs, which one should be deleted')

    args = parser.parse_args(argv)
    threshold_fr_diff = args.threshold_fr_diff
    del_1st_or_2nd = args.del_1st_or_2nd


    fit_res_dir = os.path.join(outdir, "THzsweep")
    dir_list = sorted(glob(os.path.join(fit_res_dir, "[0-9]*[0-9]")))

    raw_sweep_list = []
    fit_list = []
    for dir in dir_list:
        try:
            with open(os.path.join(dir, "raw_sweep"), 'rb') as f:
                raw_sweep = pickle.load(f)
                raw_sweep_list.append(raw_sweep)
        except:
            raw_sweep_list.append(np.nan)

        try:
            with open(os.path.join(dir, "fit"), 'rb') as f:
                fit = pickle.load(f)
                fit_list.append(fit)
        except:
            fit_list.append(np.nan)

    #residual_pmFrac, good_list, bad_list = pmFrac(Frac, threshold, fr_offset_frac)
    residual_pmFrac, good_list, bad_list = pmFrac(fit_list, raw_sweep_list, 2.0, 50e-9, 0.2)
    print("bad_list", bad_list)
    f = open(os.path.join(outdir, "flag_chi2.dat"), "w")
    f.write("# flag_chi2\n")
    for i in bad_list:
        f.write("%i\n"%i)
    f.close()

    reftxtlist = outdir + '/reference.dat'
    reference = np.array(my_loadtxt(reftxtlist, unpack=True, skiprows=1)[0])
    print("reference", reference)

    bad_candidates = np.array([i for i in reference if i in good_list])
    bad_list = np.union1d(bad_list, bad_candidates)
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
    np.save(os.path.join(outdir,"frs.npy"), frs)

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
    np.save(os.path.join(outdir,"kid_ids_good.npy"), kid_ids_good)
    np.save(os.path.join(outdir,"fr_diffs.npy"), fr_diffs)
    print("adjacent_pairs =", adjacent_pairs)

    bad_list_fr_diff = []
    for a, b in adjacent_pairs:
        if del_1st_or_2nd=="1st":
            bad_list_fr_diff.append(a)  
        elif del_1st_or_2nd=="2nd":
            bad_list_fr_diff.append(b)  
        else:
            print('"del_1st_or_2nd" should be 1st or 2nd')
            sys.exit()      

    print("bad_list_fr_diff", bad_list_fr_diff)
    f = open(os.path.join(outdir, "flag_overlap_delete_%s.dat"%del_1st_or_2nd), "w")
    f.write("# flag_overlap_%s\n"%del_1st_or_2nd)
    for i in bad_list_fr_diff:
        f.write("%i\n"%i)
    f.close()
    bad_candidates = np.array([i for i in bad_list_fr_diff if i in good_list]).astype("int")
    bad_list = np.union1d(bad_list, bad_candidates).astype("int")
    good_list = np.setdiff1d(good_list, bad_list).astype("int")
    bad_list = np.sort(bad_list).astype("int")
    print("bad_list", bad_list)

    #kid_ids = np.array([i for i in range(len(raw_sweep_list))])

    plt.figure(figsize=(16, 8))
    plt.scatter(kid_ids[good_list], residual_pmFrac[good_list], marker="o", c="b", label="Good")
    plt.scatter(kid_ids[bad_list], residual_pmFrac[bad_list], marker="x", c="r", label="Bad")
    plt.grid()
    plt.xlabel("KID ID", size=20)
    plt.ylabel(r"$\chi^2$", size=20)
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(outdir, "flag_delete_%s.png"%del_1st_or_2nd))

    f = open(os.path.join(outdir, "flag_delete_%s.dat"%del_1st_or_2nd), "w")
    f.write("# flag\n")
    for i in bad_list:
        f.write("%i\n"%i)
    f.close()

    reftxtlist = outdir + '/reference.dat'
    reference = np.array(my_loadtxt(reftxtlist, unpack=True, skiprows=1)[0])

    reference_new = np.unique(np.concatenate([reference, bad_list]))

    os.system(f"mv {reftxtlist} {outdir + '/reference_ori.dat'}")

    f = open(os.path.join(outdir, "reference.dat"), "w")
    f.write("# reference\n")
    for i in reference_new:
        f.write("%i\n"%i)
    f.close()


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

    from common import *
    main()

