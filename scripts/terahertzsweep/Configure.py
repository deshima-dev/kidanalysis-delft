#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Create output directory "out" and config directory "out/conf",
and config file "out/conf/general.conf",
out/kids dir to store kid-related data files.

run:
# 1. specify data files (asked by prompt)
# 2. modify config data at out/conf/general.conf
#  - DeltaHz in [values]: gap frequency. used when determining 
#        lowest frequency for table of filter transparency
#  - [powerfreqbin]: used for radiation power calculation (numerical integral)
#  - etendue in [optics]: optical throughput
#  - [filterfiles]: files of filter, and transparency at low frequency limit
#  - [filterstack]: all filters used

#  - [kids_analysis]: kids to use in analysis scripts
# 3. run stage1.py
#   $ ./stage1.py
# 4. if you want to check, make plots and view:
#   $ ./stage1.py --mode plot
#   $ open out/*.png
# 5. modify [kids_analysis] section of config file if needed
# 6. set t0, t1, dP_rad for slicing TOD in stage2.py: argument of SetSlice.main()
# 7. set f0, (f1) for averaging PSD in stage2.py: argument of SetAverage.main()
# 8. run stage2.py
#   $ ./stage2.py
# 9. if you want to check, make plots and view:
#   $ ./stage2.py --mode plot
#   or plot only last result: $ ./NEPData.py --mode plot
#   $ open out/SetSlice/*.png
"""

import os
import sys
import argparse
import shutil

from util import my_config_parser

def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('targetdir', nargs='?')
    parser.add_argument('outdir', nargs='?')
    parser.add_argument('kidslist', nargs='?')
    parser.add_argument('localsweep', nargs='?')
    parser.add_argument('THzsweep_freq', nargs='?')
    parser.add_argument('THzsweep_fits', nargs='?')

    parser.add_argument('--force', action='store_true') #,
    #                    help='recreate "out" and "conf" directory')
    parser.add_argument('--framelen', type=int, help='frame length spec, 16 or 19')

    args = parser.parse_args(argv)


    import readline, glob
    import os.path
    def complete(text, state):
        # http://stackoverflow.com/questions/6656819/filepath-autocompletion-using-users-input
        text_ = os.path.expanduser(text)
        comp = glob.glob(text_+'*')
        comp = [c + '/' if os.path.isdir(c) else c for c in comp]
        return (comp+[None])[state]
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)

    #
    ##outdir  = 'out'
    ##confdir = 'out/conf'
    ##disabled_kid_file = 'out/disabled_kids.dat'
    """
    outdir  = args.outdir or input('out directory:')
    confdir = outdir + '/conf'
    disabled_kid_file = outdir + '/disabled_kids.dat'

    configfile = 'general.conf'
    current_outdir = './current_outdir.conf'
    #
    
    y_or_n = input(f"Are the kidslist file, localsweep file, THzsweep frequency log file, and THzsweep fits TOD file in {os.path.dirname(outdir)} ? (y/n): ")
    
    if y_or_n=="n" or y_or_n=="N" or y_or_n=="no" or y_or_n=="No" or y_or_n=="NO":
        kidslist_path   = args.kidslist   or input('kidslist file:')
        localsweep_path = args.localsweep or input('localsweep file:')
        THzsweep_freq_path    = args.THzsweep_freq    or input('THzsweep frequency log file:')
        THzsweep_fits_path    = args.THzsweep_fits    or input('THzsweep fits TOD file:')
    elif y_or_n=="y" or y_or_n=="Y" or y_or_n=="yes" or y_or_n=="Yes" or y_or_n=="YES":
        kidslist_path   = os.path.join(os.path.dirname(outdir), "kids.list")
        localsweep_path = os.path.join(os.path.dirname(outdir), "localsweep.sweep")
        THzsweep_freq_path = os.path.join(os.path.dirname(outdir), "terasource.txt")
        THzsweep_fits_path = os.path.join(os.path.dirname(outdir), "measurement.fits")
    else:
        print("Please answer with y/n. ")
        sys.exit()
    """
    
    targetdir = args.targetdir or input('target directory:')
    run_id = [s for s in targetdir.split("/") if s[:3]=="run"][0]
    measure_id = os.path.basename(os.path.dirname(targetdir))
    #print("mkdir -p /home/deshima/data/analysis/%s/%s"%(run_id, measure_id))
    #os.system("mkdir -p /home/deshima/data/analysis/%s/%s"%(run_id, measure_id))
    print("mkdir -p /Users/sfujita/Desktop/DESHIMA/toptica/analysis/%s/%s"%(run_id, measure_id))
    os.system("mkdir -p /Users/sfujita/Desktop/DESHIMA/toptica/analysis/%s/%s"%(run_id, measure_id))
    outdir  = args.outdir or input('out directory (e.g., out_test):')
    confdir = outdir + '/conf'
    disabled_kid_file = outdir + '/disabled_kids.dat'

    configfile = 'general.conf'
    current_outdir = './current_outdir.conf'
        
    kidslist_path   = os.path.join(targetdir, "kids.list")
    localsweep_path = os.path.join(targetdir, "localsweep.sweep")
    THzsweep_freq_path = os.path.join(targetdir, "terasource.txt")
    THzsweep_fits_path = os.path.join(targetdir, "measurement.fits")
        

    if args.force:
        try:
            shutil.rmtree(outdir)
        except OSError:
            pass
        # shutil.rmtree(confdir)

    #### make data output directory
    os.makedirs(outdir)
    #### make config files
    os.makedirs(confdir)


    ## general
    config = my_config_parser()
    config.optionxform = str
    config.add_section('datafiles')
    config.set('datafiles',
               '; paths to data files')
    config.set('datafiles', 'kidslist',   kidslist_path)
    config.set('datafiles', 'localsweep', localsweep_path)
    config.set('datafiles', 'THzsweep_freq',   THzsweep_freq_path)
    config.set('datafiles', 'THzsweep_fits',   THzsweep_fits_path)


    import bbsweeplib
    config.add_section('kids_analysis')
    config.set('kids_analysis',
               '; kids indices for use in analysis')

    kids = bbsweeplib.KIDs('%s/THzsweep' %(outdir),
                           kidslist_path, localsweep_path, THzsweep_fits_path,
                           framelen=args.framelen)
    kids.save()

    # for i, k in enumerate(kids):
    #     config.set('enabled_kids', 'KID[%d]' % i, 1)
    enabled_indices = []
    for i, k in kids.items():
        if k.enabled:
            enabled_indices.append(i)
        else:
            print( 'KID[%d] was disabled (not both of blind tone)' % i )
    config.set('kids_analysis', 'disabled_kid_file', disabled_kid_file)

    with open(disabled_kid_file, 'w') as f:
        print( '## %d kids' % len(kids), file=f )
        for i, k in kids.items():
            if not k.enabled:
                print( i, file=f )

    # kids = bbsweeplib.KIDs(kidslist_path, localsweep_path, tod_path)
    # for i, k in enumerate(kids.bins_kid()):
    #     config.set('kids', 'KID[%d]' % i, 1)

    with open(os.path.join(confdir, configfile), 'w') as f:
        config.write(f)

    with open(current_outdir, 'w') as f:
        print( outdir, file=f )


if __name__ == '__main__':
    sys.path.append('../libs')

    main()

