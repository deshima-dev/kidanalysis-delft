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
    #parser.add_argument('kidslist', nargs='?')
    #parser.add_argument('localsweep', nargs='?')
    #parser.add_argument('TOD_fits', nargs='?')
    #parser.add_argument('localsweep_roomchopper', nargs='?')

    parser.add_argument('--force', action='store_true') #,
    #                    help='recreate "out" and "conf" directory')
    #parser.add_argument('--framelen', type=int, help='frame length spec, 16 or 19')

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
    targetdir = args.targetdir or input('target directory:')
    #run_id = [s for s in targetdir.split("/") if s[:3]=="run"][0]
    #print("mkdir -p /home/deshima/data/analysis/%s"%(run_id))
    #os.system("mkdir -p /home/deshima/data/analysis/%s"%(run_id))
    outdir  = args.outdir or input('out directory (e.g., out_test):')
    confdir = os.path.join(outdir, 'conf')
    disabled_kid_file = os.path.join(outdir, 'disabled_kids.dat')

    configfile = 'general.conf'
    current_outdir = './current_outdir.conf'
    #
    kidslist_path   = os.path.join(targetdir, 'kids.list')
    localsweep_path = os.path.join(targetdir, 'localsweep.sweep')
    TOD_fits_path = glob.glob(os.path.join(targetdir, '*.fits'))[0]
    localsweep_roomchopper_path = os.path.join(targetdir, 'RoomChopperClosed', 'localsweep.sweep')
    if not os.path.exists(localsweep_roomchopper_path):
        localsweep_roomchopper_path = ''


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
    config.set('datafiles', 'TOD_fits',   TOD_fits_path)
    config.set('datafiles', 'localsweep_roomchopper', localsweep_roomchopper_path)


    import bbsweeplib
    config.add_section('kids_analysis')
    config.set('kids_analysis',
               '; kids indices for use in analysis')

    if len(localsweep_roomchopper_path)>0:
        kids = bbsweeplib.KIDs('%s/aste' %(outdir),
                            kidslist_path, localsweep_path, TOD_fits_path,
                            sweeps_path_roomchopper=localsweep_roomchopper_path)
    else:
        kids = bbsweeplib.KIDs('%s/aste' %(outdir),
                            kidslist_path, localsweep_path, TOD_fits_path)
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(os.path.dirname(script_dir), 'libs')
    sys.path.append(libpath)

    main()

