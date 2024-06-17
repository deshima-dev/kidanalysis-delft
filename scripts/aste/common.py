#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import os.path
import sys
import argparse
import configparser
import pickle

import matplotlib.pyplot as plt
import numpy as np

from util import my_config_parser, myHelpFormatter, my_loadtxt, read_blocks, load_file

import mkid_data as md
import bbsweeplib as bbs

## output directories
#outdir    = 'out/'
current_outdir = './current_outdir.conf'

with open(current_outdir) as f:
    outdir = f.readline().strip()
print( '===== Output directory: %s =====' %outdir )
confdir   = outdir + '/conf/'
slice_dir = outdir + '/SetSlice/'


## config file names
configfile = 'general.conf'


## read config file
config = my_config_parser()
config.read(os.path.join(confdir, configfile))

## data file locations
kidslistfile = config.get('datafiles', 'kidslist')
default_verbose_level = 1

import ast
#kids_paths = ast.literal_eval(config.get('datafiles', 'THzsweep_fits'))
kids = bbs.Cache.load(outdir + '/aste')

# set 'enabled' from configure file
disabled_kid_file = config.get('kids_analysis', 'disabled_kid_file')
disabled_kid_indices = []
with open(disabled_kid_file) as f:
    for l in f:
        l = l.strip()
        if not l or (l[0] == '#'): continue
        disabled_kid_indices.append(int(l))

for i, kid in kids.items():
    if i in disabled_kid_indices:
        kid.enabled = False
    else:
        kid.enabled = True

del i, kid, disabled_kid_indices

