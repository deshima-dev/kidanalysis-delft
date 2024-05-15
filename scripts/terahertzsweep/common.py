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
script_dir = os.path.dirname(os.path.abspath(__file__))
current_outdir = os.path.join(script_dir, 'current_outdir.conf')

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
kids = bbs.Cache.load(outdir + '/THzsweep')

#####
##### change path to fits file (for temporary solution)
#def get_data_path(runid):
#    import sqlite3
#    dbname = '/Users/spacekids/database/kid_test.telesto.db'
#    conn = sqlite3.connect(dbname, 
#                           detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
#    sqlite3.dbapi2.converters['DATETIME'] = sqlite3.dbapi2.converters['TIMESTAMP']
#    c = conn.cursor()
#    
#    dd_path_sql = '''
#    SELECT data_dir_path FROM main
#    WHERE runid = ?
#    '''
#    c.execute(dd_path_sql, (runid,))
#    #data_path = [ddpath for (ddpath ,) in c.fetchall()]
#    data_path = c.fetchall()[0][0]
#    if not os.path.exists(data_path):
#        print '>>>>> telesto HDD might not be mounted.'
#        print '>>>>> Check whether there exist backup data on hirado ...'
#        data_path = os.path.join(*([u'/', u'Volumes', u'hirado', u'telesto', u'spacekids', u'data'] + data_path.split('/')[4:]))
#    
#    return data_path
#
#data_path = kids[0].fits_path
#runid = int( os.path.basename(data_path).split('.')[1] )
#print runid
#data_dir = get_data_path(runid)
#data_path_update = os.path.join(data_dir, os.path.basename(data_path))
#kids._tods_path=data_path_update
#print kids.raw_tods().infile
#####
#####

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

