#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Make config directory, config files, output directory
"""


import os
import os.path
import sys
import argparse
import configparser

import numpy as np

def my_config_parser():
    config = configparser.RawConfigParser(allow_no_value=True)
    config.optionxform = str
    return config


## make config parser, with both functinos of RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
class myHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    ## http://svn.python.org/projects/python/branches/pep-0384/Lib/argparse.py
    def _fill_text(self, text, width, indent):
        return ''.join([indent + line for line in text.splitlines(True)])


## loadtxt using pandas (as numpy.loadtxt is very slow...)
import pandas as pd
def my_loadtxt(file, usecols=None, unpack=None, skiprows=0, delimiter=' '):
    comment_char = '#'
    if isinstance(file, str) or isinstance(file, unicode):
        if file[-3:] == '.gz':
            compression='gzip'
        elif file[-4:] == '.bz2':
            compression='bz2'
        else:
            compression=None
    else:
        compression=None
    if delimiter=='\t' or delimiter==' ':
        ar = np.array(pd.read_csv(file, compression=compression, delim_whitespace=True,
                                  comment='#', header=None, usecols=usecols, skiprows=skiprows))
    else:
        ar = np.array(pd.read_csv(file, compression=compression, delimiter=delimiter, engine='python',
                                  comment='#', header=None, usecols=usecols, skiprows=skiprows))
    if unpack:
        return ar.T
    else:
        return ar

# http://stackoverflow.com/questions/10512026/reading-data-blocks-from-a-file-in-python
def read_blocks(input_file):
    empty_lines = 0
    blocks = []
    for line in open(input_file):
        # Check for empty/commented lines
        if not line or line.startswith('#'):
            # If 1st one: new block
            if empty_lines == 0:
                blocks.append([])
            empty_lines += 1
        # Non empty line: add line in current(last) block
        else:
            empty_lines = 0
            blocks[-1].append(line)
    return blocks



#### loading files.
def load_file(fname, kind='array'):
    if kind == 'array':
        return my_loadtxt(fname)
    if kind == 'blocks':
        from io import StringIO
        return [my_loadtxt(StringIO(u''.join(b))) for b in read_blocks(fname)]
    # elif kind == 'eval':
    #     # import ast
    #     # with open(fname) as f:
    #     #     return ast.literal_eval(f.read())
    #     with open(fname) as f:
    #         return eval(f.read())
    elif kind == 'kidslist':
        import bbsweeplib as bbs
        return bbs.read_kidslist(fname)
    elif callable(kind):
        return func(fname)


#import ROOT
###### get x and y of ROOT.TGraph as numpy array
#def GetGraphXY(gr):
#    N = int(gr.GetN())
#    grx = gr.GetX()
#    gry = gr.GetY()
#    
#    xList = np.array([grx[i] for i in range(N)])
#    yList = np.array([gry[i] for i in range(N)])
#
#    return (xList, yList)


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average( (values-average)**2, weights=weights )
    return (average, np.sqrt(variance))


def bad_index_by_fraction(par0, par1, thre0=6., thre1=5.):
    val = np.array(par1)/np.array(par0)
    mean = np.average(val)
    std  = np.std(val)
    idx_ar = np.where( abs(val-mean)<thre0*std )

    mean = np.average( val[idx_ar[0]] )
    std  = np.std( val[idx_ar[0]] )
    idx_ar = np.where( abs(val-mean)>thre1*std )
    
    return idx_ar[0]


def delete_elements(inarray, delidx):
    outarray = []
    for ar in inarray:
        #ar_ = np.delete(ar, delidx, axis=0)
        #outarray.append(ar_)
        tmp = np.ones(len(ar), dtype=bool)
        tmp[delidx] = False
#        outarray.append(np.array(ar, dtype=object)[tmp])
        outarray.append(np.array(ar)[tmp])
        
    return outarray


def rebin_array(inarray, rebin):
    return inarray[:(inarray.size//rebin)*rebin].reshape(-1, rebin).mean(axis=1)

#####
##### for fits file
from astropy.io import fits
#---------------- Create Binary Table HDU
def createBinTableHDU(data_dict):
    #-------- Set Header and Comments
    header = fits.Header()
    for (i, j, k) in zip(data_dict['hdr_key_lis'], data_dict['hdr_val_lis'], data_dict['hdr_com_lis']):
        header[i] = j, k

    #-------- Create Collumns of the Binary Table
    columns = []
    for i in range( len(data_dict['cols_key_lis']) ):
        columns.append( fits.Column(name=data_dict['cols_key_lis'][i],
                                    format=data_dict['tform'][i],
                                    array=data_dict['cols_data_lis'][i],
                                    unit=data_dict['tunit'][i]) )

    hdu = fits.BinTableHDU.from_columns(columns, header)

    #-------- Add comments
    for i in range( -(len(data_dict['hdr_com_lis'])-data_dict['hdr_com_lis'].index('label for field 0')), 0 ):
        addHeaderComments(hdu.header, list(hdu.header)[i], data_dict['hdr_com_lis'][i])
    
    return hdu

#---------------- Add Comments to Header
def addHeaderComments(hdr, key, com):
    hdr.comments[key] = com

def readout_dict():
    hdr_key_lis = ['EXTNAME', 'FILENAME', 'FRAMERT', 'FRAMELEN', 'DSAMPLE',]
    hdr_val_lis = ['READOUT', None, None, None, None]
    hdr_com_lis = ['name of binary data',
                   'input filename',
                   'sampling rate',
                   '2-log of frame length',
                   'number of down sampling',
                   'label for field 0', 'data format of field 0',
                   'label for field 1', 'data format of field 1',]
    cols_key_lis = ['timestamp', 'pixelid',]
    cols_data_lis = []
    tform = ['D', 'I',]
    tunit = [None, None,]

    r_dict = {'hdr_key_lis': hdr_key_lis,
              'hdr_val_lis': hdr_val_lis,
              'hdr_com_lis': hdr_com_lis,
              'cols_key_lis': cols_key_lis,
              'cols_data_lis': cols_data_lis,
              'tform': tform,
              'tunit': tunit}
    return r_dict

def kids_dict():
    hdr_key_lis = ['EXTNAME', 'FILENAME',]
    hdr_val_lis = ['KIDSINFO', None,]
    hdr_com_lis = ['name of binary data',
                   'localsweep filename',
                   'label for field 0', 'data format of field 0',
                   'label for field 1', 'data format of field 1',
                   'label for field 2', 'data format of field 2', 'data unit of field 2',
                   'label for field 3', 'data format of field 3', 'data unit of field 3',
                   'label for field 4', 'data format of field 4',
                   'label for field 5', 'data format of field 5', 'data unit of field 5',
                   'label for field 6', 'data format of field 6',
                   'label for field 7', 'data format of field 7',
                   'label for field 8', 'data format of field 8',
                    'label for field 9', 'data format of field 9', 'data unit of field 9']
    cols_key_lis = ['pixelid', 'kidid', 'Pread', 'fc', 'yfc, linyfc',
                    'fr, dfr (Sky)', 'Qr, dQr (Sky)', 'Qc, dQc (Sky)', 'Qi, dQi (SKy)',
                    'fr, dfr (Room)']
    cols_data_lis = []
    tform = ['I', 'I', 'E', 'E', '2E', '2E', '2E', '2E', '2E', '2E']
    tunit = [None, None, 'dBm', 'GHz', None, 'GHz', None, None, None, 'GHz']

    k_dict = {'hdr_key_lis': hdr_key_lis,
              'hdr_val_lis': hdr_val_lis,
              'hdr_com_lis': hdr_com_lis,
              'cols_key_lis': cols_key_lis,
              'cols_data_lis': cols_data_lis,
              'tform': tform,
              'tunit': tunit}
    return k_dict

