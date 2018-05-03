''' Code for translating Bad Pixel Masks from a set of definitions to another
Function-driven instead of Class, for simpler paralellisnm
'''

import os
import sys
import time
import logging
import argparse
import copy
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
try:
    import matplotlib.pyplot as plt
except:
    pass
import fitsio

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Global variables
BITDEF_INI = None
BITDEF_END = None

def open_fits(fnm):
    ''' Open the FITS, read the data and store it on a list
    '''
    tmp = fitsio.FITS(fnm)
    data, header = [], []
    for extension in tmp:
        ext_tmp = np.copy(extension.read())
        data.append(ext_tmp)
        header_tmp = copy.deepcopy(extension.read_header())
        header.append(header_tmp)
    tmp.close()
    if (len(data) == 0):
        logging.error('No extensions were found on {0}'.format(fnm))
        exit(1)
    elif (len(data) >= 1):
        return data, header

def load_bitdef(d1=None, d2=None):
    ''' Load tables of bit definitions and pass them to a dictionary
    '''
    global BITDEF_INI
    global BITDEF_END
    try:
        kw = {
            'sep' : None,
            'comment' : '#', 
            'names' : ['def', 'bit'], 
            'engine' : 'python',
        }
        t1 = pd.read_table(d1, **kw)
        t2 = pd.read_table(d2, **kw)
    except:
        t_i = 'pandas{0} doesn\'t support guess sep'.format(pd.__version__) 
        logging.info(t_i)
        kw.update({'sep' : '\s+',})
        t1 = pd.read_table(d1, **kw)
        t2 = pd.read_table(d2, **kw)
    # Construct the dictionaries
    BITDEF_INI = dict(zip(t1['def'], t1['bit']))
    BITDEF_END = dict(zip(t2['def'], t2['bit']))
    # Change data type to unsigned integers, to match the dtype of the BPMs
    for k in BITDEF_INI:
        BITDEF_INI[k] = np.uint(BITDEF_INI[k])
    for j in BITDEF_END:
        BITDEF_END[j] = np.uint(BITDEF_END[k])
    return 

def split_bitmask():
    pass

def load_bitmask(fnm_list=None):
    ''' Load set of bitmask from the input list. This function needs to change
    while the code advances/matures
    '''
    df_fnm = pd.read_table(fnm_list, names=['bpm'])
    for ind, f in df_fnm.iterrows():
        x, hdr = open_fits(f['bpm'])

if __name__ == '__main__':
    t_gral = ''
    t_epi = ''
    argu = argparse.ArgumentParser(description=t_gral, epilog=t_epi)
    # input table of definitions
    h0 = 'Filename for the old set of definitions for the bitmask. Format: 2'
    h0 += ' columns, with bit name in the first column, and'
    h0 += ' bit integer in the second'
    argu.add_argument('--ini', '-i', help=h0, metavar='filename')
    h1 = 'Filename for the new set of definitions for the bitmask. Format:'
    h1 += ' 2 columns with the bit name in the first column, and'
    h1 += ' bit integer in the second'
    argu.add_argument('--end', '-e', help=h1, metavar='filename')
    h2 = 'List of files to be migrated from a set of definitions to another'
    argu.add_argument('--mig', '-m', help=h2, metavar='filename')
    argu = argu.parse_args()

    # Load the tables
    load_bitdef(d1=argu.ini, d2=argu.end)
    # Load BPM FITS file and split in its components
    load_bitmask(fnm_list=argu.mig)
