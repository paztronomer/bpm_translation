#!/usr/bin/env python
''' Code for translating Bad Pixel Masks from a set of definitions to another
Function-driven instead of Class, for simpler paralellisnm
NOTE: in this code is assumed that the amount of masked pixels is 
roughly the same between definitions 
'''

import os
import sys
import time
import logging
import argparse
import copy
import uuid
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
        BITDEF_END[j] = np.uint(BITDEF_END[j])
    return True 

def diff_bitdef():
    ''' Compare both dictionaries bit definitions. Also compare the bit value
    for the same bit definition, on both schemas. Leave it as general as 
    possible
    Returns
    - 2 lists, one containing the bit definition names not being common for 
    both the definitions, and the other containing the bit definitions for 
    the bits having different name in both sets
    '''
    # Set intersection and differences. Beware of the use of set() and 
    # its capabilities/limitations
    inter = set(BITDEF_INI.keys()).intersection(set(BITDEF_END.keys()))
    diff = set(BITDEF_INI.keys()).symmetric_difference(set(BITDEF_END.keys()))
    # Compare in terms of keys, for both sets.  
    for k_diff in diff:
        # Do not discard the possibility of both sets having unique bit
        # definitions
        if (k_diff in set(BITDEF_INI.keys())):
            t_i0 = 'Different BITS: shown in INITIAL but not in FINAL bitdef'
            t_i0 += ' {0}:{1}'.format(k_diff, BITDEF_INI[k_diff])
            logging.info(t_i0)
        if (k_diff in set(BITDEF_END.keys())):
            t_i1 = 'Different BITS: shown in FINAL but not in INITIAL bitdef'
            t_i1 += ' {0}:{1}'.format(k_diff, BITDEF_END[k_diff])
            logging.info(t_i1)
    # Compare in terms of values, for same key across both sets.
    diff_keyvalues = []
    for k_inter in inter:
        if (BITDEF_INI[k_inter] != BITDEF_END[k_inter]):
            diff_keyvalues.append(k_iter)
            t_i2 = 'Different definitions: '
            t_i2 += '{0}:{1} in INITIAL'.format(k_inter, BITDEF_INI[k_inter])
            t_i2 += ' {0}:{1} in FINAL'.format(k_inter, BITDEF_END[k_inter])
            logging.info(t_i2)
    return list(diff), diff_keyvalues

def bit_count(int_type):
    ''' Function to count the amount of bits composing a number, that is the
    number of base 2 components on which it can be separated, and then
    reconstructed by simply sum them. Idea from Wiki Python.
    Brian Kernighan's way for counting bits. Thismethod was in fact
    discovered by Wegner and then Lehmer in the 60s
    This method counts bit-wise. Each iteration is not simply a step of 1.
    Example: iter1: (2066, 2065), iter2: (2064, 2063), iter3: (2048, 2047)
    Inputs
    - int_type: integer
    Output
    - integer with the number of base-2 numbers needed for the decomposition
    '''
    counter = 0
    while int_type:
        int_type &= int_type - 1
        counter += 1
    return counter

def bit_decompose(int_x):
    ''' Function to decompose a number in base-2 numbers. This is performed by
    two binary operators. Idea from Stackoverflow.
        x << y
    Returns x with the bits shifted to the left by y places (and new bits on
    the right-hand-side are zeros). This is the same as multiplying x by 2**y.
        x & y
    Does a "bitwise and". Each bit of the output is 1 if the corresponding bit
    of x AND of y is 1, otherwise it's 0.
    Inputs
    - int_x: integer
    Returns
    - list of base-2 values from which adding them, the input integer can be
    recovered
    '''
    base2 = []
    i = 1
    while (i <= int_x):
        if (i & int_x):
            base2.append(i)
        i <<= 1
    return base2

def flatten_list(list_2levels):
    ''' Function to flatten a list of lists, generating an output with
    all elements ia a single level. No duplicate drop neither sort are
    performed
    '''
    f = lambda x: [item for sublist in x for item in sublist]
    res = f(list_2levels)
    return res

def split_bitmask_FITS(arr, ccdnum, save_fits=False, outnm=None):
    ''' Return a n-dimensional array were each layer is an individual bitmask,
    then, can be loaded into DS9. This function helps as diagnostic.
    '''
    # I need 3 lists/arrays to perform the splitting:
    # 1) different values composing the array
    # 2) the bits on which the above values can be decomposed
    # 3) the number of unique bits used on the above decomposition
    # First get all the different values the mask has
    diff_val = np.sort(np.unique(arr.ravel()))
    # Decompose each of the unique values in its bits
    decomp2bit = []
    for d_i in diff_val:
        dcomp = bit_decompose(d_i)
        decomp2bit.append(dcomp)
    diff_val = tuple(diff_val)
    decomp2bit = tuple(decomp2bit)
    # Be careful to keep diff_val and decomp2bit with the same element order,
    # because both will be compared
    # Get all the used bits, by the unique components of the flatten list
    # Option for collection of unordered unique elements:
    #   bit_uniq = repr(sorted(set(bit_uniq)))
    bit_uniq = flatten_list(list(decomp2bit))
    bit_uniq = np.sort(np.unique(bit_uniq))
    # Safe checks
    #
    # Check there are no missing definitions!
    #
    # Go through the unique bits, and get the positions of the values
    # containig such bit
    bit2val = dict()
    # Over the unique bits
    for b in bit_uniq:
        tmp_b = []
        # Over the bits composing each of the values of the array
        for idx_nb, nb in enumerate(decomp2bit):
            if (b in nb):
                tmp_b.append(diff_val[idx_nb])
        # Fill a dictionary with the values that contains every bit
        bit2val.update({'BIT_{0}'.format(b) : tmp_b})
    # Create a ndimensional matrix were to store the positions having
    # each of the bits. As many layers as unique bits are required to
    # construct all the values
    is1st = True
    for ib in bit_uniq:
        # Zeros or NaN?
        # tmp_arr = np.full(arr.shape, np.nan)
        tmp_arr = np.zeros_like(arr)
        # Where do the initial array contains the values that are 
        # composed by the actual bit?
        for k in bit2val['BIT_{0}'.format(ib)]:
            tmp_arr[np.where(arr == k)] = ib
            # print 'bit: ', ib, ' N: ', len(np.where(arr == k)[0])
        # print '==========', ib, len(np.where(tmp_arr == ib)[0])
        # Add new axis
        tmp_arr = tmp_arr[np.newaxis , : , :]
        if is1st:
            ndimBit = tmp_arr
            is1st = False
        # NOTE: FITS files recognizes depth as the 1st dimesion
        # ndimBit = np.dstack((ndimBit, tmp_arr))
        ndimBit = np.vstack((ndimBit, tmp_arr))
    if save_fits:
        if (outnm is None):
            outnm = str(uuid.uuid4())
            outnm = os.path.joint(outnm, '.fits')
        if os.path.exists(outnm):
            t_w = 'File {0} exists. Will not overwrite'.format(outnm)
            logging.warning(t_w)
        else: 
            fits = fitsio.FITS(outnm, 'rw')
            fits.write(ndimBit)
            fits[-1].write_checksum()
            hlist = [
                {'name' : 'CCDNUM', 'value' : ccdnum, 'comment' : 'CCD number'},
                {'name' : 'COMMENT', 
                 'value' : 'Multilayer bitmask, Francisco Paz-Chinchon'}
                ]
            fits[-1].write_keys(hlist)
            fits.close()
            t_i = 'Multilayer bitmaks saved {0}'.format(outnm)
            logging.info(t_i)
    return True

def change_bitmask(tab_ini='bad_pixels.lst', 
                   tab_end='bad_pixels_20160506.lst'):
    ''' NOTE: this function is so similar to split_bitmask_FITS()
    Do it as general as possible
    '''
    # 1) Load sections corresponding to each bit
    df_end = pd.read_table(tab_ini, comment='#', 
                           names=['ccdnum', 'x0', 'x1', 'y0', 'y1', 'bpmdef'])
    df_ini = pd.read_table(tab_end, comment='#', 
                           names=['ccdnum', 'x0', 'x1', 'y0', 'y1', 'bpmdef']) 
    # 2) Check which definitions are not common
    diff_name, diff_keyval = diff_bitdef()
    print diff_name, diff_keyval
    #
    #
    # Here!
    #
    #

    # Next steps
    # - Check the number of pixels masked for each bit are the same
    # - Change bits without overlap to previous ones
    # - Add layes for new bits
    # - Save modified bits array
    
    
    exit()


def load_bitmask(fnm_list=None, ext=0):
    ''' Load set of bitmask from the input list. This function needs to change
    while the code advances/matures
    '''
    df_fnm = pd.read_table(fnm_list, names=['bpm'])
    for ind, f in df_fnm.iterrows():
        x, hdr = open_fits(f['bpm'])
        if ((len(x) > 1) or (len(hdr) > 1)):
            t_w = 'FITS file {0} has more than 1 extension'.format(f['bpm'])
            logging.warning(t_w)
        x = x[ext]
        hdr = hdr[ext]
        ccdnum = hdr['CCDNUM']
        out_bitLayer = os.path.join(os.getcwd(), 
                                    'bitLayer_' + os.path.basename(f['bpm']))
        # Translate from a definition to another
        # Check at least there is one pixel masked with each bit in the 
        # layers. 
        # For translating need to use the spatial position of the bits that 
        # has no definition on the BPMDEF_INI but yes in BPMDEF_END
        #  
        x_new = change_bitmask()
        
        exit()

        # Split bitmask into its components
        # NOTE: would be helpful to have this function run with the final
        # translated BPM and also with the previous
        split_bitmask_FITS(x_new, ccdnum, save_fits=True, outnm=out_bitLayer)

def get_args():
    ''' Construct the argument parser 
    '''
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
    return argu

if __name__ == '__main__':
    # Argument parser
    argu = get_args()
    # Load the tables, storing info in global variables
    load_bitdef(d1=argu.ini, d2=argu.end)
    # Load BPM FITS file and split in its components
    load_bitmask(fnm_list=argu.mig)
    # 
