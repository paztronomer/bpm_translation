# bpm_translation
Updates BPMs from a set of definitions to another, defining the equivalence 
based on the spatial location of the polygons. Note the polygons must be 
defined in the input files.

A typical call is
`python bitmask_update.py -i bpmdef_initial.txt -e bpmdef_final.txt -m BPM.txt`

Where _BPM.txt_ is a text file with one full path per line, e.g.,

    /archive_data/desarchive/OPS/cal/bpm/20141020t1030-r1474/p01/D_n20141020t1030_c01_r1474p01_bpm.fits
    /archive_data/desarchive/OPS/cal/bpm/20141020t1030-r1474/p01/D_n20141020t1030_c03_r1474p01_bpm.fits
    ...
    
And bpmdef_initial.txt, bpmdef_final.txt are 2 columns text files where first
column is the bit name, and second column is the bit value, e.g.,

    BPMDEF_FLAT_MIN     1
    BPMDEF_FLAT_MAX     2
    BPMDEF_FLAT_MASK    4
    BPMDEF_BIAS_HOT     8
    BPMDEF_BIAS_WARM   16
    BPMDEF_BIAS_MASK   32
    BPMDEF_BIAS_COL    64
    BPMDEF_EDGE       128
    BPMDEF_CORR       256
    ...
