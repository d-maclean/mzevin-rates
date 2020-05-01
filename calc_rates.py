import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as C

import os
import pickle
import sys
import h5py
import itertools
import importlib
import argparse
import glob
from tqdm import tqdm
from IPython.core.debugger import set_trace

import rate_functions

# set free parameters in the model
zmin = 0 #lowest redshift we are considering
zmax = 15 #highest redshift we are considering
local_z = 0.1 #max redshift of mergers in the "local" universe
N_zbins = 1000

# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("-p", "--population", type=str)
args = argp.parse_args()


# read in pop model, separate BBHs, NSBHs, and BNSs
mdl_path = '/projects/b1095/michaelzevin/ligo/gw190814/populations/'+args.population
mets = os.listdir(mdl_path)

# process different CBC populations
model = {}
for met in tqdm(mets):
    cosmic_files = os.listdir(os.path.join(mdl_path,met))
    dat_file = [item for item in cosmic_files if (item.startswith('dat')) and (item.endswith('.h5'))][0]
    model[met] = {}
    bpp = pd.read_hdf(os.path.join(mdl_path,met,dat_file), key='bpp')

    bbh_idxs = bpp.loc[(bpp['kstar_1']==14) & (bpp['kstar_2']==14)].index.unique()
    model[met]['bbh'] = bpp.loc[bbh_idxs]
    nsbh_idxs = bpp.loc[((bpp['kstar_1']==13) & (bpp['kstar_2']==14)) | ((bpp['kstar_1']==14) & (bpp['kstar_2']==13))].index.unique()
    model[met]['nsbh'] = bpp.loc[nsbh_idxs]
    bns_idxs = bpp.loc[(bpp['kstar_1']==13) & (bpp['kstar_2']==13)].index.unique()
    model[met]['bns'] = bpp.loc[bns_idxs]

    model[met]['mass_stars'] = float(pd.read_hdf(os.path.join(mdl_path,met,dat_file), key='mass_stars').iloc[-1])


# Calculate rates
for cbc in ['bbh','nsbh','bns']:
    R,_,_ = rate_functions.local_rate(model, zmin, zmax, cbc_type=cbc, zmerge_min=0, zmerge_max=local_z, N_zbins=N_zbins)
    print("{} rate: {} Gpc^-3 yr^-1".format(cbc,np.round(R.value,2)))
