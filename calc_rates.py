import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as C

import os
import pdb
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
local_z = 0.01 #max redshift of mergers in the "local" universe
N_zbins = 10000

# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("-p", "--population", type=str, help="Path to the population you wish to calculate rates for. If --cosmic is set, it will assume the runs are saved in the standard COSMIC output with subdirectories for each metallicity run. Otherwise, will expect an hdf file with key 'model' that is a dataframe which includes 't_delay', 'met', and 'mass_per_Z'.")
argp.add_argument("--cosmic", action="store_true", help='Specifies whether to expect COSMIC dat files. Default=False')
argp.add_argument("--pessimistic", action="store_true", help="Specifies whether to filter bpp arrays using the 'pessimistic' CE scenario. Default=False")
args = argp.parse_args()

# read in pop model, save the metallicities that are specified. 
mdl_path = args.population
model = {}

if args.cosmic:
    mets = os.listdir(mdl_path)
    # process different CBC populations
    for met in tqdm(mets):
        cosmic_files = os.listdir(os.path.join(mdl_path,met))
        dat_file = [item for item in cosmic_files if (item.startswith('dat')) and (item.endswith('.h5'))][0]
        model[float(met)] = {}
        bpp = pd.read_hdf(os.path.join(mdl_path,met,dat_file), key='bpp')
        if args.pessimistic:
            bpp = rate_functions.filter_optimistic_bpp(bpp)

        bbh_idxs = bpp.loc[(bpp['kstar_1']==14) & (bpp['kstar_2']==14)].index.unique()
        model[float(met)]['bbh'] = bpp.loc[bbh_idxs]
        nsbh_idxs = bpp.loc[((bpp['kstar_1']==13) & (bpp['kstar_2']==14)) | ((bpp['kstar_1']==14) & (bpp['kstar_2']==13))].index.unique()
        model[float(met)]['nsbh'] = bpp.loc[nsbh_idxs]
        bns_idxs = bpp.loc[(bpp['kstar_1']==13) & (bpp['kstar_2']==13)].index.unique()
        model[float(met)]['bns'] = bpp.loc[bns_idxs]

        model[float(met)]['mass_stars'] = float(pd.read_hdf(os.path.join(mdl_path,met,dat_file), key='mass_stars').iloc[-1])

    #  Calculate rates
    for cbc in ['bbh','nsbh','bns']:
        R,_,_ = rate_functions.local_rate(model, zmin, zmax, cosmic=True, cbc_type=cbc, zmerge_min=0, zmerge_max=local_z, N_zbins=N_zbins)
        print("{} rate: {} Gpc^-3 yr^-1".format(cbc,np.round(R.value,2)))

else:
    df = pd.read_hdf(mdl_path, key='model')
    for met in df['met'].unique():
        model[met] = {}
        df_tmp = df.loc[df['met']==met]

        mass_stars = df_tmp['mass_per_Z'].unique()
        assert len(mass_stars) == 1
        model[met]['mass_stars'] = float(mass_stars)

        model[met]['mergers'] = df_tmp

    # Calculate rates
    R,_,_ = rate_functions.local_rate(model, zmin, zmax, cosmic=False, cbc_type=None, zmerge_min=0, zmerge_max=local_z, N_zbins=N_zbins)
    print("rate: {} Gpc^-3 yr^-1".format(np.round(R.value,2)))
    
