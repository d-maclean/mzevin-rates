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

import rate_functions
import filters

# set free parameters in the model
zmin = 0 #lowest redshift we are considering
zmax = 15 #highest redshift we are considering
local_z = 0.1 #max redshift of mergers in the "local" universe
N_zbins = 1000

# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("-p", "--population", type=str, help="Path to the population you wish to calculate rates for. If --cosmic is set, it will assume the runs are saved in the standard COSMIC output with subdirectories for each metallicity run. Otherwise, will expect an hdf file with key 'model' that is a dataframe which includes 't_delay', 'met', and 'mass_per_Z'.")
argp.add_argument("--cosmic", action="store_true", help='Specifies whether to expect COSMIC dat files. Default=False')
argp.add_argument("--pessimistic", action="store_true", help="Specifies whether to filter bpp arrays using the 'pessimistic' CE scenario. Default=False")
argp.add_argument("--filters", nargs="+", default=['bbh','nsbh','bns'], help="Specify filtering scheme(s) to get rates from a specified subset of the population. Filters are coded up in the 'filters.py' function.")
<<<<<<< HEAD
argp.add_argument("--sigmaZ", type=float, default=0.5, help="Sets the metallicity dispersion for the mean metallicity relation Z(z). Default=0.5")
=======
argp.add_argument("--Zdisp", default=0.5, help="Sets the dispersion of the log-normal distribution for metallicities. Default=0.5.")
>>>>>>> 0c7eea434af7178256fe4fb38092119276f91757
args = argp.parse_args()

# read in pop model, save the metallicities that are specified.
mdl_path = args.population
model = {}

if args.cosmic:
    met_files = os.listdir(mdl_path)
    # process different CBC populations
    for met_file in tqdm(met_files):
        cosmic_files = os.listdir(os.path.join(mdl_path,met_file))
        dat_file = [item for item in cosmic_files if (item.startswith('dat')) and (item.endswith('.h5'))][0]

        # get metallicity for this COSMIC run
        initC = pd.read_hdf(os.path.join(mdl_path,met_file,dat_file), key='initCond')
        assert len(np.unique(initC['metallicity'])) == 1
        met = float(initC['metallicity'].iloc[0])
        model[met] = {}

        # get total stellar mass sampled
        model[met]['mass_stars'] = float(pd.read_hdf(os.path.join(mdl_path,met_file,dat_file), key='mass_stars').iloc[-1])

        # read in bpp array
        bpp = pd.read_hdf(os.path.join(mdl_path,met_file,dat_file), key='bpp')

        # filter to get pessimistic CE, if specified
        if args.pessimistic:
            bpp = filters.pessimistic_CE(bpp)

        # filters the bpp array for specified populations
        cbc_classes = []
        for filt in args.filters:
            if filt not in filters._valid_filters:
                raise ValueError('The filter you specified ({}) is not defined in the filters function!'.format(filt))
            filter_func  = filters._valid_filters[filt]
            model[met][filt] = filter_func(bpp)
            cbc_classes.append(filt)

    #  Calculate rates
    for cbc in cbc_classes:
<<<<<<< HEAD
        R,_,_ = rate_functions.local_rate(model, zmin, zmax, cosmic=True, cbc_type=cbc, sigmaZ=args.sigmaZ, zmerge_min=0, zmerge_max=local_z, N_zbins=N_zbins)
=======
        R,_,_ = rate_functions.local_rate(model, zmin, zmax, cosmic=True, cbc_type=cbc, zmerge_min=0, zmerge_max=local_z, sigmaZ=args.Zdisp, N_zbins=N_zbins)
>>>>>>> 0c7eea434af7178256fe4fb38092119276f91757
        print("{} rate: {:0.2E} Gpc^-3 yr^-1".format(cbc,R.value))

# do for general population
else:
    df = pd.read_hdf(mdl_path, key='model')
    for met in df['metallicity'].unique():
        model[met] = {}
        df_tmp = df.loc[df['metallicity']==met]

        mass_stars = df_tmp['mass_per_Z'].unique()
        assert len(mass_stars) == 1
        model[met]['mass_stars'] = float(mass_stars)

        model[met]['mergers'] = df_tmp

    # Calculate rates
<<<<<<< HEAD
    R,_,_ = rate_functions.local_rate(model, zmin, zmax, cosmic=False, cbc_type=None, sigmaZ=args.sigmaZ, zmerge_min=0, zmerge_max=local_z, N_zbins=N_zbins)
=======
    R,_,_ = rate_functions.local_rate(model, zmin, zmax, cosmic=False, cbc_type=None, zmerge_min=0, zmerge_max=local_z, sigmaZ=args.Zdisp, N_zbins=N_zbins)
>>>>>>> 0c7eea434af7178256fe4fb38092119276f91757
    print("rate: {:0.2E} Gpc^-3 yr^-1".format(R.value))

