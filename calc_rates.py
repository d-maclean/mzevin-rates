import numpy as np
import pandas as pd

import os
import pdb
import argparse
from tqdm import tqdm

import rate_functions
import filters

# set free parameters in the model

# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("-p", "--population", type=str, help="Path to the population you wish to calculate rates for. If --cosmic is set, it will assume the runs are saved in the standard COSMIC output with subdirectories for each metallicity run. Otherwise, will expect an hdf file with key 'model' that is a dataframe which includes 't_delay', 'met', and 'mass_per_Z'.")

argp.add_argument("--cosmic", action="store_true", help="Specifies whether to expect COSMIC dat files. Default=False.")
argp.add_argument("--pessimistic", action="store_true", help="Specifies whether to filter bpp arrays using the 'pessimistic' CE scenario. Default=False.")
argp.add_argument("--filters", nargs="+", default=['bbh','nsbh','bns'], help="Specify filtering scheme(s) to get rates from a specified subset of the population. Filters are coded up in the 'filters.py' function.")

argp.add_argument("--zmin", type=float, default=0, help="Minimum redshift in redshift grid. Default=0.")
argp.add_argument("--zmax", type=float, default=20, help="Maximum redshift in redshift grid. Default=20.")
argp.add_argument("--localz", type=float, default=0.1, help="Max redshift for what we consider 'local' mergers. Default=0.1.")
argp.add_argument("--Nzbins", type=int, default=1000, help="Number of redshift bins to use in calculation. Default=1000.")

argp.add_argument("--Zsun", type=float, default=0.017, help="Sets the assumed Solar metallicity. Note that the relations from Madau & Fragos 2017 assume a Solar metallicity of Zsun=0.017. Default=0.017.")
argp.add_argument("--Zlow", type=float, default=0.0001, help="Sets the lower bound of the metallicity in absolute unites. Default=0.0001.")
argp.add_argument("--Zhigh", type=float, default=0.03, help="Sets the upper bound of the metallicity in absolute units. Default=0.03.")
argp.add_argument("--sigmaZ", type=float, default=0.5, help="Sets the metallicity dispersion for the mean metallicity relation Z(z). Default=0.5.")
argp.add_argument("--sigmaZ-method", type=str, default="truncnorm", help="Sets the method for calculating the metallicity dispersion. 'truncnorm' will use the standard truncated normal at the mean metallicity Z(z). 'corrected_truncnorm' will adjust the mean provided when defining the probability density such that it reproduces the correct mean metallicity Z(z) when the truncated normal probability density is constructed. 'illustris' will use the SFR(Z,z) from the Illustris-TNG simulation, and will supersede whatever is given in the 'sigmaZ' argument.  Default='truncnorm'.")

argp.add_argument("--verbose", action="store_true", help="Prints extra info. Default=False.")
args = argp.parse_args()

# change working directory to where this code is
abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
os.chdir(dirpath)

# Get pertinent metallicities
Zsun = args.Zsun
Zlow = args.Zlow
Zhigh = args.Zhigh

# read in pop model, save the metallicities that are specified.
mdl_path = args.population
model = {}

if args.verbose:
    print("Reading population models...\n")
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
        if args.verbose:
            print("Calculating rate for {} class...".format(cbc))
        R,_,_ = rate_functions.local_rate(model, \
                zgrid_min=args.zmin, zgrid_max=args.zmax, zmerge_max=args.localz, Nzbins=args.Nzbins, \
                Zlow=Zlow, Zhigh=Zhigh, sigmaZ=args.sigmaZ, met_disp_method=args.sigmaZ_method, \
                Zsun=Zsun, cbc_type=cbc, cosmic=True)
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
    if args.verbose:
        print("Calculating rates...")
    R,_,_ = rate_functions.local_rate(model, \
            zgrid_min=args.zmin, zgrid_max=args.zmax, zmerge_max=args.localz, Nzbins=args.Nzbins, \
            Zlow=Zlow, Zhigh=Zhigh, sigmaZ=args.sigmaZ, met_disp_method=args.sigmaZ_method, \
            Zsun=Zsun, cosmic=False)
    print("rate: {:0.2E} Gpc^-3 yr^-1".format(R.value))

