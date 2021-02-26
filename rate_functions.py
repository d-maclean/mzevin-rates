import numpy as np
import pandas as pd

from scipy.stats import norm, truncnorm
from scipy.special import erf
from scipy.integrate import quad
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as C
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15 as cosmo

import os
import pickle
import sys
import h5py
import itertools
import importlib
import glob
from tqdm import tqdm
import pdb


def sfr_z(z, mdl='2017'):
    """
    Star formation rate as a function in redshift, in units of M_sun / Mpc^3 / yr
    mdl='2017': Default, from Madau & Fragos 2017. Tassos added more X-ray binaries at higher Z, brings rates down
    mdl='2014': From Madau & Dickenson 2014, used in Belczynski et al. 2016
        """
    if mdl=='2017':
        return 0.01*(1+z)**(2.6)  / (1+((1+z)/3.2)**6.2)
    if mdl=='2014':
        return 0.015*(1+z)**(2.7) / (1+((1+z)/2.9)**5.6)

def mean_metal_z(z, Zsun=0.017):
    """
    Mean (mass-weighted) metallicity as a function of redshift
    From Madau & Fragos 2017

    Returns [O/H] metallicity (not in Zsun units)
    """
    log_Z_Zsun = 0.153 - 0.074 * z**(1.34)
    return 10**(log_Z_Zsun) * Zsun

def metal_disp_truncnorm(z, sigmaZ, Zlow, Zhigh, Zsun=0.017):
    """
    Gives a weight for each metallicity Z at a redshift of z by assuming
    the metallicities are log-normally distributed about Z

    Metallicities are in standard units ([Fe/H])
    Default dispersion is half a dex

    lowZ and highZ indicate the lower and upper metallicity bounds; values drawn below these
    will be reflected to ensure the distribution is properly normalized

    NOTE: Be careful in calculating the mean of a log-normal distribution correctly!
    """
    log_mean_Z = np.log10(mean_metal_z(z, Zsun)) - (np.log(10)/2)*sigmaZ**2

    a, b = (np.log10(Zlow) - log_mean_Z) / sigmaZ, (np.log10(Zhigh) - log_mean_Z) / sigmaZ
    Z_dist = truncnorm(a, b, loc=log_mean_Z, scale=sigmaZ)

    return Z_dist

def corrected_means_for_truncated_lognormal(sigmaZ, Zlow, Zhigh):
    """
    Function that returns an interpolant to get an adjusted log-normal mean
    such that the resultant truncated normal distribution preserves the mean
    
    The interpolant will take in the log-normal mean that you *want* after truncation, 
    and gives you the mean you should use when constructing your truncated normal distribution
    """
    log_desired_means = np.linspace(-5,1, 1000)   # tunable, eventually range will give bogus values
    means_for_constructing_lognormal = log_desired_means - (np.log(10)/2)*sigmaZ**2

    means_from_truncated_lognormal = []
    for m in means_for_constructing_lognormal:
        a, b = (np.log10(Zlow) - m) / sigmaZ, (np.log10(Zhigh) - m) / sigmaZ
        Z_dist = truncnorm(a, b, loc=m, scale=sigmaZ)
        means_from_truncated_lognormal.append(Z_dist.moment(1))
        
    truncated_mean_to_gaussian_mean = interp1d(means_from_truncated_lognormal, log_desired_means, \
                   bounds_error=False, fill_value=(np.min(log_desired_means), np.max(log_desired_means)))

    return truncated_mean_to_gaussian_mean

def metal_disp_truncnorm_corrected(z, mean_transformation_interp, sigmaZ, Zlow, Zhigh, Zsun=0.017):
    """
    Gives the probability density function for the metallicity distribution at a given redshift
    using a 'corrected' mean to reproduce the mean of the Z(z) relation

    NOTE: Be careful in calculating the mean of a log-normal distribution correctly!
    """
    log_desired_mean_Z = np.log10(mean_metal_z(z))
    corrected_mean = mean_transformation_interp(log_desired_mean_Z)
    
    corrected_mean_for_truncated_lognormal = corrected_mean - (np.log(10)/2)*sigmaZ**2
    
    a, b = (np.log10(Zlow) - corrected_mean_for_truncated_lognormal) / sigmaZ, (np.log10(Zhigh) - corrected_mean_for_truncated_lognormal) / sigmaZ
    Z_dist = truncnorm(a, b, loc=corrected_mean_for_truncated_lognormal, scale=sigmaZ)
    
    return Z_dist


def fmerge_at_z(model, zbin_low, zbin_high, zmerge_max, Zlow, Zhigh, sigmaZ, Zsun=0.017, cbc_type=None, cosmic=False, corrected_mean_interp=None):
    """
    Calculates the number of mergers of a particular CBC type per unit mass
    N_cor,i = f_bin f_IMF N_merger,i / Mtot,sim

    `model` should be a dict containing multiple models at different metallicities

    Each model will receive a weight based on its metallicity, and the metallicity
    distribution at that particular redshift

    In COSMIC, Mtot,sim already accounts for f_bin and f_IMF
    """

    f_merge = []
    met_weights = []

    for met in sorted(model.keys()):

        mass_stars = model[met]['mass_stars']
        if cosmic:
            bpp = model[met][cbc_type]
            # get delay times for all merging binaries (merger happens at evol_type=6)
            merger = bpp.loc[bpp['evol_type']==6]
        else:
            merger = model[met]['mergers']

        # get the fraction of systems born between [zlow,zhigh] that merger between [0,zmerge_max]
        if zbin_low==0:
            # special treatment for the first bin in the local universe
            tdelay_min = 0
            tdelay_max = cosmo.lookback_time(zbin_high).to(u.Myr).value
        else:
            tdelay_min = (cosmo.lookback_time(zbin_low) - cosmo.lookback_time(zmerge_max)).to(u.Myr).value
            tdelay_max = (cosmo.lookback_time(zbin_high)).to(u.Myr).value

        if cosmic:
            Nmerge_zbin = len(merger.loc[(merger['tphys']<=tdelay_max) & (merger['tphys']>tdelay_min)])
        else:
            Nmerge_zbin = len(merger.loc[(merger['t_delay']<=tdelay_max) & (merger['t_delay']>tdelay_min)])


        # get the number of mergers per unit mass
        f_merge.append(float(Nmerge_zbin) / mass_stars)

        # get redshift in the middle of this log-spaced interval
        midz = 10**(np.log10(zbin_low) + (np.log10(zbin_high)-np.log10(zbin_low))/2.0)

        # append the relative weight of this metallicity at this particular redshift
        if corrected_mean_interp is not None:
            dispersion = metal_disp_truncnorm_corrected(midz, corrected_mean_interp, sigmaZ, Zlow, Zhigh, Zsun=Zsun)
        else:
            dispersion = metal_disp_truncnorm(midz, sigmaZ, Zlow, Zhigh, Zsun=Zsun)
        met_weights.append(dispersion.pdf(np.log10(met)))

    # normalize metallicity weights so that they sum to unity
    met_weights = np.asarray(met_weights)/np.sum(met_weights)

    # return weighted sum of f_merge, units of Msun**-1
    return np.sum(np.asarray(f_merge)*met_weights)*u.Msun**(-1)


def local_rate(model, zgrid_min, zgrid_max, zmerge_max, Nzbins, Zlow, Zhigh, sigmaZ, Zsun=0.017, cbc_type=None, cosmic=False, met_disp_method='truncnorm'):
    """
    Calculates the local merger rate, i.e. mergers that occur between z=0 and zmerge_max
    """
    if zgrid_min==0:
        # account for log-spaced bins
        zgrid_min = 1e-3
    zbins = np.logspace(np.log10(zgrid_min), np.log10(zgrid_max), Nzbins+1)
    zbin_contribution = []

    # depending on the metallicity dispersion method, pre-calculate the interpolant to convert
    # from the log-normal mean to the truncated log-normal mean
    if met_disp_method=='truncnorm':
        corrected_mean_interp = None
    elif met_disp_method=='corrected_truncnorm':
        corrected_mean_interp = corrected_means_for_truncated_lognormal(sigmaZ, Zlow, Zhigh)
    else:
        raise NameError("The metallicity dispersion method you provided ({}) is not defined!".format(met_disp_method))

    # work down from highest zbin
    for zbin_low, zbin_high in tqdm(zip(zbins[::-1][1:], zbins[::-1][:-1]), total=len(zbins)-1):
        # get local mergers per unit mass
        floc = fmerge_at_z(model, zbin_low, zbin_high, zmerge_max, Zlow, Zhigh, sigmaZ, Zsun, cbc_type, cosmic, corrected_mean_interp=corrected_mean_interp)
        # get redshift at middle of the log-spaced zbin
        midz = 10**(np.log10(zbin_low) + (np.log10(zbin_high)-np.log10(zbin_low))/2.0)
        # get SFR at this redshift
        sfr = sfr_z(midz) * u.M_sun * u.Mpc**(-3) * u.yr**(-1)
        # cosmological factor
        E_z = (cosmo._Onu0*(1+midz)**4 + cosmo._Om0*(1+midz)**3 + cosmo._Ok0*(1+midz)**2 + cosmo._Ode0)**(1./2)
        # add the contribution from this zbin to the sum
        zbin_contribution.append(((sfr*floc / ((1+midz) * E_z)) * (zbin_high-zbin_low)).value)

    # reintroduce units to the list
    zbin_contribution = np.asarray(zbin_contribution) * u.Mpc**-3 * u.yr**-1
    # get the total contribution for all zbins and local rate
    zbin_summation = np.sum(zbin_contribution).to(u.Gpc**-3 * u.yr**-1)
    R_local = ((1.0/(cosmo._H0*cosmo.lookback_time(zmerge_max))) * zbin_summation).to(u.Gpc**-3  * u.yr**-1)

    # get the midpoints of the bins to return
    midz = 10**(np.log10(zbins[:-1]) + (np.log10(zbins[1:])-np.log10(zbins[:-1]))/2.0)

    return R_local, zbin_contribution, midz
