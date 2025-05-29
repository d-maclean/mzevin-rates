import numpy as np

from scipy.stats import norm, truncnorm
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value

import h5py
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


def fmerge_at_z(model, zbin_low, zbin_high, zmerge_max, Zlow, Zhigh, sigmaZ, met_disp_method, Zsun=0.017, cbc_type=None, cosmic=False, corrected_mean_interp=None, extra_data=None):
    """
    Calculates the number of mergers of a particular CBC type per unit mass
    N_cor,i = f_bin f_IMF N_merger,i / Mtot,sim

    `model` should be a dict containing multiple models at different metallicities

    Each model will receive a weight based on its metallicity, and the metallicity
    distribution at that particular redshift

    In COSMIC, Mtot,sim already accounts for f_bin and f_IMF
    """

    f_merge = []
    met_cdfs = []
    extra_model_data: list[tuple] = [] # list to hold birth time of each system merging in the bin & the metal weights
    zform_lists: list = []
    
    tvals_for_interp = np.logspace(np.log10(200), np.log10(13700), 500) * u.Myr
    zvals_for_interp = z_at_value(cosmo.age, tvals_for_interp)

    # get the minimum and maximum lookback times for this redshift
    if zbin_low==0:
        # special treatment for the first bin in the local universe
        tdelay_min = 0
        tdelay_max = cosmo.lookback_time(zbin_high).to(u.Myr).value
    else:
        tdelay_min = (cosmo.lookback_time(zbin_low) - cosmo.lookback_time(zmerge_max)).to(u.Myr).value
        tdelay_max = (cosmo.lookback_time(zbin_high)).to(u.Myr).value

    
    # redshift in the middle of this log-spaced interval
    midz = 10**(np.log10(zbin_low) + (np.log10(zbin_high)-np.log10(zbin_low))/2.0)
    midt = cosmo.age(midz).to(u.Myr)

    # relavent probability density function at this redshift
    if met_disp_method=='truncnorm':
        Z_dist = metal_disp_truncnorm(midz, sigmaZ, Zlow, Zhigh, Zsun=Zsun)
    elif met_disp_method=='corrected_truncnorm':
        Z_dist = metal_disp_truncnorm_corrected(midz, corrected_mean_interp, sigmaZ, Zlow, Zhigh, Zsun=Zsun)
    elif met_disp_method=='illustris':
        (redz, Z, M) = extra_data
        # only use data within the metallicity bounds
        valid_met_idxs = np.where((Z >= Zlow) & (Z <= Zhigh))[0]
        # get the index of the correct redshift in the data
        redz_idx = np.where(redz <= midz)[0][0]
        # take values of the data at this redshift between our metallicity bounds
        Z_dist = M[redz_idx, valid_met_idxs]
        if np.sum(Z_dist)==0:
            # no star formation in range of metallicities at this redshift
            Z_dist_cdf_interp = None
        else:
            Z_dist_cdf = np.cumsum(Z_dist / Z_dist.sum())
            Z_dist_cdf_interp = interp1d(np.log10(Z[valid_met_idxs]), Z_dist_cdf, fill_value="extrapolate")

    for met in sorted(model.keys()):
        mass_stars = model[met]['mass_stars']
        if cosmic:
            bpp = model[met][cbc_type]
            # get delay times for all merging binaries (merger happens at evol_type=6)
            merger = bpp.loc[bpp['evol_type']==6]
        else:
            merger = model[met]['mergers']

        # get the fraction of systems born between [zlow,zhigh] that merger between [0,zmerge_max]
        if cosmic:
            merger_df = merger.loc[(merger['tphys']<=tdelay_max) & (merger['tphys']>tdelay_min)]
            Nmerge_zbin = len(merger_df)
        else:
            Nmerge_zbin = len(merger.loc[(merger['t_delay']<=tdelay_max) & (merger['t_delay']>tdelay_min)])

        # get birth times
        t_birth_for_met = midt.value - (merger_df.tphys * u.Myr)
        z_birth_for_met= np.interp(t_birth_for_met, tvals_for_interp, zvals_for_interp)
        zform_lists.append(z_birth_for_met)
        
        # get the number of mergers per unit mass
        f_merge.append(float(Nmerge_zbin) / mass_stars)

        if met_disp_method=='illustris':
            if Z_dist_cdf_interp is not None:
                met_cdfs.append(Z_dist_cdf_interp(np.log10(met)))
            else:
                met_cdfs = None
        else:
            met_cdfs.append(Z_dist.cdf(np.log10(met)))

    # get the weight of each metallicity model at this redshift by taking the midpoints
    # between all cdf values, with the lowest and highest metallicities getting the weight
    # up to Zlow and Zhigh, respectively
        
    if met_cdfs is not None:
        met_cdfs = np.asarray(met_cdfs)
        cdf_midpts = met_cdfs[:-1] + (met_cdfs[1:]-met_cdfs[:-1])/2
        met_cdf_ranges = np.append(np.append(0, cdf_midpts), 1)
        met_weights = met_cdf_ranges[1:] - met_cdf_ranges[:-1]


        # metallicity weights should sum to unity (to numerical precision)
        assert ((np.sum(met_weights) > 0.9999) and (np.sum(met_weights) < 1.0001)), "The weights for the metallicities at redshift z={:0.2f} do not sum to unity (they sum to {:0.5f})!".format(midz, np.sum(met_weights))
    else:
        met_weights = 0.0

    for i in range(len(zform_lists)): # collect the extra data
        extra_model_data.append((zform_lists[i], met_weights[i]))
    # return weighted sum of f_merge, units of Msun**-1
    return (np.sum(np.asarray(f_merge)*met_weights)*u.Msun**(-1), extra_model_data)


def local_rate(model, zgrid_min, zgrid_max, zmerge_max, Nzbins, Zlow, Zhigh, sigmaZ, met_disp_method, Zsun=0.017, cbc_type=None, cosmic=False):
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
        illustris_data = None
    elif met_disp_method=='corrected_truncnorm':
        corrected_mean_interp = corrected_means_for_truncated_lognormal(sigmaZ, Zlow, Zhigh)
        illustris_data = None
    # get stuff that we need from the Illustris data
    elif met_disp_method=='illustris':
        corrected_mean_interp = None
        with h5py.File('./data/TNG100_L75n1820TNG__x-t-log_y-Z-log.hdf5', 'r') as f:
            time_bins = f['xedges'][:]
            met_bins = f['yedges'][1:-1]   # kill the top and bottom metallicity bins, we won't need them and they go to inf
            Mform = f['mass'][:,1:-1]
        # we only care about stuff before today
        tmax = cosmo.age(0).to(u.yr).value
        young_enough = np.argwhere(time_bins <= tmax)
        time_bins = np.squeeze(time_bins[young_enough])
        Mform = np.squeeze(Mform[young_enough, :])
        # add tH to beginning of time_bins, assume this bin just spans until today
        time_bins = np.append(time_bins, tmax)
        # get times and metallicity bin centers
        times = 10**(np.log10(time_bins[:-1])+((np.log10(time_bins[1:])-np.log10(time_bins[:-1]))/2))
        mets = 10**(np.log10(met_bins[:-1])+((np.log10(met_bins[1:])-np.log10(met_bins[:-1]))/2))
        # calculate redshifts for midpoints and bin edges
        redshifts = []
        for t in tqdm(times):
            redshifts.append(z_at_value(cosmo.age, t*u.yr))
        redshifts = np.asarray(redshifts)
        redshift_bins = []
        for t in tqdm(time_bins[:-1]):
            redshift_bins.append(z_at_value(cosmo.age, t*u.yr))
        redshift_bins.append(0)   # special treatment for most local bin
        redshift_bins = np.asarray(redshift_bins)
        # time duration in each bin
        dt = time_bins[1:] - time_bins[:-1]
        # get interpolant for the SFR as as function of time for considered metallicities, convert 100 Mpc^3 box to Msun Mpc^-3 yr^-1
        valid_met_idxs = np.where((mets >= Zlow) & (mets <= Zhigh))[0]
        sfr_pts = np.sum(Mform[:,valid_met_idxs], axis=1) / dt / 100**3
        sfr_interp = interp1d(redshifts, sfr_pts)
        # we need to pass some of this Illustris data to fmerge_at_z function
        illustris_data = (redshifts, mets, Mform)
    else:
        raise NameError("The metallicity dispersion method you provided ({}) is not defined!".format(met_disp_method))


    # TRY THIS AGAIN
    floc_at_z = []
    model_data_at_z = []
    E_z = []

    # get the midpoints of the bins to return
    midz = 10**(np.log10(zbins[:-1]) + (np.log10(zbins[1:])-np.log10(zbins[:-1]))/2.0)

    # work down from highest zbin
    for zbin_low, zbin_high in tqdm(zip(zbins[::-1][1:], zbins[::-1][:-1]), total=len(zbins)-1):
        # get local mergers per unit mass
        floc, extra_data = fmerge_at_z(model, zbin_low, zbin_high, zmerge_max, Zlow, Zhigh, \
                    sigmaZ, met_disp_method, Zsun, cbc_type, cosmic, \
                    corrected_mean_interp=corrected_mean_interp, extra_data=illustris_data)
        # get redshift at middle of the log-spaced zbin
        midz = 10**(np.log10(zbin_low) + (np.log10(zbin_high)-np.log10(zbin_low))/2.0)
        # get SFR at this redshift
        if met_disp_method=='illustris':
            sfr = sfr_interp(midz) * u.M_sun * u.Mpc**(-3) * u.yr**(-1)
        else:
            model_data_at_z.append(extra_data)
            #sfr = sfr_z(midz) * u.M_sun * u.Mpc**(-3) * u.yr**(-1)
            # alternative SFR method - multiply 
            
        # cosmological factor
        E_z.append(\
            (cosmo._Onu0*(1+midz)**4 + cosmo._Om0*(1+midz)**3 + cosmo._Ok0*(1+midz)**2 + cosmo._Ode0)**(1./2)
            )
        # add the contribution from this zbin to the sum
        #zbin_contribution.append(((sfr*floc / ((1+midz) * E_z)) * (zbin_high-zbin_low)).value)

    weights_per_metal = []
    for j in range(len(model)):
        ack = [x[1][j] for x in model_data_at_z]
        weights_per_metal.append(ack)

    i = 0
    for zbin_low, zbin_high in tqdm(zip(zbins[::-1][1:], zbins[::-1][:-1]), total=len(zbins)-1):
        
        midz_i = midz[i]
        extra_data = model_data_at_z[i]
        sfr_per_system = np.zeros(0)
        
        sfr = np.zeros(shape=0)
        for j in range(len(extra_data)):
            z_form = extra_data[0]
            Zweight = get_met_weights_as_z(z_form, midz, weights_per_metal[j])
            helpme = sfr_z(z_form, mdl='2017')
            
            sfr_per_system_at_this_Z = helpme * Zweight
            sfr_per_system = np.concat([sfr_per_system, sfr_per_system_at_this_Z])
            
        zbin_contribution.append(\
            ((sfr_per_system*floc / ((1+midz_i) * E_z)) * (zbin_high-zbin_low)).value)
        
        i += 1
        
    # reintroduce units to the list
    zbin_contribution = np.asarray(zbin_contribution) * u.Mpc**-3 * u.yr**-1
    # get the total contribution for all zbins and local rate
    zbin_summation = np.sum(zbin_contribution).to(u.Gpc**-3 * u.yr**-1)
    R_local = ((1.0/(cosmo._H0*cosmo.lookback_time(zmerge_max))) * zbin_summation).to(u.Gpc**-3  * u.yr**-1)


    return R_local, zbin_contribution, midz

def get_met_weights_as_z(z, zrange, weights):
    '''
    As the end of days, the beast will rise from the sea. Upon its shoulders will be seven heads,
    and upon its seven heads will rest ten horns and ten crowns, and it will bear the name of blasphemy.
    ### Parameters:
    z - array[float]
    zrange - array[float]
    '''
    z_idx = np.argmin(np.abs(z - zrange))
    return weights[z_idx]
