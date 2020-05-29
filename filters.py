import numpy as np
import pandas as pd

def pessimistic_CE(bpp, pessimistic_merge=[0,1,2,7,8,10,11,12]):
    CE = bpp.loc[bpp['evol_type']==7]
    # check which one is the donor
    CE_star1donor =  CE.loc[CE['RRLO_1']>=CE['RRLO_2']]
    CE_star2donor =  CE.loc[CE['RRLO_1']<CE['RRLO_2']]
    # Get the ones that would merge
    CE_merge_star1donor = CE_star1donor.loc[CE_star1donor['kstar_1'].isin(pessimistic_merge)].index.unique()
    CE_merge_star2donor = CE_star2donor.loc[CE_star2donor['kstar_2'].isin(pessimistic_merge)].index.unique()
    # drop these
    bpp_cut = bpp.drop(CE_merge_star1donor)
    bpp_cut = bpp.drop(CE_merge_star2donor)
    return bpp_cut

def bbh_filter(bpp):
    bbh_idxs = bpp.loc[(bpp['kstar_1']==14) & (bpp['kstar_2']==14)].index.unique()
    bpp_cut = bpp.loc[bbh_idxs]
    return bpp_cut

def nsbh_filter(bpp):
    nsbh_idxs = bpp.loc[((bpp['kstar_1']==13) & (bpp['kstar_2']==14)) |\
         ((bpp['kstar_1']==14) & (bpp['kstar_2']==13))].index.unique()
    bpp_cut = bpp.loc[nsbh_idxs]
    return bpp_cut

def bns_filter(bpp):
    bns_idxs = bpp.loc[(bpp['kstar_1']==13) & (bpp['kstar_2']==13)].index.unique()
    bpp_cut = bpp.loc[bns_idxs]
    return bpp_cut


def gw190814(bpp):
    q_range = (0.112-0.009, 0.112+0.008)
    Mtot_range = (25.8-0.9, 25.8+1.0)

    cbs = bpp.loc[((bpp['kstar_1']==13) | (bpp['kstar_1']==14)) & \
        ((bpp['kstar_2']==13) | (bpp['kstar_2']==14))].groupby('bin_num').first()

    cbs['q'] = (np.min([np.asarray(cbs['mass_2']),np.asarray(cbs['mass_1'])], axis=0) / \
                    np.max([np.asarray(cbs['mass_2']),np.asarray(cbs['mass_1'])], axis=0))
    cbs['Mtot'] = np.asarray(cbs['mass_1']) + np.asarray(cbs['mass_2'])

    cut_idxs = cbs.loc[((cbs['q']>=q_range[0]) & (cbs['q']<=q_range[1])) & \
                ((cbs['Mtot']>=Mtot_range[0]) & (cbs['Mtot']<=Mtot_range[1]))].index.unique()
    bpp_cut = bpp.loc[cut_idxs]

    return bpp_cut


def gw190814_approx(bpp):
    q_range = (0.06,0.16)
    Mtot_range = (20, 30)

    cbs = bpp.loc[((bpp['kstar_1']==13) | (bpp['kstar_1']==14)) & \
        ((bpp['kstar_2']==13) | (bpp['kstar_2']==14))].groupby('bin_num').first()

    cbs['q'] = (np.min([np.asarray(cbs['mass_2']),np.asarray(cbs['mass_1'])], axis=0) / \
                    np.max([np.asarray(cbs['mass_2']),np.asarray(cbs['mass_1'])], axis=0))
    cbs['Mtot'] = np.asarray(cbs['mass_1']) + np.asarray(cbs['mass_2'])

    cut_idxs = cbs.loc[((cbs['q']>=q_range[0]) & (cbs['q']<=q_range[1])) & \
                ((cbs['Mtot']>=Mtot_range[0]) & (cbs['Mtot']<=Mtot_range[1]))].index.unique()
    bpp_cut = bpp.loc[cut_idxs]

    return bpp_cut


def gw190814_coarse(bpp):
    q_range = (0.00,0.20)
    Mtot_range = (20, 100)

    cbs = bpp.loc[((bpp['kstar_1']==13) | (bpp['kstar_1']==14)) & \
        ((bpp['kstar_2']==13) | (bpp['kstar_2']==14))].groupby('bin_num').first()

    cbs['q'] = (np.min([np.asarray(cbs['mass_2']),np.asarray(cbs['mass_1'])], axis=0) / \
                    np.max([np.asarray(cbs['mass_2']),np.asarray(cbs['mass_1'])], axis=0))
    cbs['Mtot'] = np.asarray(cbs['mass_1']) + np.asarray(cbs['mass_2'])

    cut_idxs = cbs.loc[((cbs['q']>=q_range[0]) & (cbs['q']<=q_range[1])) & \
                ((cbs['Mtot']>=Mtot_range[0]) & (cbs['Mtot']<=Mtot_range[1]))].index.unique()
    bpp_cut = bpp.loc[cut_idxs]

    return bpp_cut

_valid_filters = {'pessimistic_CE': pessimistic_CE, \
                  'bbh': bbh_filter, \
                  'nsbh': nsbh_filter, \
                  'bns': bns_filter, \
                  'gw190814': gw190814, \
                  'gw190814_approx': gw190814_approx, \
                  'gw190814_coarse': gw190814_coarse}
