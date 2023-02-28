# Weighted Belief Fusion Operator
# Author: Xuhui Zhang

import math

"""Iteratively fuse the opinion at each time instant from past to present"""
def fuse_pas_pre(ini_bel, opi_pas_pre, mu_str='mu'):
    bel_dev = {}

    # update one time step each time
    for k in opi_pas_pre.keys():
        # keep the temporal and also the final fused result in cur_bel
        bel_dev[k] = WBF(ini_bel, opi_pas_pre[k], mu_str=mu_str)
        ini_bel = bel_dev[k]
    return bel_dev

"""Implements WBF to update old belief with new fused opinion"""
def WBF(old_bel, new_opi, mu_str='mu'):

    all_pos = old_bel.keys() # all singletons and uncertainty 
    new_bel = {} # store the new fused belief
    
    # WBF formulas
    for k in all_pos:
        if k is mu_str:
            unce = (2 - old_bel[mu_str] - new_opi[mu_str]) * old_bel[mu_str] * new_opi[mu_str]
            unce = unce/ (old_bel[mu_str] + new_opi[mu_str] - 2 * old_bel[mu_str] * new_opi[mu_str])
            new_bel[mu_str] = unce
        else:
            mass = old_bel[k] * (1 - old_bel[mu_str]) * new_opi[mu_str] + new_opi[k] * (1 - new_opi[mu_str]) * old_bel[mu_str]
            mass = mass / (old_bel[mu_str] + new_opi[mu_str] - 2 * old_bel[mu_str] * new_opi[mu_str])
            new_bel[k] = mass

    return new_bel