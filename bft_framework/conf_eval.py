import math
import itertools
import yaml
import numpy as np
from bel_ass import ass_bel
from opi_gen import gen_opi_pos_y
from opi_gen import gen_opi_vel_x
from opi_gen import read_data
from functools import reduce

"""Redistribution of subunions' masses to corresponding singletons"""
def bel_mass_redistr(bel_subu):
    k_subu = []
    bel_redistr = {}
    for k in bel_subu.keys():
        if len(str(k)) == 2:
            if str(k)[0] in bel_redistr.keys():
                bel_redistr[str(k)[0]] += bel_subu[k]/2
            else:
                bel_redistr[str(k)[0]] = bel_subu[k]/2
            if str(k)[1] in bel_redistr.keys():
                bel_redistr[str(k)[1]] += bel_subu[k]/2
            else:
                bel_redistr[str(k)[1]] = bel_subu[k]/2
            k_subu.append(k)
    for k in bel_redistr.keys():
        if k in bel_subu.keys():
            bel_subu[k] += bel_redistr[k]
        else:
            bel_subu[k] = bel_redistr[k]
    for k in k_subu:
        bel_subu.pop(k)
    return bel_subu

"""Conflict detection between two opinions"""
def conf_dect(bel_s1, bel_s2):
    bel_s1_c = bel_s1.copy()
    bel_s2_c = bel_s2.copy()
    for k in bel_s1_c.keys():
        if len(str(k)) == 3:
            unc_str = k

    unc_val_s1 = bel_s1_c.pop(unc_str)
    sorted(bel_s1_c.keys(), key=lambda x:x.lower())
    bel_s1_c_dis_val = list(bel_s1_c.values())
    bel_s1_c_dis_val = bel_s1_c_dis_val/np.linalg.norm(bel_s1_c_dis_val, ord=1)

    unc_val_s2 = bel_s2_c.pop(unc_str)
    sorted(bel_s2_c.keys(), key=lambda x:x.lower())
    bel_s2_c_dis_val = list(bel_s2_c.values())
    bel_s2_c_dis_val = bel_s2_c_dis_val/np.linalg.norm(bel_s2_c_dis_val, ord=1)

    bel_dis_diff = [s1 - s2 for s1, s2 in zip(bel_s1_c_dis_val, bel_s2_c_dis_val)]
    doc = 0.5*np.linalg.norm(bel_dis_diff, ord=1)*math.sqrt((1 - unc_val_s1)*(1 - unc_val_s2))

    return doc

"""Degree of conflict among all opinions at the same time instant"""
def deg_conf(opi_t):
    opi_sing_t = []
    for opi in opi_t:
        opi_sing = bel_mass_redistr(opi)
        opi_sing_t.append(opi_sing)
    num_opi = 0
    deg_conf_t = []
    for o_s_i in opi_sing_t[0:]:
        for o_s_j in opi_sing_t[opi_sing_t.index(o_s_i)+1:]:
            num_opi += 1
            deg_conf_t.append(conf_dect(o_s_i, o_s_j))
    doc = reduce(lambda x, y: x*y, deg_conf_t)**(1/num_opi)
    red_t = []
    for k in deg_conf_t:
        red_t.append(1 - k)
    red = reduce(lambda x, y: x*y, red_t)**(1/num_opi)
    return red

#bel = {'n': 0.1, 'e': 0.1, 's': 0.6, 'sen': 0.1}
#print(bel_mass_redistr(bel))
#bel_s1 = {'n': 0.2, 'e': 0.3, 's': 0.4, 'sen': 0.1}
#bel_s2 = {'n': 0.1, 'e': 0.1, 's': 0.6, 'sen': 0.2}
#bel = [bel_s1, bel_s2]
#print(deg_conf(bel))
#print(conf_dect(bel_s1, bel_s2))#