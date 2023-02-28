import math
import itertools
import yaml
import numpy as np
from bel_ass import ass_bel
from opi_gen import gen_opi_pos_y
from opi_gen import gen_opi_vel_x
from opi_gen import read_data
from functools import reduce

"""Move part of mass to uncertainty based on degree of conflict"""
def conf_hand(opi_fused, red):
    for k in opi_fused.keys():
        if len(str(k)) == 3:
            unc_str = k
    #print(opi_fused)
    opi_fused_distr = opi_fused.copy()
    #print(opi_fused_distr)
    unc = opi_fused_distr.pop(unc_str)
    unc_add = 0
    for k in opi_fused.keys():
        if len(str(k)) < 3:
            unc_add += opi_fused[k]*(1 - red)
            #unc_add += opi_fused[k]*red
            opi_fused[k] = float(opi_fused[k]*red)
            #print(unc_add)
    opi_fused[unc_str] += float(unc_add)
    return opi_fused

#opi_fused = {'s':0.2,'n':0.1,'e':0.3,'sen':0.4}
#red = 0.2
#print(conf_hand(opi_fused, red))