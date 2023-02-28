# Belief Mass Assignment for Each Opinion
# Author: Xuhui Zhang

import math
import scipy.stats

"""Assign belief mass based the difference between measured value and model-given value"""
def ass_bel(mea_val, mod_val, std_dev):
    diff_val = {}
    for k in mod_val.keys():
      diff_val[k] = mod_val[k] - mea_val ## calculate the difference

    bel_mass = {}
    for k in mod_val.keys():
        #bel_mass[k] = max(scipy.stats.norm(0, std_dev).pdf(diff_val[k]), 1e-18) # assign mass based on normal PDF
        bel_mass[k] = scipy.stats.norm(0, std_dev).pdf(diff_val[k])
    
    sum = math.fsum(list(bel_mass.values()))
    #print(sum)
    for k in bel_mass.keys(): bel_mass[k] = float(bel_mass[k]/sum) # normalize masses

    return bel_mass

#mea_val = 0.1
#mod_val = {'s': -0.3, 'e': 0.2, 'n': 0.7}
#std_dev = 0.1
#bel_dis = ass_bel(mea_val, mod_val, std_dev)
#print(bel_dis)
#mea_val = 0.0
#mod_val = {'e': 12.0, 'sn': 12.0}
#bel_dis = ass_bel(mea_val, mod_val, 0.1)
#print(bel_dis)