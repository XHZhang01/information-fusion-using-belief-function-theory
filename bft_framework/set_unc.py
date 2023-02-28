import math
import itertools
import yaml
import numpy as np
from bel_ass import ass_bel
from opi_gen import gen_opi_pos_y
from opi_gen import gen_opi_vel_x
from opi_gen import read_data

"""Set uncertainty based on perturbation of belief distribution"""
#def set_unc(bel_dis, size_win):
#    bel_dis_pos_x = bel_dis['pos_x']
#    bel_dis_pos_y = bel_dis['pos_y']
#    bel_dis_vel_x = bel_dis['vel_x']
#    bel_dis_vel_y = bel_dis['vel_y']
#    time_stamps = [float(t) for t in list(bel_dis['pos_x'].keys())]
#    diff_pos_x = {}
#    diff_pos_y = {}
#    diff_vel_x = {}
#    diff_vel_y = {}
#    for i in range(1, len(time_stamps) - 1):
#        t_current = time_stamps[i]
#        t_last = time_stamps[i-1]
#        diff_pos_x[t_current] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_pos_x[t_current].values()), list(bel_dis_pos_x[t_last].values()))], ord=1)
#        diff_pos_y[t_current] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_pos_y[t_current].values()), list(bel_dis_pos_y[t_last].values()))], ord=1)
#        diff_vel_x[t_current] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_vel_x[t_current].values()), list(bel_dis_vel_x[t_last].values()))], ord=1)
#        diff_vel_y[t_current] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_vel_y[t_current].values()), list(bel_dis_vel_y[t_last].values()))], ord=1)
#    unc_pos_x = {}
#    unc_pos_y = {}
#    unc_vel_x = {}
#    unc_vel_y = {}
#    for i in range(0, len(time_stamps) - 1):
#        t_current = time_stamps[i]
#        if i < size_win:
#            unc_pos_x[t_current] = 0.9
#            unc_pos_y[t_current] = 0.9
#            unc_vel_x[t_current] = 0.9
#            unc_vel_y[t_current] = 0.9
#        else:
#            unc_pos_x[t_current] = max(0.05, sum([diff_pos_x[time_stamps[t]] for t in range(i - size_win + 1, i)])/2/(size_win-1))
#            unc_pos_y[t_current] = max(0.05, sum([diff_pos_y[time_stamps[t]] for t in range(i - size_win + 1, i)])/2/(size_win-1))
#            unc_vel_x[t_current] = max(0.05, sum([diff_vel_x[time_stamps[t]] for t in range(i - size_win + 1, i)])/2/(size_win-1))
#            unc_vel_y[t_current] = max(0.05, sum([diff_vel_y[time_stamps[t]] for t in range(i - size_win + 1, i)])/2/(size_win-1))
#    opi = {}
#    for i in range(0, len(time_stamps) - 1):
#        t_current = time_stamps[i]
#        bel_dis_pos_x[t_current].update((key, val*(1-unc_pos_x[t_current])) for key, val in bel_dis_pos_x[t_current].items())
#        bel_dis_pos_y[t_current].update((key, val*(1-unc_pos_y[t_current])) for key, val in bel_dis_pos_y[t_current].items())
#        bel_dis_vel_x[t_current].update((key, val*(1-unc_vel_x[t_current])) for key, val in bel_dis_vel_x[t_current].items())
#        bel_dis_vel_y[t_current].update((key, val*(1-unc_vel_y[t_current])) for key, val in bel_dis_vel_y[t_current].items())
#
#        opi[t_current] = [bel_dis_pos_y[t_current], bel_dis_vel_x[t_current]]
#    return opi

"""Set uncertainty based on perturbation of belief distribution"""
def set_unc_pos_y(bel_dis_pos_y, size_win):
    diff_pos_y = {}
    for t in list(bel_dis_pos_y.keys())[1:]:
        t_cur = t
        t_las = t-1
        diff_pos_y[t_cur] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_pos_y[t_cur].values()), list(bel_dis_pos_y[t_las].values()))], ord=1)
    
    unc_pos_y = {}
    for t in list(bel_dis_pos_y.keys()):
        if t <= size_win+2:
            unc_pos_y[t] = 0.9
        else:
            unc_pos_y[t] = max(0.05, sum([diff_pos_y[t] for t in range(t - size_win + 1, t)])/2/(size_win-1))

    for t in list(bel_dis_pos_y.keys()):
        bel_dis_pos_y[t].update((key, val*(1-unc_pos_y[t])) for key, val in bel_dis_pos_y[t].items())
    
    return bel_dis_pos_y

"""Set uncertainty based on perturbation of belief distribution"""
def set_unc_vel_x(bel_dis_vel_x, size_win):
    diff_vel_x = {}
    for t in list(bel_dis_vel_x.keys())[1:]:
        t_cur = t
        t_las = t-1
        diff_vel_x[t_cur] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_vel_x[t_cur].values()), list(bel_dis_vel_x[t_las].values()))], ord=1)
    
    unc_vel_x = {}
    for t in list(bel_dis_vel_x.keys()):
        if t <= size_win+2:
            unc_vel_x[t] = 0.9
        else:
            unc_vel_x[t] = max(0.05, sum([diff_vel_x[t] for t in range(t - size_win + 1, t)])/2/(size_win-1))
    
    for t in list(bel_dis_vel_x.keys()):
        bel_dis_vel_x[t].update((key, val*(1-unc_vel_x[t])) for key, val in bel_dis_vel_x[t].items())
    
    return bel_dis_vel_x

"""Set uncertainty based on perturbation of belief distribution"""
def set_unc_vel_y(bel_dis_vel_y, size_win):
    diff_vel_y = {}
    for t in list(bel_dis_vel_y.keys())[1:]:
        t_cur = t
        t_las = t-1
        diff_vel_y[t_cur] = np.linalg.norm([cur - las for cur, las in zip(list(bel_dis_vel_y[t_cur].values()), list(bel_dis_vel_y[t_las].values()))], ord=1)
    
    unc_vel_y = {}
    for t in list(bel_dis_vel_y.keys()):
        if t <= size_win+2:
            unc_vel_y[t] = 0.9
        else:
            unc_vel_y[t] = max(0.05, sum([diff_vel_y[t] for t in range(t - size_win + 1, t)])/2/(size_win-1))
    
    for t in list(bel_dis_vel_y.keys()):
        bel_dis_vel_y[t].update((key, val*(1-unc_vel_y[t])) for key, val in bel_dis_vel_y[t].items())
    
    return bel_dis_vel_y


# print the belief distributions based on all quantities
#data_all_veh = read_data()
#data_mea = data_all_veh['t0_w2s']
#std_dev = {'t0_w2s':0.1, 't1_w2e':0.1, 't2_w2n':0.1}
#bel_dis = gen_opi(data_all_veh, data_mea, std_dev)
#print(bel_dis['pos_x'])
#print(bel_dis['pos_y'])
#print(bel_dis['vel_x'])
#print(bel_dis['vel_y'])
#opi = set_unc(bel_dis, 10)

#data_mea = read_data()
#bel_dis_pos_y = gen_opi_pos_y(data_mea, 0.1)
##print(set_unc_pos_y(bel_dis_pos_y, 10))
#bel_dis_vel_x = gen_opi_vel_x(data_mea, 5.0)
#print(set_unc_vel_x(bel_dis_vel_x, 10))