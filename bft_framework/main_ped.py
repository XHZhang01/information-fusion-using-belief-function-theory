import numpy as np
import math
import itertools
import yaml
import pickle
from dcr import current_step_mix
from wbf import fuse_pas_pre
from bel_ass import ass_bel
from opi_gen import gen_opi_pos_y
from opi_gen import gen_opi_vel_x
from opi_gen import gen_opi_bias
from opi_gen import gen_opi_vel_x_ped
from opi_gen import gen_opi_vel_y_ped
from opi_gen import gen_opi_bias_ped
from opi_gen import check_switch
from opi_gen import read_data
from set_unc import set_unc_pos_y
from set_unc import set_unc_vel_x
from set_unc import set_unc_vel_y
from conf_eval import deg_conf
from conf_hand import conf_hand
import matplotlib.pyplot as plt

# calculate uncertainty and insert it in the given opinion
def add_uncertainty(w, str='mu'):
    w[str] = 1-math.fsum(list(w.values()))
    if w[str]<0:
        print('Negative uncertainty')
        breakpoint()

# set relevant parameters
size_win = 4
bel_ini_ped = {'c': 0.1/3, 's': 0.1/3, 'g': 0.1/3, 'csg': 0.9}
std_dev_vel_x = 0.2#1.0
std_dev_vel_y = 0.3#5.0

data_mea = read_data()
t_sidewalk = check_switch()[0]
t_street = check_switch()[1]
bel_dis_vel_x_ped = gen_opi_vel_x_ped(t_sidewalk, t_street, data_mea, std_dev_vel_x)
bel_dis_vel_y_ped = gen_opi_vel_y_ped(t_sidewalk, t_street, data_mea, std_dev_vel_y)
bel_dis_bias_ped = gen_opi_bias_ped(t_sidewalk, t_street)

opi_bef_fus = {'vel_x': bel_dis_vel_x_ped, 'vel_y': bel_dis_vel_y_ped, 'bias': bel_dis_bias_ped}
#print(opi_bef_fus)

opi_vel_x = set_unc_vel_x(bel_dis_vel_x_ped, size_win)
opi_vel_y = set_unc_vel_y(bel_dis_vel_y_ped, size_win)
unc_str = 'csg' # string of uncertainty

#print(opi_bef_fus)
#print(opi_vel_x)

#print(bel_dis_pos_y)
#bel_pos_y_e = {}
#t = list(bel_dis_pos_y.keys())
#for k in bel_dis_pos_y.keys():
#    bel_pos_y_e[k] = list(bel_dis_pos_y[k].values())[1]
#
#plt.figure(1)
#plt.subplot(411)
#plt.plot(t, list(bel_pos_y_e.values()))
#plt.xlabel('time [s]')
#plt.ylabel('belief mass of straight vel_x')
#plt.subplots_adjust(hspace=1)
#plt.show()

#print(bel_dis_vel_x)
#bel_vel_y_c = {}
#t = list(bel_dis_vel_y_ped.keys())
#for k in bel_dis_vel_y_ped.keys():
#    bel_vel_y_c[k] = list(bel_dis_vel_y_ped[k].values())[1]
#bel_vel_y_c = {}
##t = list(bel_dis_vel_y_ped.keys())
#for k in bel_dis_vel_y_ped.keys():
#    bel_vel_y_c[k] = list(bel_dis_vel_y_ped[k].values())[0]

#plt.figure(2)
#plt.subplot(411)
#plt.plot(t, list(bel_vel_x_e.values()))
#plt.xlabel('time [s]')
#plt.ylabel('straight vel_x')
#plt.subplot(412)
#plt.plot(t, list(bel_vel_y_e.values()))
#plt.xlabel('time [s]')
#plt.ylabel('straight pos_y')
#plt.subplots_adjust(hspace=1)
#plt.show()

#print(bel_dis_pos_y)
#print(opi_vel_x)

#print(opi_vel_x)
#print(opi_pos_y)

opi = {}
#opi_bef_fus = {}
opi_ori = {}
for k in opi_vel_x.keys():
    opi[k] = [opi_vel_x[k], opi_vel_y[k], bel_dis_bias_ped[k]]
    #opi[k] = [opi_vel_x[k], opi_vel_y[k]]
    #print(opi)

#print(opi)
#for k in opi_ori.keys():
#    for opi_bf in opi_ori[k]: add_uncertainty(opi_bf, str=unc_str)
#print(opi_bf)
#print(opi_ori)

opi_fused = {}
for k in opi.keys():
    for opi_sou in opi[k]: add_uncertainty(opi_sou, str=unc_str)
    #for opi_sou_ori in opi_original[k]: add_uncertainty(opi_sou_ori, str=unc_str)
    #print(opi[k])
    red = deg_conf(opi[k])
    #print(red)
    #print(opi[k])
    opi_fused[k] = current_step_mix(opi[k], mu_str=unc_str)
    #print(opi_fused[k])
    opi_fused[k] = conf_hand(opi_fused[k], red)
#print(opi_fused)
#print(opi_original)
#opi_vel_x = {}
#opi_vel_y = {}
#opi_bias = {}
#for t in list(opi.keys())[1:]:
#    opi_vel_x[t] = opi[t][0]
#    opi_vel_y[t] = opi[t][1]
#    opi_bias[t] = opi[t][2]

#print(opi_pos_y)

opi_time = list(opi_fused.values())
opi_dev = fuse_pas_pre(bel_ini_ped, opi_fused, mu_str=unc_str)
#print(opi_dev)

#data_bft = {'opinions before fusion': opi_bef_fus, 'belief results given by BFT': opi_dev}
#with open('data/data_ped_bft_imm.txt', 'wb') as handle:
#    pickle.dump(data_bft, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('data/data_ped_bft_imm.txt', 'rb') as handle:
#    test = pickle.load(handle)

#print(test['opinions before fusion'])

t = list(opi_dev.keys())
bel_c = {}
bel_s = {}
bel_g = {}
unc = {}
for k in list(opi_dev.keys()):
    bel_c[k] = float(opi_dev[k]['c'])
    bel_s[k] = float(opi_dev[k]['s'])
    bel_g[k] = float(opi_dev[k]['g'])
    unc[k] = float(opi_dev[k]['csg'])

#print(opi_ori)
#for k in opi_ori.keys():
#    for opi_bf in opi_ori[k]: add_uncertainty(opi_bf, str=unc_str)
#print(opi_ori)
#opi_original = {}
#for k in opi_ori.keys():
#    opi_original[k] = {'pos_y': opi_ori[k][0], 'vel_x': opi_ori[k][1], 'bias': opi_ori[k][2]}
#print(opi_original)
print(opi_bef_fus)

#data_bft = {'opinions before fusion': opi_bef_fus, 'belief results given by BFT': opi_dev}

#with open('data/data_bft_imm.pickle', 'wb') as handle:
#    pickle.dump(data_bft, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('data/data_bft_imm.pickle', 'rb') as handle:
#    test = pickle.load(handle)
#print(test['opinions before fusion'])
#with open('data/data_bft_imm.yaml', 'w') as outfile:
#    yaml.dump(opi, outfile, default_flow_style=False)
#data_opi = yaml.load(open('data/data_bft_imm.yaml', 'r'))
#print(data_opi)

plt.figure(1)
plt.subplot(411)
plt.plot(t, list(bel_c.values()))
plt.xlabel('time [s]')
plt.ylabel('belief mass of crossing')
plt.subplot(412)
plt.plot(t, list(bel_s.values()))
plt.xlabel('time [s]')
plt.ylabel('belief mass of sidewalk')
plt.subplot(413)
plt.plot(t, list(bel_g.values()))
plt.xlabel('time [s]')
plt.ylabel('belief mass of grass')
plt.subplot(414)
plt.plot(t, list(unc.values()))
plt.xlabel('time [s]')
plt.ylabel('uncertainty')
plt.subplots_adjust(hspace=1)
plt.show()

#T = 1
#A = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])
#B = np.array([[0.5*T*T, 0], [T, 0], [0, 0.5*T*T], [0, T]])
#print(A.dot(B) + B)