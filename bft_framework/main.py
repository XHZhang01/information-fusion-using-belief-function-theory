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
from opi_gen import read_data
from set_unc import set_unc_pos_y
from set_unc import set_unc_vel_x
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
#size_win = 10
#bel_ini = {'t0_w2s': 0.1/3, 't1_w2e': 0.1/3, 't0_w2st1_w2et2_w2n': 0.9, 't2_w2n': 0.1/3}
#data_all_veh = read_data()
#data_mea = data_all_veh['t0_w2s']
#std_dev = {'t0_w2s':0.1, 't1_w2e':0.1, 't2_w2n':0.1}
#bel_dis = gen_opi(data_all_veh, data_mea, std_dev)
#opi = set_unc(bel_dis, size_win)
#unc_str = 't0_w2st1_w2et2_w2n' # string of uncertainty
#opi_fused = {}
#for k in opi.keys():
#    for opi_sou in opi[k]: add_uncertainty(opi_sou, str=unc_str)
#    opi_fused[k] = current_step_mix(opi[k], mu_str=unc_str)
#opi_time = list(opi_fused.values())
#opi_dev = fuse_pas_pre(bel_ini, opi_fused, mu_str=unc_str)
#print(opi_dev)

# set relevant parameters
#size_win = 4
#bel_ini = {'s': 0.1/6, 'e': 0.1/6, 'se': 0.1/6, 'sn': 0.1/6, 'en': 0.1/6, 'sen': 0.9, 'n': 0.1/6}
#data_all_veh = read_data()
#data_mea = data_all_veh['v0_w2e']
#std_dev = {'s': 0.1, 'e': 0.1, 'se': 0.1, 'sn': 0.1, 'en': 0.1, 'n': 0.1}
#bel_dis = gen_opi(data_all_veh, data_mea, std_dev)
#opi = set_unc(bel_dis, size_win)
#unc_str = 'sen' # string of uncertainty
#opi_fused = {}
#for k in opi.keys():
#    for opi_sou in opi[k]: add_uncertainty(opi_sou, str=unc_str)
#    opi_fused[k] = current_step_mix(opi[k], mu_str=unc_str)
#opi_time = list(opi_fused.values())
#opi_dev = fuse_pas_pre(bel_ini, opi_fused, mu_str=unc_str)
#print(opi_dev)

# set relevant parameters
size_win = 10
bel_ini_pos_y = {'s': 0.1/3, 'e': 0.1/3, 'sen': 0.9, 'n': 0.1/3}
bel_ini_vel_x = {'e': 0.05, 'sn': 0.5, 'sen': 0.9}
data_mea = read_data()
std_dev_pos_y = 1.6#1.0
std_dev_vel_x = 1.0#5.0
bel_dis_pos_y = gen_opi_pos_y(data_mea, std_dev_pos_y)
bel_dis_vel_x = gen_opi_vel_x(data_mea, std_dev_vel_x)
opi_pos_y = set_unc_pos_y(bel_dis_pos_y, size_win)
opi_vel_x = set_unc_vel_x(bel_dis_vel_x, size_win)
opi_bias = gen_opi_bias()
unc_str = 'sen' # string of uncertainty

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
bel_pos_y_e = {}
t = list(bel_dis_pos_y.keys())
for k in bel_dis_pos_y.keys():
    bel_pos_y_e[k] = list(bel_dis_pos_y[k].values())[1]
bel_vel_x_e = {}
#t = list(bel_dis_vel_x.keys())
for k in bel_dis_vel_x.keys():
    bel_vel_x_e[k] = list(bel_dis_vel_x[k].values())[0]

plt.figure(2)
plt.subplot(411)
plt.plot(t, list(bel_vel_x_e.values()))
plt.xlabel('time [s]')
plt.ylabel('straight vel_x')
plt.subplot(412)
plt.plot(t, list(bel_pos_y_e.values()))
plt.xlabel('time [s]')
plt.ylabel('straight pos_y')
plt.subplots_adjust(hspace=1)
#plt.show()

#print(bel_dis_pos_y)
#print(opi_vel_x)

#print(opi_vel_x)
#print(opi_pos_y)

opi = {}
opi_bef_fus = {}
opi_ori = {}
for k in opi_pos_y.keys():
    opi[k] = [opi_pos_y[k], opi_vel_x[k], opi_bias[k]]
    #print(opi)
    if int(str(k)) >= 4:
        opi_bef_fus[k] = {'pos_y': opi_pos_y[k], 'vel_x': opi_vel_x[k], 'bias': opi_bias[k]}
        opi_ori[k] = [opi_pos_y[k].copy(), opi_vel_x[k].copy(), opi_bias[k].copy()]

#print(opi_ori)
#for k in opi_ori.keys():
#    for opi_bf in opi_ori[k]: add_uncertainty(opi_bf, str=unc_str)
#print(opi_bf)
#print(opi_ori)

opi_fused = {}
for k in list(opi.keys())[3:]:
    for opi_sou in opi[k]: add_uncertainty(opi_sou, str=unc_str)
    #for opi_sou_ori in opi_original[k]: add_uncertainty(opi_sou_ori, str=unc_str)
    #print(opi[k])
    red = deg_conf(opi[k])
    #print(opi[k])
    opi_fused[k] = current_step_mix(opi[k], mu_str=unc_str)
    #print(opi_fused[k])
    opi_fused[k] = conf_hand(opi_fused[k], red)
#print(opi_fused)
#print(opi_original)
opi_pos_y = {}
opi_vel_x = {}
opi_bias = {}
for t in list(opi.keys())[4:]:
    opi_pos_y[t] = opi[t][0]
    opi_vel_x[t] = opi[t][1]
    opi_bias[t] = opi[t][2]

#print(opi_pos_y)

opi_time = list(opi_fused.values())
opi_dev = fuse_pas_pre(bel_ini_pos_y, opi_fused, mu_str=unc_str)
#print(opi_dev)

t = list(opi_dev.keys())
bel_s = {}
bel_e = {}
bel_n = {}
unc = {}
for k in list(opi_dev.keys()):
    bel_s[k] = float(opi_dev[k]['s'])
    bel_e[k] = float(opi_dev[k]['e'])
    bel_n[k] = float(opi_dev[k]['n'])
    unc[k] = float(opi_dev[k]['sen'])

#print(opi_ori)
for k in opi_ori.keys():
    for opi_bf in opi_ori[k]: add_uncertainty(opi_bf, str=unc_str)
#print(opi_ori)
opi_original = {}
for k in opi_ori.keys():
    opi_original[k] = {'pos_y': opi_ori[k][0], 'vel_x': opi_ori[k][1], 'bias': opi_ori[k][2]}
#print(opi_original)

data_bft = {'opinions before fusion': opi_bef_fus, 'belief results given by BFT': opi_dev}

with open('data/data_bft_imm.pickle', 'wb') as handle:
    pickle.dump(data_bft, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/data_bft_imm.pickle', 'rb') as handle:
    test = pickle.load(handle)
#print(test['opinions before fusion'])
#with open('data/data_bft_imm.yaml', 'w') as outfile:
#    yaml.dump(opi, outfile, default_flow_style=False)
#data_opi = yaml.load(open('data/data_bft_imm.yaml', 'r'))
#print(data_opi)

plt.figure(1)
plt.subplot(411)
plt.plot(t, list(bel_s.values()))
plt.xlabel('time [s]')
plt.ylabel('belief mass of right turn')
plt.subplot(412)
plt.plot(t, list(bel_e.values()))
plt.xlabel('time [s]')
plt.ylabel('belief mass of straight')
plt.subplot(413)
plt.plot(t, list(bel_n.values()))
plt.xlabel('time [s]')
plt.ylabel('belief mass of left turn')
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