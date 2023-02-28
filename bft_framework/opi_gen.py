import math
import itertools
import yaml
import numpy as np
from bel_ass import ass_bel
from models import lane_mod
from models import vel_mod
from models import vel_x_ped
from models import vel_y_ped
import matplotlib.pyplot as plt

"""Read vehicles' data from simulation"""
def read_data():
    #data_all_veh = yaml.load(open('data/all_veh_data_tune2.yaml', 'r'))
    data_all_veh = yaml.load(open('data/ped_data.yaml', 'r'))
    return data_all_veh

"""Generate opinions based on measured value and model-given values"""
#def gen_opi(data_all_mod, data_mea, std_dev):
#    modkeys = data_all_veh.keys() # all possible models
#    # find the period wthin which all models are active
#    t_start = [min(data_mea['time'])]
#    for k in modkeys:
#        t_start.append(min(data_all_veh[k]['time']))
#    t_start = max(t_start)
#    t_end = [max(data_mea['time'])]
#    for k in modkeys:
#        t_end.append(max(data_all_veh[k]['time']))
#    t_end = min(t_end)
#
#    index_start = data_mea['time'].index(t_start)
#    index_end = data_mea['time'].index(t_end)
#
#    bel_dis_pos_x = {}
#    bel_dis_pos_y = {}
#    bel_dis_vel_x = {}
#    bel_dis_vel_y = {}
#    bel_dis = {'pos_x': bel_dis_pos_x, 'pos_y': bel_dis_pos_y, 'vel_x': bel_dis_vel_x, 'vel_y': bel_dis_vel_y}
#    for i in range(index_start, index_end+1):
#        t_current = data_mea['time'][i]
#        pos_x_mea = data_mea['pos_x'][i] + np.random.normal(0,0.5)
#        pos_y_mea = data_mea['pos_y'][i] + np.random.normal(0,0.5)
#        vel_x_mea = data_mea['vel_x'][i] + np.random.normal(0,0.5)
#        vel_y_mea = data_mea['vel_y'][i] + np.random.normal(0,0.5)
#        pos_x_diff = {}
#        pos_y_diff = {}
#        vel_x_diff = {}
#        vel_y_diff = {}
#        for k in modkeys:
#            index_veh = data_all_veh[k]['time'].index(t_current)
#            pos_x_diff[k] = data_all_veh[k]['pos_x'][index_veh]
#            pos_y_diff[k] = data_all_veh[k]['pos_y'][index_veh]
#            vel_x_diff[k] = data_all_veh[k]['vel_x'][index_veh]
#            vel_y_diff[k] = data_all_veh[k]['vel_y'][index_veh]
#            std_dev_k = std_dev[k]
#        bel_dis_pos_x[t_current] = ass_bel(pos_x_mea, pos_x_diff, std_dev_k)
#        bel_dis_pos_y[t_current] = ass_bel(pos_y_mea, pos_y_diff, std_dev_k)
#        bel_dis_vel_x[t_current] = ass_bel(vel_x_mea, vel_x_diff, std_dev_k)
#        bel_dis_vel_y[t_current] = ass_bel(vel_y_mea, vel_y_diff, std_dev_k)
#
#    return bel_dis

"""Generate opinions based on y positions"""
def gen_opi_pos_y(data_mea, std_dev_pos_y):
    t_start = 1
    t_end = 80

    bel_dis_pos_y = {}
    for t in range(t_start, t_end):
        #pos_y_mea = data_mea['v0_w2e']['pos_y'][t] + np.random.normal(0,0.2)
        pos_y_mea = data_mea['v0_w2e']['pos_y'][t] + np.random.normal(0,0.2)
        pos_y_mod_s = lane_mod(t, mod = 's')
        pos_y_mod_e = lane_mod(t, mod = 'e')
        pos_y_mod_n = lane_mod(t, mod = 'n')
        
        pos_y_mod = {'s': pos_y_mod_s, 'e': pos_y_mod_e, 'n': pos_y_mod_n}

        bel_dis_pos_y[t] = ass_bel(pos_y_mea, pos_y_mod, std_dev_pos_y)
    return bel_dis_pos_y

"""Generate opinions based on x velocities"""
def gen_opi_vel_x(data_mea, std_dev_vel_x):
    t_start = 1
    t_end = 80

    bel_dis_vel_x = {}
    for t in range(t_start, t_end):
        #vel_x_mea = data_mea['v0_w2e']['vel_x'][t] + np.random.normal(0,1)
        if t > 30 and t < 40:
            vel_x_mea = data_mea['v0_w2e']['vel_x'][t] + np.random.normal(0,2)
        else:
            if t > 60 and t < 70:
                vel_x_mea = data_mea['v0_w2e']['vel_x'][t] + np.random.normal(0,2)
            else:
                vel_x_mea = data_mea['v0_w2e']['vel_x'][t] + np.random.normal(0,1)
        vel_x_mod_e = vel_mod(t, mod = 'e')
        vel_x_mod_sn = vel_mod(t, mod = 'sn')
        
        vel_x_mod = {'e': vel_x_mod_e, 'sn': vel_x_mod_sn}

        bel_dis_vel_x[t] = ass_bel(vel_x_mea, vel_x_mod, std_dev_vel_x)
    return bel_dis_vel_x

"""Generate the bias opinions"""
def gen_opi_bias():
    t_start = 1
    t_end = 80

    bel_dis_bias = {}
    for t in range(t_start, t_end):
        #bias_mod_s = float(0.2 + np.random.normal(0,0.02))
        #bias_mod_n = float(0.2 + np.random.normal(0,0.02))
        #bias_unc = float(0.2 + np.random.normal(0,0.02))
        bias_mod_s = float(0.18)
        bias_mod_n = float(0.17)
        bias_unc = float(0.33)
        bias_mod_e = float(1 - bias_mod_s - bias_mod_n - bias_unc)
        bel_dis_bias[t] = {'s': bias_mod_s, 'e': bias_mod_e, 'n': bias_mod_n}

    return bel_dis_bias

def check_switch():
    data_mea = read_data()
    pos_y = data_mea['ped0']['pos_y']
    t_sidewalk = 0
    t_street = 0
    for pos_y_t in pos_y:
        if pos_y_t > -2.0 and pos_y_t <-1.6:
            i = pos_y.index(pos_y_t)
            t_sidewalk = data_mea['ped0']['time'][i]
        if pos_y_t > 8.2 and pos_y_t < 8.8:
            i = pos_y.index(pos_y_t)
            t_street = data_mea['ped0']['time'][i]
    return t_sidewalk, t_street   

"""opinion generation based on vel_x"""
def gen_opi_vel_x_ped(t_sidewalk, t_street, data_mea, std_dev_vel_x):
    t_start = 2
    t_end = 62

    bel_dis_vel_x = {}
    for t in range(t_start, t_end):
        vel_x_mea = data_mea['ped0']['vel_x'][t] + np.random.normal(0,0.2)
        vel_x_mod_c = vel_x_ped(t_sidewalk, t_street, t, mod = 'c')
        vel_x_mod_s = vel_x_ped(t_sidewalk, t_street, t, mod = 's')
        vel_x_mod_g = vel_x_ped(t_sidewalk, t_street, t, mod = 'g')
        
        vel_x_mod = {'c': vel_x_mod_c, 's': vel_x_mod_s, 'g': vel_x_mod_g}

        bel_dis_vel_x[t] = ass_bel(vel_x_mea, vel_x_mod, std_dev_vel_x)
    return bel_dis_vel_x

"""opinion generation based on vel_y"""
def gen_opi_vel_y_ped(t_sidewalk, t_street, data_mea, std_dev_vel_y):
    t_start = 2
    t_end = 62

    bel_dis_vel_y = {}
    for t in range(t_start, t_end):
        vel_y_mea = data_mea['ped0']['vel_y'][t] + np.random.normal(0,0.4)
        vel_y_mod_c = vel_y_ped(t_sidewalk, t_street, t, mod = 'c')
        vel_y_mod_s = vel_y_ped(t_sidewalk, t_street, t, mod = 's')
        vel_y_mod_g = vel_y_ped(t_sidewalk, t_street, t, mod = 'g')
        
        vel_y_mod = {'c': vel_y_mod_c, 's': vel_y_mod_s, 'g': vel_y_mod_g}

        bel_dis_vel_y[t] = ass_bel(vel_y_mea, vel_y_mod, std_dev_vel_y)
    return bel_dis_vel_y

"""bias opinion generation"""
def gen_opi_bias_ped(t_sidewalk, t_street):
    t_start = 2
    t_end = 62

    bel_dis_bias = {}
    for t in range(t_start, t_end):
        #bias_mod_s = float(0.2 + np.random.normal(0,0.02))
        #bias_mod_n = float(0.2 + np.random.normal(0,0.02))
        #bias_unc = float(0.2 + np.random.normal(0,0.02))
        bias_mod_c = float(0.36)
        bias_mod_s = float(0.22)
        bias_unc = float(0.32)
        #if t < 12:
        #    bias_mod_c = float(0.12)
        #    bias_mod_s = float(0.22)
        #    bias_unc = float(0.34)
        #else:
        #    bias_mod_c = float(0.36)
        #    bias_mod_s = float(0.22)
        #    bias_unc = float(0.32)
        bias_mod_g = float(1 - bias_mod_c - bias_mod_s - bias_unc)
        bel_dis_bias[t] = {'c': bias_mod_c, 's': bias_mod_s, 'g': bias_mod_g}

    return bel_dis_bias

# print the belief distributions based on all quantities
#data_all_veh = read_data()
#data_mea = data_all_veh['t0_w2s']
#std_dev = {'t0_w2s':0.1, 't1_w2e':0.1, 't2_w2n':0.1}
#bel_dis = gen_opi(data_all_veh, data_mea, std_dev)
#print(bel_dis['pos_x'])
#print(bel_dis['pos_y'])
#print(bel_dis['vel_x'])
#print(bel_dis['vel_y'])

#data_mea = read_data()
#print(data_mea)
#print(data_mea['v0_w2e']['pos_y'][10])
#std_dev_pos_y = 0.1
#print(gen_opi_pos_y(data_mea, std_dev_pos_y))

#data_mea = read_data()
#print(data_mea)
#print(data_mea['v0_w2e']['vel_x'][10])
#std_dev_vel_x = 0.5
#gen_opi_vel_x(data_mea, std_dev_vel_x)

#print(gen_opi_bias())

#data_mea = read_data()
##print(data_mea['ped0']['pos_y'])
#t_sidewalk = check_switch()[0]
#t_street = check_switch()[1]
##print(t_sidewalk, t_street)
#bel_dis_vel_x_ped = gen_opi_vel_x_ped(t_sidewalk, t_street, data_mea, 0.2)
#bel_dis_vel_y_ped = gen_opi_vel_y_ped(t_sidewalk, t_street, data_mea, 0.4)
#bel_dis_bias_ped = gen_opi_bias_ped(t_sidewalk, t_street)
##print(bel_dis_vel_x_ped)
##print(bel_dis_vel_y_ped)
##print(bel_dis_bias_ped)
#
#t = list(bel_dis_vel_x_ped.keys())
#m_velx_c = {}
#m_velx_s = {}
#m_velx_g = {}
#for k in bel_dis_vel_x_ped.keys():
#    m_velx_c[k] = bel_dis_vel_x_ped[k]['c']
#    m_velx_s[k] = bel_dis_vel_x_ped[k]['s']
#    m_velx_g[k] = bel_dis_vel_x_ped[k]['g']
#
#m_vely_c = {}
#m_vely_s = {}
#m_vely_g = {}
#for k in bel_dis_vel_y_ped.keys():
#    m_vely_c[k] = bel_dis_vel_y_ped[k]['c']
#    m_vely_s[k] = bel_dis_vel_y_ped[k]['s']
#    m_vely_g[k] = bel_dis_vel_y_ped[k]['g']

#plt.figure(1)
#plt.subplot(411)
#plt.plot(t, list(m_velx_c.values()))
#plt.xlabel('time [s]')
#plt.ylabel('mass of crossing vel_x')
#plt.subplot(412)
#plt.plot(t, list(m_vely_c.values()))
#plt.xlabel('time [s]')
#plt.ylabel('mass of crossing vel_y')
#plt.show()