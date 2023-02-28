import numpy as np
import yaml
import pickle
#from models import lane_mod
#from models import vel_mod
from models import pos_vel_x
from models import pos_vel_y
from opi_gen import read_data
import matplotlib.pyplot as plt

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
from models import vel_x_ped
from models import vel_y_ped
from opi_gen import check_switch
from opi_gen import read_data
from set_unc import set_unc_pos_y
from set_unc import set_unc_vel_x
from conf_eval import deg_conf
from conf_hand import conf_hand
import matplotlib.pyplot as plt

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

def imm_alg_inputs(nI, F, G, H, Q, R, Pi_ap, y, u, muj, xj, Pj):
    x = 0*xj[0]
    P = 0*Pj[0]
    muj_tilde = 0*Pi_ap
    xj0 = [0*x for _ in range(nI)] # combined state estimate
    Pj0 = [0*P for _ in range (nI)] # combined covariance matrix
    cjbar = []
    # state interaction
    for j in range(nI):
        cjbar.append(Pi_ap[:, j].T.dot(muj))
        #print(cjbar)
        for i in range(nI):
            muj_tilde[i, j] = muj[i]*Pi_ap[i, j]/cjbar[j]
            xj0[j] += xj[i]*muj_tilde[i, j]
        for i in range(nI):
            Pj0[j] += muj_tilde[i, j]*(Pj[i]+(xj[i]-xj0[j]).dot((xj[i]-xj0[j]).T))
    # model probability update
    lambdaj = np.zeros((nI, 1))
    for j in range(nI):
        # constant velocity or linear velocity, still need this dynamic equation
        xj_tilde = F[j].dot(xj0[j])+G[j].dot(u[j])
        #print(xj_tilde)
        Pj_tilde = F[j]*Pj0[j]*F[j].T+Q[j]
        z = y-H.dot(xj_tilde) # the same measured state for all models, specify for 2 types of sensors, pos_y and vel_x
        Sj_tilde = H.dot(Pj_tilde).dot(H.T)+R
        Sj_tildeinv = np.linalg.inv(Sj_tilde)
        Lj = Pj_tilde.dot(H.T).dot(np.linalg.inv(Sj_tilde))
        xj[j] = xj_tilde+Lj.dot(z)
        Pj[j] = (np.eye(x.shape[0])-Lj.dot(H)).dot(Pj_tilde)
        lambdaj[j] = np.exp(-0.5*z.T.dot(Sj_tildeinv).dot(z))/np.sqrt(np.linalg.det(2*np.pi*Sj_tilde))
    const = lambdaj.T.dot(cjbar)
    # # state estimate combination
    for j in range(nI):
        muj[j] = lambdaj[j]*cjbar[j]/const
        x += muj[j]*xj[j]
    #for j in range(nI):
    #    P += muj[j]*(Pj[j]+(x-xj[j]).dot((x-xj[j]).T))
    #print(muj)
    return muj, xj, Pj, x, P

nI = 3; #  number of candidate models
Pi_ap =  np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]) #   a-priori switching probability
T = 1 # sample time in the simulation
# dynamic equations of the linear model
# z_plus = A*z + B*K*(z - z_ref)
# z = [x, vx, y, vy]'
A = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])
B = np.array([[0.5*T*T, 0], [T, 0], [0, 0.5*T*T], [0, T]])
H = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])  # measurement matrix (vel_x and pos_y)
#Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # weight matrix of states in LQR
#R = np.array([[1, 0], [0, 1]]) # weight matrix of inputs in LQR
#K = np.array([[-0.4345, -1.2085, 0, 0], [0, 0, -0.435, -1.2085]]) # feedback from Infinite-horizon and discrete-time LQR
K = np.array([[0, -1.2085, 0, 0], [0, 0, 0, -1.2085]])
F = [A + B.dot(K) for j in range(nI)]
G = [B for j in range(nI)]
# list of closed-loop inputs (-K*x_ref)
#u = [np.random.rand(2, 1) for j in range(nI)]

data_mea = read_data()
t_start = 2
t_end = 62
t_sidewalk = check_switch()[0]
t_street = check_switch()[1]

u = {}
z_crossing_ref = {}
z_sidewalk_ref = {}
z_grass_ref = {}
for t in range(t_start, t_end):
    vel_x_crossing = vel_x_ped(t_sidewalk, t_street, t, mod = 'c')
    vel_y_crossing = vel_y_ped(t_sidewalk, t_street, t, mod = 'c')
    #print(vel_x_crossing)
    z_crossing_ref[t] = np.array([[0, vel_x_crossing, 0, vel_y_crossing]]).T
    #print(z_crossing_ref)
    u_crossing = -K.dot(z_crossing_ref[t])
    #print(u_crossing)

    vel_x_sidewalk = vel_x_ped(t_sidewalk, t_street, t, mod = 's')
    vel_y_sidewalk = vel_y_ped(t_sidewalk, t_street, t, mod = 's')
    z_sidewalk_ref[t] = np.array([[0, vel_x_sidewalk, 0, vel_y_sidewalk]]).T
    u_sidewalk = -K.dot(z_sidewalk_ref[t])

    vel_x_grass = vel_x_ped(t_sidewalk, t_street, t, mod = 'g')
    vel_y_grass = vel_y_ped(t_sidewalk, t_street, t, mod = 'g')
    z_grass_ref[t] = np.array([[0, vel_x_grass, 0, vel_y_grass]]).T
    u_grass = -K.dot(z_grass_ref[t])

    u[t] = [u_crossing, u_sidewalk, u_grass]

muj = np.ones((nI, 1))/nI   # probabilities current estimation (initialized as equally likely)
Q = [np.diag([2.8, 2.8, 0.6, 0.6]) for j in range(nI)] # process noise covariance
R = np.diag([2.8, 0.6]) # measurement noise covariance (this is not a list, is the same for all models)
xj = [np.array([[0.0, 0.0, 0.0, 0.0]]).T for j in range(nI)] # list of states for every model (internal variable of IMM, must be initialized with something reasonable before iteration 0)
Pj = [Q[0] for j in range(nI)] # covariance matrix of estimate according to each model (internal variable of IMM)

#Nsteps = 100
#for k in range(Nsteps):

mea_val = read_data()
mea_pos_x = mea_val['ped0']['pos_x'] #+ float(np.random.normal(0,0.2))
mea_vel_x = mea_val['ped0']['vel_x'] #+ float(np.random.normal(0,0.2))
mea_pos_y = mea_val['ped0']['pos_y'] #+ float(np.random.normal(0,1))
mea_vel_y = mea_val['ped0']['vel_y'] #+ float(np.random.normal(0,1))

muj_dev = {}
muj_w2s_dev = {}
muj_w2e_dev = {}
muj_w2n_dev = {}
xj_dev = {}
Pj_dev = {}
for k in range(t_start, t_end):
    #y = np.random.rand(2, 1) # last measurement (updated at every sampling time)
    y = np.array([[mea_vel_x[k] + float(np.random.normal(0,0.2)), mea_pos_y[k] + float(np.random.normal(0,1))]]).T
    #print(y)
    #print([np.random.rand(2, 1) for j in range(nI)])
    #print(u[k])
    #print(z_w2n_ref[k])
    imm_output = imm_alg_inputs(nI, F, G, H, Q, R, Pi_ap, y, u[k], muj, xj, Pj)
    muj = imm_output[0]
    muj_w2s_dev[k] = float(muj[0][0])
    muj_w2e_dev[k] = float(muj[1][0])
    muj_w2n_dev[k] = float(muj[2][0])
    muj_dev[k] = [float(muj[0][0]), float(muj[1][0]), float(muj[2][0])]
    xj = imm_output[1]
    xj_dev[k] = xj
    Pj = imm_output[2]
    Pj_dev[k] = Pj

print(muj_dev)

#with open('data/data_imm.pickle', 'wb') as handle:
#    pickle.dump(muj_dev, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('data/data_imm.pickle', 'rb') as handle:
#    test = pickle.load(handle)
#print(test)
#with open(r'data/data_bft_imm.yaml','w') as file:
#    doc = yaml.dump(muj_dev, file)
#data = yaml.load(open('data/data_bft_imm.yaml', 'r'))
#print(data)

#t = list(muj_dev.keys())
#plt.figure(1)
#plt.subplot(311)
#plt.plot(t, list(muj_w2s_dev.values()))
#plt.xlabel('time [s]')
#plt.ylabel('probability of right turn')
#plt.subplot(312)
#plt.plot(t, list(muj_w2e_dev.values()))
#plt.xlabel('time [s]')
#plt.ylabel('probability of straight')
#plt.subplot(313)
#plt.plot(t, list(muj_w2n_dev.values()))
#plt.xlabel('time [s]')
#plt.ylabel('probability of left turn')
#plt.subplots_adjust(hspace=1)
#plt.show()


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

data_bft = {'opinions before fusion': opi_bef_fus, 'belief results given by BFT': opi_dev}
data_imm = {'probability results given by imm': muj_dev}
data_both = {'bft': data_bft, 'imm': data_imm}
with open('data/data_ped_bft_imm_new1.txt', 'wb') as handle:
    pickle.dump(data_both, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/data_ped_bft_imm_new1.txt', 'rb') as handle:
    test = pickle.load(handle)

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
