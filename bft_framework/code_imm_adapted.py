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
from opi_gen import read_data
from set_unc import set_unc_pos_y
from set_unc import set_unc_vel_x
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
K = np.array([[0, -1.2085, 0, 0], [0, 0, -0.435, -1.2085]])
F = [A + B.dot(K) for j in range(nI)]
G = [B for j in range(nI)]
# list of closed-loop inputs (-K*x_ref)
#u = [np.random.rand(2, 1) for j in range(nI)]
pos_x_w2s = pos_vel_x(mod = 's')[1]
vel_x_w2s = pos_vel_x(mod = 's')[0]
pos_y_w2s = pos_vel_y(mod = 's')[1]
vel_y_w2s = pos_vel_y(mod = 's')[0]
pos_x_w2e = pos_vel_x(mod = 'e')[1]
vel_x_w2e = pos_vel_x(mod = 'e')[0]
pos_y_w2e = pos_vel_y(mod = 'e')[1]
vel_y_w2e = pos_vel_y(mod = 'e')[0]
pos_x_w2n = pos_vel_x(mod = 'n')[1]
vel_x_w2n = pos_vel_x(mod = 'n')[0]
pos_y_w2n = pos_vel_y(mod = 'n')[1]
vel_y_w2n = pos_vel_y(mod = 'n')[0]
u = {}
z_w2s_ref = {}
z_w2e_ref = {}
z_w2n_ref = {}
for k in pos_x_w2s.keys():
    z_w2s = np.array([[pos_x_w2s[k], vel_x_w2s[k], pos_y_w2s[k], vel_y_w2s[k]]]).T
    u_w2s = -K.dot(z_w2s)
    z_w2s_ref[k] = z_w2s
    z_w2e = np.array([[pos_x_w2e[k], vel_x_w2e[k], pos_y_w2e[k], vel_y_w2e[k]]]).T
    u_w2e = -K.dot(z_w2e)
    z_w2e_ref[k] = z_w2e
    z_w2n = np.array([[pos_x_w2n[k], vel_x_w2n[k], pos_y_w2n[k], vel_y_w2n[k]]]).T
    u_w2n = -K.dot(z_w2n)
    z_w2n_ref[k] = z_w2n

    u[k] = [u_w2s, u_w2e, u_w2n]

muj = np.ones((nI, 1))/nI   # probabilities current estimation (initialized as equally likely)
Q = [np.diag([2.8, 2.8, 0.6, 0.6]) for j in range(nI)] # process noise covariance
R = np.diag([2.8, 0.6]) # measurement noise covariance (this is not a list, is the same for all models)
xj = [np.array([[0.0, 0.0, 0.0, 0.0]]).T for j in range(nI)] # list of states for every model (internal variable of IMM, must be initialized with something reasonable before iteration 0)
Pj = [Q[0] for j in range(nI)] # covariance matrix of estimate according to each model (internal variable of IMM)

#Nsteps = 100
#for k in range(Nsteps):

mea_val = read_data()
mea_pos_x = mea_val['v0_w2e']['pos_x'] #+ float(np.random.normal(0,0.2))
mea_vel_x = mea_val['v0_w2e']['vel_x'] #+ float(np.random.normal(0,0.2))
mea_pos_y = mea_val['v0_w2e']['pos_y'] #+ float(np.random.normal(0,1))
mea_vel_y = mea_val['v0_w2e']['vel_y'] #+ float(np.random.normal(0,1))

muj_dev = {}
muj_w2s_dev = {}
muj_w2e_dev = {}
muj_w2n_dev = {}
xj_dev = {}
Pj_dev = {}
for k in list(pos_x_w2s.keys())[4:80]:
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

#print(muj_dev)

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
bel_ini_vel_x = {'sn': 0.1, 'sen': 0.9}
data_mea = read_data()
std_dev_pos_y = 1.6
std_dev_vel_x = 1.0
bel_dis_pos_y = gen_opi_pos_y(data_mea, std_dev_pos_y)
bel_dis_vel_x = gen_opi_vel_x(data_mea, std_dev_vel_x)
opi_pos_y = set_unc_pos_y(bel_dis_pos_y, size_win)
opi_vel_x = set_unc_vel_x(bel_dis_vel_x, size_win)
opi_bias = gen_opi_bias()
#print(opi_bias)
unc_str = 'sen' # string of uncertainty

#print(bel_dis_vel_x)
#print(opi_vel_x)
#print(opi_pos_y)

opi = {}
opi_bef_fus = {}
opi_ori = {}
for k in opi_pos_y.keys():
    opi[k] = [opi_pos_y[k], opi_vel_x[k], opi_bias[k]]
    if int(str(k)) >= 4:
        opi_bef_fus[k] = {'pos_y': opi_pos_y[k], 'vel_x': opi_vel_x[k], 'bias': opi_bias[k]}
        opi_ori[k] = [opi_pos_y[k].copy(), opi_vel_x[k].copy(), opi_bias[k].copy()]
#print(opi_ori)
opi_fused = {}
for k in list(opi.keys())[3:]:
    for opi_sou in opi[k]: add_uncertainty(opi_sou, str=unc_str)
    red = deg_conf(opi[k])
    #print(opi[k])
    opi_fused[k] = current_step_mix(opi[k], mu_str=unc_str)
    #print(opi_fused[k])
    opi_fused[k] = conf_hand(opi_fused[k], red)
#print(opi_fused)
#print(opi)
opi_pos_y = {}
opi_vel_x = {}
opi_bias = {}
for t in list(opi.keys())[5:]:
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

#bel_test = bel_s[10]
#print(bel_test)

for k in opi_ori.keys():
    for opi_bf in opi_ori[k]: add_uncertainty(opi_bf, str=unc_str)
#print(opi_ori)
opi_original = {}
for k in opi_ori.keys():
    opi_original[k] = {'pos_y': opi_ori[k][0], 'vel_x': opi_ori[k][1], 'bias': opi_ori[k][2]}
#print(opi_original)

data_bft = {'opinions before fusion': opi_original, 'belief results given by BFT': opi_dev}
data_imm = {'probability results given by imm': muj_dev}
data_both = {'bft': data_bft, 'imm': data_imm}

with open('data/data_bft_imm.txt', 'wb') as handle:
    pickle.dump(data_both, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/data_bft_imm.txt', 'rb') as handle:
    test = pickle.load(handle)
#print(test['bft'])