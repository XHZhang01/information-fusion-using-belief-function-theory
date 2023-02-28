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

with open('data/data_ped_bft_imm_new1.txt', 'rb') as handle:
    data_all = pickle.load(handle)

data_bft = data_all['bft'] # contains both opinions before fusion and update at each instant and bft's results (belief development after fusion and update)
opi_bef_fus = data_bft['opinions before fusion'] # opinions before fusion, keys of this dict is time instant
print(opi_bef_fus)
opi_dev = data_bft['belief results given by BFT'] # bft's results (belief development after fusion and update), keys of this dict is time instant
#print(opi_dev)
data_imm = data_all['imm']
data_imm_muj = data_imm['probability results given by imm'] # muj probability development given by imm, keys of this dict is time instant
#print(data_imm_muj)
t = list(data_imm_muj.keys())
muj_crossing_dev = {}
for i in t:
    muj_crossing_dev[i] = data_imm_muj[i][0]
plt.figure(1)
plt.subplot(311)
plt.plot(t, list(muj_crossing_dev.values()))
plt.xlabel('time [s]')
plt.ylabel('probability of crossing')
plt.subplots_adjust(hspace=1)
plt.show()

opi_vel_x = opi_bef_fus['vel_x']
opi_vel_y = opi_bef_fus['vel_y']
opi_bias = opi_bef_fus['bias']
#print(opi_vel_x) # development of opinions based on pos_y at each intant, keys of this dict is time instant
#print(opi_vel_y) # development of opinions based on vel_x at each intant (before redistribution with subunion's mass), keys of this dict is time instant
#print(opi_bias) # development of opinions based on bias at each intant, keys of this dict is time instant

# in each opinion, 'c' is crossing, 's' is sidewalk, 'g' is grass, 'csg': is the uncertainty

# time instants, from 2 to 61