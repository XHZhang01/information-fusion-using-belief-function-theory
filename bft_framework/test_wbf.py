# Test WBF with Examples
# Author: Xuhui Zhang

import numpy as np
import math
from wbf import fuse_pas_pre

mu_str = 'ABC'  # string of uncertainty
ini_bel = {'A': 0.1, 'B': 0.1, 'C': 0.1, 'ABC': 0.7} # initial belief distribution
opi_pas_pre = [{'A': 0.5, 'B': 0.1, 'C': 0.2, 'ABC': 0.2},
               {'A': 0.4, 'B': 0.2, 'C': 0.1, 'ABC': 0.3},]  # fused opinion at each time instant 

# update belief with WBF to current time instant
cur_bel = fuse_pas_pre(ini_bel, opi_pas_pre, mu_str = mu_str)
print(sorted(cur_bel.items(), key=len))
