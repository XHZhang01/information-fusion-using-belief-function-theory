#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2008-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# Generate simulation scenarios and read data from simulation
# Author: Xuhui Zhang

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import math
import numpy as np
import yaml
import matplotlib.pyplot as plt

# import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
import sumolib  # noqa

if __name__ == "__main__":
    data_all_veh = yaml.load(open('data/all_veh_data.yaml', 'r'))
    # plot quantities of t0_w2s
    plt.figure(1)
    plt.subplot(311)
    plt.plot(data_all_veh['t0_w2s']['pos_x'], data_all_veh['t0_w2s']['pos_y'])
    plt.xlabel('t0_w2s pos_x')
    plt.ylabel('t0_w2s pos_y')
    plt.subplot(312)
    plt.plot(data_all_veh['t0_w2s']['time'], data_all_veh['t0_w2s']['vel_x'])
    plt.xlabel('time')
    plt.ylabel('t0_w2s vel_x')
    plt.subplot(313)
    plt.plot(data_all_veh['t0_w2s']['time'], data_all_veh['t0_w2s']['vel_y'])
    plt.xlabel('time')
    plt.ylabel('t0_w2s vel_y')
    # plot quantities of t1_w2e
    plt.figure(2)
    plt.subplot(311)
    plt.plot(data_all_veh['t1_w2e']['pos_x'], data_all_veh['t1_w2e']['pos_y'])
    plt.xlabel('t1_w2e pos_x')
    plt.ylabel('t1_w2e pos_y')
    plt.subplot(312)
    plt.plot(data_all_veh['t1_w2e']['time'], data_all_veh['t1_w2e']['vel_x'])
    plt.xlabel('time')
    plt.ylabel('t1_w2e vel_x')
    plt.subplot(313)
    plt.plot(data_all_veh['t1_w2e']['time'], data_all_veh['t1_w2e']['vel_y'])
    plt.xlabel('time')
    plt.ylabel('t1_w2e vel_y')
    # plot quantities of t2_w2n
    plt.figure(3)
    plt.subplot(311)
    plt.plot(data_all_veh['t2_w2n']['pos_x'], data_all_veh['t2_w2n']['pos_y'])
    plt.xlabel('t2_w2n pos_x')
    plt.ylabel('t2_w2n pos_y')
    plt.subplot(312)
    plt.plot(data_all_veh['t2_w2n']['time'], data_all_veh['t2_w2n']['vel_x'])
    plt.xlabel('time')
    plt.ylabel('t2_w2n vel_x')
    plt.subplot(313)
    plt.plot(data_all_veh['t2_w2n']['time'], data_all_veh['t2_w2n']['vel_y'])
    plt.xlabel('time')
    plt.ylabel('t2_w2n vel_y')

    plt.subplots_adjust(hspace=1)
    plt.show()