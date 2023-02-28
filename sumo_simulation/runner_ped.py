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

# import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
import sumolib  # noqa

# input relevant parameters and initialize variables
#veh_list = ['t0_w2s', 't1_w2e', 't2_w2n'] # IDs of vehicles of interest
veh_list = ['ped0'] # IDs of vehicles of interest
quan_list = ['time', 'angle', 'pos_x', 'pos_y', 'vel_x', 'vel_y'] # quantities of interest
data_all = dict.fromkeys(veh_list) # store all vehicles' all quantities
for i in data_all.keys():
    data_all[i] = dict.fromkeys(quan_list)
    for j in data_all[i].keys():
      data_all[i][j] = []

def ctrl_veh():
    #traci.vehicle.setSpeed('t0_w2s') # set the longitudinal speed
    #traci.vehicle.changeLane('t0_w2s', 1, 5) # change to the specific lane for a given duration
    #traci.vehicle.setSpeed('t0_w2s', 5.00)
    #traci.vehicle.setSpeed('t1_w2e', 5.00)
    #traci.vehicle.setSpeed('t2_w2n', 5.00)
    return 0

def read_data(veh_id, data_all):
    # record the current simulation time
    data_all[veh_id]['time'].append(traci.simulation.getTime())
    # convert the read angle to conventional angle
    real_angle = -traci.vehicle.getAngle(veh_id) + 90
    # record the vehicle's current angle
    data_all[veh_id]['angle'].append(real_angle)
    # record the x position in global frame
    data_all[veh_id]['pos_x'].append(traci.vehicle.getPosition(veh_id)[0])
    # record the y position in global frame
    data_all[veh_id]['pos_y'].append(traci.vehicle.getPosition(veh_id)[1])
    # get the vehicle's longitudinal velocity (along the lane)
    vel_long = traci.vehicle.getSpeed(veh_id)
    # compute the vehicle's x velocity in global frame
    vel_x = vel_long * math.cos(real_angle * np.pi / 180)
    data_all[veh_id]['vel_x'].append(vel_x)
    # compute the vehicle's y velocity in global frame
    vel_y = vel_long * math.sin(real_angle * np.pi / 180)
    data_all[veh_id]['vel_y'].append(vel_y)
    # record the vehicle's lateral velocity (across lanes)
    #traci.vehicle.getLateralSpeed(veh_id)

def run():
    """execute the TraCI control loop"""
    start_time = 0
    #end_time = 60
    step = start_time

    pos_y_ref = -4.5
    np.random.normal(0,0.1)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        #if step == 1:
        #    traci.vehicle.changeLane('v0_w2e', 1, 100)
        #    traci.vehicle.changeSublane('v0_w2e', 3)
        #if (step%5) == 0:
        #    traci.vehicle.changeSublane('v0_w2e', np.random.normal(0,0.5))
        #if step == 18:
        #    traci.vehicle.slowDown('v0_w2e', 2.0, 28)
        if step < 62:
            vel_ped = 1.2 + np.random.normal(0,0.1)
            traci.vehicle.setSpeed('ped0', vel_ped)

        for veh_id in veh_list:
            if veh_id in traci.vehicle.getIDList():
                read_data(veh_id, data_all)
        
        step += 1
    with open(r'data/ped_data.yaml','w') as file:
        doc = yaml.dump(data_all, file)

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":

    # start the simulation
    #sumoBinary = checkBinary('sumo') # with no visualization of the simulation
    sumoBinary = checkBinary('sumo-gui') # with visualization of the simulation
    traci.start([sumoBinary, "-c", "structure/jaywalk_ped1.sumocfg", "--lateral-resolution=0.2"])
    run()