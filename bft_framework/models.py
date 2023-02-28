import math
import numpy as np
import itertools

# Models for opinion generation
# For sensor based on pos_y
# pos_y = -1.6, the left lane represents left turn
# pos_y = -4.8, the middle lane represents straight
# pos_y = -8.0, the right lane represents right turn
# For sensor based on vel_x
# vel_x = 12.0, constant velocity represents straight
# vel_x = 12.0 for t < 20.0s, constant velocity before 20.0s before intersection
#       = 12.0 - 11.0/28*(t - 20) for t >= 20.0s and t < 48.0s, linear decceleration from 12m/s to 1m/s between [20, 48) before intersection (if needed, we can only use this part)
#       = 1.0 + 11.0/5*(t - 48) for t >= 48.0s and t < 53.0s, linear acceleration between [48, 53) from 1m/s to 12m/s after intersection
#       = 12.0 for t >= 53.0s and t <= 80.0s, constant velocity until the end after intersection
# The vehicle starts at 0.0s and ends at 80.0s

# assume each acceleration starts at corresponding time stampe and keeps constant until next
def acc_gen(acc, timestamp):
    timestep = 1
    t_start = timestamp[0]
    t_end = timestamp[-1]
    acc_time = {}
    for a in acc:
        i = acc.index(a)
        t_s = timestamp[i]
        t_e = timestamp[i+1]
        for t in range(t_s, t_e, timestep):
            acc_time[t] = a
    for t in range(timestamp[-2], timestamp[-1]):
        acc_time[t] = acc[-1]
    acc_time[t_end] = acc[-1]
    return acc_time

# integrate temporal derivative with forward Euler method
def for_euler(diff, ini):
    int = {}
    last = 0
    diffkeys = list(diff.keys())
    for k in diff.keys():
        if k is diffkeys[0]:
            int[k] = ini
            last = ini
        else:
            i = diffkeys.index(k)
            int[k] = last + diff[i - 1]
            last = int[k]
    return int

# generate model-given pos_x and vel_x values
def pos_vel_x(mod = 'sen'):
    acc_w2s = [0, -10/28, 10/5, 0] # right, west to south
    ini_vel_w2s = 12
    ini_pos_w2s = -394.9
    timestamp_w2s = [0, 20, 48, 53, 80]
    acc_x_w2s = acc_gen(acc_w2s, timestamp_w2s)
    acc_w2e = [0]# straight, west to east
    timestamp_w2e = [0, 80]
    ini_vel_w2e = 12
    ini_pos_w2e = -394.9
    acc_x_w2e = acc_gen(acc_w2e, timestamp_w2e)
    acc_w2n = [0, -10/28, 10/5, 0] # left, west to north
    ini_vel_w2n = 12
    ini_pos_w2n = -394.9
    timestamp_w2n = [0, 20, 48, 53, 80]
    acc_x_w2n = acc_gen(acc_w2n, timestamp_w2n)

    if mod == 'e':
        vel_x = for_euler(acc_x_w2e, ini_vel_w2e)
        pos_x = for_euler(vel_x, ini_pos_w2e)
    else:
        if mod == 's':
            vel_x = for_euler(acc_x_w2s, ini_vel_w2s)
            pos_x = for_euler(vel_x, ini_pos_w2s)
        else:
            if mod == 'n':
                vel_x = for_euler(acc_x_w2n, ini_vel_w2n)
                pos_x = for_euler(vel_x, ini_pos_w2n)
    return vel_x, pos_x

# generate model-given pos_y and vel_y values
def pos_vel_y(mod = 'sen'):
    acc_w2e = [0] # straight, west to east
    ini_vel_w2e = 0
    ini_pos_w2e = -4.8
    timestamp_w2e = [0, 80]
    acc_y_w2e = acc_gen(acc_w2e, timestamp_w2e)
    acc_w2n = [0]# left, west to north
    timestamp_w2n = [0, 80]
    ini_vel_w2n = 0
    ini_pos_w2n = -1.6
    acc_y_w2n = acc_gen(acc_w2n, timestamp_w2n)
    acc_w2s = [0]# right, west to south
    timestamp_w2s = [0, 80]
    ini_vel_w2s = 0
    ini_pos_w2s = -8.0
    acc_y_w2s = acc_gen(acc_w2s, timestamp_w2s)

    if mod == 'e':
        vel_y = for_euler(acc_y_w2e, ini_vel_w2e)
        pos_y = for_euler(vel_y, ini_pos_w2e)
    else:
        if mod == 's':
            vel_y = for_euler(acc_y_w2s, ini_vel_w2s)
            pos_y = for_euler(vel_y, ini_pos_w2s)
        else:
            if mod == 'n':
                vel_y = for_euler(acc_y_w2n, ini_vel_w2n)
                pos_y = for_euler(vel_y, ini_pos_w2n)
    return vel_y, pos_y

#acc = [1, 3]
#timestamp = [0, 3, 7]
#ini = 12
#acc = acc_gen(acc, timestamp)
#print(acc)
#vel = for_euler(acc, ini)
#print(vel)
#pos = for_euler(vel, 0)
#print(pos)
#print(pos_vel_x(mod = 'e')[0])
#print(pos_vel_y(mod = 's')[1])

def lane_mod(time, mod='sen'):
    if mod == 's':
        y_pos = -8.0
        #y_pos = -7.0
    else:
        if mod == 'e':
            y_pos = -4.8
            #y_pos = -3.8
        else:
            if mod == 'n':
                y_pos = -1.6
                #y_pos = -0.6
    return y_pos

def vel_mod(time, mod='sen'):
    if time < 20:
        if mod == 'e':
            vel_x = 12.0
        else:
            vel_x = 12.0
    else:
        if time < 48:
            if mod == 'e':
                vel_x = 12.0
                #vel_x = 12.0 - 12.0*(time - 20)/28
            else:
                vel_x = 12.0 - 12.0*(time - 20)/28
        else:
            if time < 53:
                if mod == 'e':
                    vel_x = 12.0
                    #vel_x = 0.0 + 12.0*(time - 48)/5
                else:
                    #vel_x = 0.0
                    vel_x = 0.0 + 12.0*(time - 48)/5
            else:
                if mod == 'e':
                    vel_x = 12.0
                else:
                    vel_x = 0.0
                    #vel_x = 12.0
    return vel_x 

def vel_x_ped(t_sidewalk, t_street, time, mod = 'csg'):
    # -21.6: grass
    # -1.6: sidewalk
    # 8.0: street
    if time < t_sidewalk:
        if mod == 'c':
            vel_x = 0.0
        if mod == 's':
            vel_x = 0.6
        if mod == 'g':
            vel_x = 1.2
    else:
        if time < t_street:
            if mod == 'c':
                vel_x = 0.0
            if mod == 's':
                vel_x = 1.2
            if mod == 'g':
                vel_x = 0.8485
        else:
            if mod == 'c':
                vel_x = 0.0
            if mod == 's':
                vel_x = 0.8485
            if mod == 'g':
                vel_x = 0.6
    return vel_x

def vel_y_ped(t_sidewalk, t_street, time, mod = 'csg'):
    # -21.6: grass
    # -1.6: sidewalk
    # 8.0: street
    if time < t_sidewalk:
        if mod == 'c':
            vel_y = 1.2
        if mod == 's':
            vel_y = 1.03923
        if mod == 'g':
            vel_y = 0.0
    else:
        if time < t_street:
            if mod == 'c':
                vel_y = 1.2
            if mod == 's':
                vel_y = 0.0
            if mod == 'g':
                vel_y = -0.8485
        else:
            if mod == 'c':
                vel_y = 1.2
            if mod == 's':
                vel_y = -0.8485
            if mod == 'g':
                vel_y = -1.03923
    return vel_y

#for i in range(1, 70):
#    print(i, vel_mod(i, mod = 's'))