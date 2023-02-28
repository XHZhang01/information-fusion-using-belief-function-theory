import math
import itertools

"""Iteratively merges opinions from current step"""
def current_step_mix(W, mu_str='mu'):
    W_mix = W[0].copy()
    for w in W[1:]:
        W_mix = DCR([W_mix, w], mu_str=mu_str)
        
    return W_mix

"""Implements Dempster's rule of combination for two"""
def DCR(W, mu_str='mu'):
    allkeys = set().union(*W)
    Wdr = {}
    for k in allkeys:
        if len(str(k))==1 or k is mu_str:
            #print('')
            #print(k)
            b = 0
            for k1, k2 in itertools.product(W[0].keys(), W[1].keys()):
                # if k is not mu_str and (k1 is mu_str or k2 is mu_str):
                #     continue
                if len(k1)>len(k) and len(k2)>len(k):
                    continue
                if k in k1 and k in k2:
                    #print(k1, '-', k2)
                    b += W[0][k1]*W[1][k2]
            Wdr[k] = b
    den = math.fsum(list(Wdr.values())) # normalization factor
    for k in Wdr.keys(): Wdr[k] = Wdr[k]/den
    return Wdr

#mu_str = 'sen'
#W = [{'e': 0.01, 'sn': 0.95, 'sen': 0.04}, {'s': 0.05, 'n': 0.05, 'e': 0.9, 'sen': 0.1}]
#print(current_step_mix(W, mu_str='mu'))