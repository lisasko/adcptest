# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:36:04 2022

@author: blais
"""
import math
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import log10, floor

def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
   
def interpolation(data1, data2, data1_interp_value, style='linear'):
    funcs = {'linear': lambda x, a, b: a * x + b,
             'power': lambda x, a, b: a * x**b,
             'power_rectangular': lambda x, a: a * x**0.1,
             'power_triangular': lambda x, a: a * x**0.41}
    valid = ~(np.isnan(data2) | np.isinf(data2))
    if np.isnan(data2[valid]).all():
        value = np.nan
    elif len(data2[valid])==1:
            value = data2[valid]
    else:
        try:
            popt, _ = curve_fit(funcs[style], data1[valid], data2[valid], maxfev=1000)
            value = funcs[style](data1_interp_value, *popt)
        except Exception:
                value = np.nan
    return value

def round_it(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)