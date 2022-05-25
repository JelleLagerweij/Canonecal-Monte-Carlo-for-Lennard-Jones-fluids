# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:35:59 2022

@author: Jelle
"""

import LJ_Monte_Carlo as mc
import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'  # This enables editing plots as svg's.
plt.close('all')


"""
First lets set our model to the right state. To compute the pressure of methane
at 150K and a density of 358.4, we need the amu mass of methane and the LJ
parameters that are best suited to model to treat CH4 as a single particle.
After which we create an initial configuration and we set the parameters we
will use for the moddeling itself, R_cut and the tail and shift corrections.
The tail and shift corrections are set to True by default.
"""

m_a = 16.04246  # amu
eps = 148  # k_b*T
sig = 3.73  # Angstrom

T = np.linspace(150, 400, 6)
P_L_362 = np.zeros((len(T), 2))
P_G_362 = np.zeros((len(T), 2))
P_L_1000 = np.zeros((len(T), 2))
P_G_1000 = np.zeros((len(T), 2))

for i in range(len(T)):
    n = 362
    N = 691
    rho = 358.4  # kg/m^3
    methane = mc.State(T[i], rho, m_a, eps, sig, N)
    methane.modelCorrections(Rcut=14, Tail=True, Shift=True)
    E, P_L_362[i, :], rad, r = mc.monteCarlo(methane, n, max_step_init=0.5)

    rho = 1.6  # kg/m^3
    methane = mc.State(T[i], rho, m_a, eps, sig, N)
    methane.modelCorrections(Rcut=50, Tail=True, Shift=True)
    E, P_G_362[i, :], rad, r = mc.monteCarlo(methane, n, max_step_init=0.5)

    # n = 1000
    # N = 1000
    # rho = 358.4  # kg/m^3
    # methane = mc.State(T[i], rho, m_a, eps, sig, N)
    # methane.modelCorrections(Tail=True, Shift=True)
    # E, P_L_1000[i, :], rad, r = mc.monteCarlo(methane, n, max_step_init=0.5)

    # rho = 1.6  # kg/m^3
    # methane = mc.State(T[i], rho, m_a, eps, sig, N)
    # methane.modelCorrections(Tail=True, Shift=True)
    # E, P_G_1000[i, :], rad, r = mc.monteCarlo(methane, n, max_step_init=0.5)


T_coolp = np.linspace(150, 400, 250)
P_L_coolp = np.zeros(len(T_coolp))
P_G_coolp = np.zeros(len(T_coolp))
for i in range(len(T_coolp)):
    rho = 358.4  # kg/m^3
    P_L_coolp[i] = PropsSI('P', 'T', T_coolp[i], 'D', rho, 'methane')
    rho = 1.6  # kg/m^3
    P_G_coolp[i] = PropsSI('P', 'T', T_coolp[i], 'D', rho, 'methane')


plt.figure(1)
plt.errorbar(T, P_L_362[:, 0]*1e-5, yerr=P_L_362[:, 1]*1e-5, color='C0',
             label='Liquid N=362', fmt='o', capsize=3)
# plt.errorbar(T, P_L_1000[:, 0]*1e-5, yerr=P_L_1000[:, 1]*1e-5, color='C1',
#               label='Liquid N=1000', fmt='o', capsize=3)
plt.plot(T_coolp, P_L_coolp*1e-5, color='C2', label='Liquid measured')
plt.xlabel('Temperature in \si{\K}')
plt.ylabel('pressure in \si{bar}')
plt.legend()
plt.xlim(145, 405)

plt.figure(2)
plt.errorbar(T, P_G_362[:, 0]*1e-5, yerr=P_G_362[:, 1]*1e-5, color='C0',
             label='Gas Liquid N=362', fmt='o', capsize=3)
# plt.errorbar(T, P_G_1000[:, 0]*1e-5, yerr=P_G_1000[:, 1]*1e-5, color='C1',
#               label='Gas Liquid N=1000', fmt='o', capsize=3)
plt.plot(T_coolp, P_G_coolp*1e-5, color='C2', label='Gas measured')
plt.xlabel('Temperature in \si{\K}')
plt.ylabel('pressure in \si{bar}')
plt.legend()
plt.xlim(145, 405)

T = 150  # Kelvin
N = 500
n = 1000
rho = 358.4  # kg/m^3
methaneL = mc.State(T, rho, m_a, eps, sig, N)
methaneL.modelCorrections(Tail=True, Shift=True)
E, P, rad_L, r_L = mc.monteCarlo(methaneL, n, max_step_init=0.5)

rho = 9.68  # kg/m^3
methaneG = mc.State(T, rho, m_a, eps, sig, N)
methaneG.modelCorrections(Tail=True, Shift=True)
E, P, rad_G, r_G = mc.monteCarlo(methaneG, n, max_step_init=0.5)

plt.figure(3)
plt.errorbar(r_L*1e10, rad_L[:, 0], yerr=rad_L[:, 1], color='C0', label='rho=\SI{357.4}{\kg\per\m^3')
plt.errorbar(r_G*1e10, rad_G[:, 0], yerr=rad_G[:, 1], color='C1', label='rho=\SI{9.68}{\kg\per\m^3')
plt.xlabel('radial distance in \si{\angstrom}')
plt.ylabel('occurence')
plt.xlim(0, 20)
plt.legend()
