# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import rk4
import pid
import model_lego

Nx = 3
Ny = 1
Nu = 2

dt = 0.15
x0 = np.zeros((Nx,))
# Vb0 = 8.0
Nsim = 200
tsim = np.arange(0,Nsim)*dt

vel_0 = 0.5

v_max =  0.5
v_min = -0.5
w_max =  2.5
w_min = -2.5

xk = np.zeros((Nx,Nsim+1))
xk[:,0] = x0
uk = np.zeros((Nu,Nsim))

# Velocity control
Kp_v = 0.5
Ki_v = 0.0
Kd_v = 0.0
Kp_w = 2.0
Ki_w = 0.0
Kd_w = 0.0
pid_v = pid.PID(Kp_v, Ki_v, Kd_v, dt, v_max, v_min)
pid_w = pid.PID(Kp_w, Ki_w, Kd_w, dt, w_max, w_min)

x_target = np.array([-2.0, -2.0, 0.0])
w_target_prev = 0.0

for k in xrange(Nsim):
    w_target = np.arctan2((x_target[1] - xk[1, k]), (x_target[0] - xk[0, k]))
    w_target = np.unwrap([w_target_prev, w_target])[-1]
    d_target = np.linalg.norm(x_target[:2]- xk[:2, k])

    uk[0,k] = pid_v.get_output(d_target,0.0)
    uk[1,k] = pid_w.get_output(w_target, xk[2, k])

    print model_lego.vw2vlvr(*uk[:,k])

    xk[:, k+1] = rk4.rk4(model_lego.differential_drive, xk[:,k], [uk[:,k]], dt, 50)
    w_target_prev = w_target


plt_titles = ("$x [m]$","$y [m]$","$\psi [rad]$","$v$", "$\omega$")
f, axarr = plt.subplots(Nx+Nu,1)

for i in xrange(Nx):
    axarr[i].plot(tsim,xk[i,:-1],".-", color='k')
    axarr[i].set_title(plt_titles[i])
    axarr[i].grid()

for i in xrange(Nu):
    axarr[Nx+i].step(tsim,uk[i,:],"k.-", where='post')
    axarr[Nx+i].set_title(plt_titles[Nx+i])
    axarr[Nx+i].grid()

plt.tight_layout()

plt.figure()
plt.plot(xk[0,:],xk[1,:])
plt.plot(x_target[0], x_target[1], 'ro')
plt.grid()