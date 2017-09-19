import matplotlib.pyplot as plt

import model_lego
import numpy as np

import pid
import rk4

Nx = 9
Ny = 2
Nu = 2


dt = 0.1
x0 = np.zeros((Nx,))
Nsim = 500
tsim = np.arange(0,Nsim)*dt

v_max =  0.5
v_min = -0.5
w_max =  0.5
w_min = -0.5

xk = np.zeros((Nx,Nsim+1))
xk[:,0] = x0
uk = np.zeros((Nu,Nsim))

# Velocity control
Kp_v =  1.0
Ki_v =  0.0
Kd_v =  0.0
Kp_w =  2.0
Ki_w =  0.0
Kd_w =  0.0
pid_v = pid.PID(Kp_v, Ki_v, Kd_v, dt, v_max, v_min)
pid_w = pid.PID(Kp_w, Ki_w, Kd_w, dt, w_max, w_min)

x_target = np.array([2.0, 2.0, 0.0])
w_target_prev = 0.0

measurement_prev = np.zeros((Ny,))
xhat = np.zeros((Nx, Nsim+1))

for k in xrange(Nsim):
    measurement = model_lego.h(xk[:, k], nxt_sim=True)

    travelled_distance = model_lego.wheel_radius * (measurement - measurement_prev) #*dt/dt

    xhat[0, k + 1] = xhat[0, k] + 0.5 * (travelled_distance[0] + travelled_distance[1]) * np.cos(xk[2, k])
    xhat[1, k + 1] = xhat[1, k] + 0.5 * (travelled_distance[0] + travelled_distance[1]) * np.sin(xk[2, k])
    xhat[2, k + 1] = xhat[2, k] + (travelled_distance[1] - travelled_distance[0]) / model_lego.wheel_distance

    w_target = np.arctan2((x_target[1] - xk[1, k]), (x_target[0] - xk[0, k]))
    w_target = np.unwrap([w_target_prev, w_target])[-1]

    d_target = np.linalg.norm(x_target[:2]- xk[:2, k])

    _vk = pid_v.get_output(d_target, 0.0)
    _wk = pid_w.get_output(w_target, xk[2, k])

    _vlvr = model_lego.vw2vlvr_1(_vk, _wk)
    uk[:,k] = model_lego.v2pwm(_vlvr)

    print uk[:,k]
    xk[:, k+1] = rk4.rk4(model_lego.f, xk[:, k], [uk[:, k]], dt, 50)
    w_target_prev = w_target
    measurement_prev = measurement


plt_titles = ("$x [m]$","$y [m]$","$\psi [rad]$","$u_l$", "$u_r$")
Nx = 3
f, axarr = plt.subplots(Nx+Nu,1)
for i in xrange(Nx):
    axarr[i].plot(tsim,xk[i,:-1],".-", color='k')
    axarr[i].plot(tsim, xhat[i, :-1], "o-", color='r')
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