import numpy as np
import casadi

Nx = 9
Ny = 2
Nu = 2
Nw = Nx
Nv = Ny
Np = 1

w0 = np.zeros(Nx)
v0 = np.zeros(Ny)

# Lego parameters
# http://www.mathworks.com/matlabcentral/fileexchange/35206-simulink-support-package-for-lego-mindstorms-nxt-hardware--r2012a-/content/lego/legodemos/lego_selfbalance_plant.m
# http://www.nt.ntnu.no/users/skoge/prost/proceedings/ecc-2013/data/papers/0959.pdf
# DC motor state space system model
# http://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=SystemModeling

wheel_radius = 0.0216  # 0.028 # (43.2*0.5)/1000.0   # Wheel radius [m]
wheel_distance = 0.17  # 0.105 # 105.0/1000.0       # Distance between wheels [m]
fm = 0.0022  # motor viscous friction constant
Jm = 1e-5  # DC motor inertia moment [kgm^2]
Rm = 6.69  # DC motor resistance []
Kb = 0.468  # DC motor back EMF constant [Vsec/rad]
Kt = 0.317  # DC motor torque constant [Nm/A]
Gu = 1E-2  # PWM gain factor
Vb0 = 8.00  # V Power Supply voltage
Vo = 0.625  # V Power Supply offset
mu = 1.089  # Power Supply gain factor =
Vo_l = 0.68
Vo_r = 0.68
mu_l = 1.04
mu_r = 0.99
L = 1.0


def f(x, u, Vb=Vb0):
    """
    Model for a differential drive mobile robot.
    x[0] -> x
    x[1] -> y
    x[2] -> \psi
    x[3] -> \theta_l
    x[4] -> \omega_l
    x[5] -> i_l
    x[6] -> \theta_r
    x[7] -> \omega_r
    x[8] -> i_r
    """
    # wR = 43.2*0.5 # Wheel radius [mm]
    # wB = 82.0 # Distance between wheels [mm]
    # fm = 0.0022 # motor viscous friction constant
    # Jm = 1e-5           # DC motor inertia moment [kgm^2]
    # Rm = 6.69           # DC motor resistance []
    # Kb = 0.468          # DC motor back EMF constant [Vsec/rad]
    # Kt = 0.317          # DC motor torque constant [Nm/A]
    # Gu = 1E-2 #PWM gain factor
    # Vb = 8.00 #V Power Supply voltage
    # Vo = 0.625 #V Power Supply offset
    # mu = 1.089 #Power Supply gain factor
    # L = 1.0
    # if len(x.shape)>1:
    #    x = x.ravel()
    # if len(u.shape)>1:
    #    u = u.ravel()

    # return np.array([0.5 * wheel_radius * (x[4] + x[7]) * np.cos(x[2]),
    #                  0.5 * wheel_radius * (x[4] + x[7]) * np.sin(x[2]),
    #                  (wheel_radius / wheel_distance) * (x[7] - x[4]),
    #                  x[4],
    #                  -(fm / Jm) * x[4] + (Kt / Jm) * x[5],
    #                  -(Kb / L) * x[4] - (Rm / L) * x[5] + ((Gu * (x[9] * Vb - Vo_l)) / L) * u[0],
    #                  x[7],
    #                  -(fm / Jm) * x[7] + (Kt / Jm) * x[8],
    #                  -(Kb / L) * x[7] - (Rm / L) * x[8] + ((Gu * (x[10] * Vb - Vo_r)) / L) * u[1],
    #                  0,
    #                  0])

    return np.array([0.5*wheel_radius*(x[4]+x[7])*np.cos(x[2]),
                     0.5*wheel_radius*(x[4]+x[7])*np.sin(x[2]),
                     (wheel_radius/wheel_distance)*(x[7]-x[4]),
                     x[4],
                     -(fm/Jm)*x[4] + (Kt/Jm)*x[5],
                     -(Kb/L)* x[4] - (Rm/L)* x[5] + ((Gu*(mu*Vb-Vo_l))/L)*u[0],
                     x[7],
                     -(fm/Jm)*x[7] + (Kt/Jm)*x[8],
                     -(Kb/L)* x[7] - (Rm/L)* x[8] + ((Gu*(mu*Vb-Vo_r))/L)*u[1] ])


def h(x, nxt_sim=False):
    y = np.array([x[3], x[6]]).ravel()
    if nxt_sim:
        y_deg = np.rad2deg(y)
        y_deg = np.around(y_deg, decimals=0)
        return np.deg2rad(y_deg)
    else:
        return y


def differential_drive(x, u):
    """

    :param x: current state
    :param u: vector of control signals, u[0] is the linear velocity and u[1] is the angular velocity
    :return: state of the system given the current state and the desired control signals
    """
    v = u[0]
    w = u[1]
    return np.array([v*np.cos(x[2]), v*np.sin(x[2]), w])


def vw2vlvr(v, w, b=wheel_distance, r=wheel_radius):
    Dmat = np.array([[0.5*r, 0.5*r], [-r/b, r/b]])
    return np.linalg.inv(Dmat).dot(np.array([v,w]))


def vw2vlvr_1(v, w, b=wheel_distance, r=wheel_radius):
    vl = (2.0 * v - w * b) / (2.0 * r)
    vr = (2.0 * v + w * b) / (2.0 * r)
    return np.array([vl,vr]).flatten()


def v2pwm(v, Vb=Vb0):
    # Esta conversion deberia saturar entre -100 y 100.
    u = v/(Gu * (mu * Vb - Vo))
    return u


def dcmotor_model(x, u=0.0, Vb=8.00):
    """
    Model for a Lego NXT Mindstorms DC motor.
    x[0] -> DC motor position \theta
    x[1] -> DC motor speed \omega
    x[2] -> DC motor armature current i
    """
    v = Gu * (mu * Vb - Vo) * u

    return np.array([x[1],
                     -(fm / Jm) * x[1] + (Kt / Jm) * x[2],
                     -(Kb / L) * x[1] - (Rm / L) * x[2] + v / L
                     ])


def dcmotor_measurement(x, nxt_sim=False):
    if nxt_sim:
        x_deg = np.rad2deg(x[0])
        x_deg = np.around(x_deg, decimals=0)
        return np.deg2rad(x_deg)
    else:
        return np.array([x[0]]).flatten()
