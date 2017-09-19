# -*- coding: utf-8 -*-

import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, delta, out_max=np.inf, out_min=-np.inf):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.out_max = out_max
        self.out_min = out_min
        self.delta = delta
        self.integral = 0.0
        self.prev_err = 0.0
                
    def get_output(self, reference, measurement):
        current_err = reference - measurement
        self.integral = self.integral + current_err*self.delta
        derivative = (current_err - self.prev_err)/self.delta
        self.prev_err = current_err;
        output = self.Kp*current_err + self.Ki*self.integral + self.Kd*derivative
        if output>self.out_max:
            output = self.out_max
        if output<self.out_min:
            output = self.out_min
        return output
