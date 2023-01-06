import numpy as np
import math


class Vehicle:
    def __init__(self):
        self.lw = 2.8  # wheelbase
        self.lf = 0.96  # front hang length
        self.lr = 0.929  # rear hang length
        self.lb = 1.942  # width
        self.MAX_STEER = 0.75  # 方向盘最大转角
        self.MIN_CIRCLE = self.lw / math.tan(self.MAX_STEER)  # 车辆最小转角
        self.MAX_THETA = np.pi
        self.MIN_THETA = -np.pi
        self.MAX_A = 1
        self.MIN_A = -1
        self.MAX_V = 2.5
        self.MIN_V = -2.5
        self.MAX_OMEGA = 0.5
        self.MIN_OMEGA = -0.5

    def create_polygon(self, x, y, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        points = np.array([
            [-self.lr, -self.lb / 2, 1],
            [self.lf + self.lw, -self.lb / 2, 1],
            [self.lf + self.lw, self.lb / 2, 1],
            [-self.lr, self.lb / 2, 1],
            [-self.lr, -self.lb / 2, 1],
        ]).dot(np.array([
            [cos_theta, -sin_theta, x],
            [sin_theta, cos_theta, y],
            [0, 0, 1]
        ]).transpose())
        return points[:, 0:2]


class Path(object):
    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw


class OBCAPath(object):
    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading
        self.a = 0
        self.v = 0
        self.steer = 0
