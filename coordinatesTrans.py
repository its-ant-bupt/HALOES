import numpy as np


def coTrans(pvec, tvec):
    """
    :param pvec: 新的坐标系的原点在原坐标系下的坐标（x,y,theta）
    :param tvec: 要转换到新坐标系下的点在原坐标系下的坐标(x1,y1,theta1)
    :return: tvec 在以 pvec 为原点的坐标系下的坐标
    """
    alpha = np.arctan2((tvec[1]-pvec[1]), (tvec[0]-pvec[0]))
    dis = np.hypot((tvec[1]-pvec[1]), (tvec[0]-pvec[0]))
    delta = alpha - pvec[2]  # delta = alpha - End_theta
    new_x = dis * np.cos(delta)
    new_y = dis * np.sin(delta)
    new_theta = mod2pi(tvec[2] - pvec[2])

    return [new_x, new_y, new_theta]


def mod2pi(theta):
    """
    将车辆的角度转换到(-pi,pi)之间
    :param theta: 角度值 rad
    :return: newtheta
    """

    v = theta % (2*np.pi)
    if v < -1*np.pi:
        v = v + 2*np.pi
    elif v > np.pi:
        v = v - 2*np.pi
    return v
