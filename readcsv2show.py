import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from case import Case


def readState(path):
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    path_t = data[:, 0]
    path_x = data[:, 1]
    path_y = data[:, 2]
    path_v = data[:, 3]
    path_yaw = data[:, 4]
    path_steer = data[:, 5]
    return path_t, path_x, path_y, path_v, path_yaw, path_steer


def readControl(path):
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    path_a = data[:, 1]
    path_steer_rate = data[:, 2]
    return path_a, path_steer_rate


def showVehicleTraj(path, case, minx, miny, maxx, maxy, path_num, exp_name):
    path_t, path_x, path_y, path_v, path_yaw, path_steer = readState(path)
    plt.figure()
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_axisbelow(True)
    plt.title('Trajectory of Case %d' % path_num)
    plt.grid(linewidth=0.2)
    plt.xlabel('X / m', fontsize=14)
    plt.ylabel('Y / m', fontsize=14)
    for j in range(0, case.obs_num):
        plt.fill(case.obs[j][:, 0], case.obs[j][:, 1], facecolor='k', alpha=0.5)

    temp = case.vehicle.create_polygon(case.x0, case.y0, case.theta0)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='green')
    temp = case.vehicle.create_polygon(case.xf, case.yf, case.thetaf)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='red')

    for i in range(len(path_x)):
        temp = case.vehicle.create_polygon(path_x[i], path_y[i], path_yaw[i])
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.15, color='blue')
    plt.plot(path_x, path_y, color='red', linewidth=0.1)
    plt.savefig("./Result/{}-traj{}.svg".format(exp_name, path_num))
    plt.show()


def showStateControl(path_state, path_control, path_num, exp_name):
    path_t, path_x, path_y, path_v, path_yaw, path_steer = readState(path_state)
    path_a, path_steer_rate = readControl(path_control)
    sampleT = path_t[1] - path_t[0]
    fig2, ax2 = plt.subplots(4)
    plt.subplots_adjust(hspace=0.35)
    t_v = [sampleT * k for k in range(len(path_v))]
    t_a = [sampleT * k for k in range(len(path_a))]
    t_steer = [sampleT * k for k in range(len(path_steer))]
    t_steer_rate = [sampleT * k for k in range(len(path_steer_rate))]
    ax2[0].plot(t_v, path_v, label='v-t')
    ax2[1].plot(t_a, path_a, label='a-t')
    ax2[2].plot(t_steer, path_steer, label='steer-t')
    ax2[3].plot(t_steer_rate, path_steer_rate, label='steer-rate-t')
    ax2[0].legend()
    ax2[1].legend()
    ax2[2].legend()
    ax2[3].legend()
    plt.savefig("./Result/{}-kina-{}.svg".format(exp_name, path_num))


if __name__ == '__main__':
    case = Case.read('BenchmarkCases/Case%d.csv' % 7)
    showVehicleTraj("./Result/5-state-result-state-7.csv", case, minx=-16.5, maxx=-15.9, miny=-2.6, maxy=-2,
                    path_num=7, exp_name="5-state-obca-zoomzoom")
