from case import Case
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
from tqdm import tqdm
import threading
import queue



def trianglePoly(x1, y1, x2, y2, x3, y3):
    # edgeLen1 = ca.sqrt(ca.power(x1 - x2, 2) + ca.power(y1 - y2, 2))
    # edgeLen2 = ca.sqrt(ca.power(x1 - x3, 2) + ca.power(y1 - y3, 2))
    # edgeLen3 = ca.sqrt(ca.power(x2 - x3, 2) + ca.power(y2 - y3, 2))
    # s = (edgeLen1 + edgeLen2 + edgeLen3) / 2
    # shapeArea = ca.sqrt(s * (s - edgeLen1) * (s - edgeLen2) * (s - edgeLen3))
    # shapeArea = (s * (s - edgeLen1) * (s - edgeLen2) * (s - edgeLen3))
    _x = np.array([x1, x2, x3])
    _y = np.array([y1, y2, y3])
    shapeArea = 0.5 * np.abs(np.dot(_x, np.roll(_y, 1)) - np.dot(_y, np.roll(_x, 1)))
    return shapeArea

def component_polygon_area(poly):
    """Compute the area of a component of a polygon.
    Args:
        x (ndarray): x coordinates of the component
        y (ndarray): y coordinates of the component

    Return:
        float: the are of the component
    """
    _x = poly[:, 0]
    _y = poly[:, 1]
    return 0.5 * np.abs(
        # 注意这里的np.dot表示一维向量相乘
        np.dot(_x, np.roll(_y, 1)) - np.dot(_y, np.roll(_x, 1)))  # np.roll 意即“滚动”，类似移位操作

def VehObsArea(vehList, obsList):
    # vehList: [4*2]
    # obsList: [n*2] n:障碍物点包围
    obsArea = component_polygon_area(np.array(obsList))
    obsList.append(obsList[0])
    exArea = [0] * 4
    for v_i in range(4):
        for j in range(len(obsList) - 1):
            exArea[v_i] += trianglePoly(vehList[v_i][0], vehList[v_i][1],
                                        obsList[j][0], obsList[j][1],
                                        obsList[j + 1][0], obsList[j + 1][1])

        exArea[v_i] = exArea[v_i] - obsArea
    return exArea

def ObsVehArea(obsVec, veh):
    vehArea = 1.942 * (2.8+0.96+0.929)
    veh.append(veh[0])
    exArea = 0
    for i in range(len(veh)-1):
        exArea += trianglePoly(obsVec[0], obsVec[1],
                               veh[i][0], veh[i][1],
                               veh[i+1][0], veh[i+1][1])
    exArea -= vehArea
    return exArea


def plot_OPF(case):
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    yaw_min = -np.pi
    yaw_max = np.pi
    obsList = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obsList.append(obs)
    power_num = []
    x_num = []
    y_num = []
    yaw_num = []
    for x in tqdm(np.arange(x_min, x_max, 0.1)):
        for y in np.arange(y_min, y_max, 0.1):
            for yaw in np.arange(yaw_min, yaw_max, np.pi/30):
                x_num.append(x)
                y_num.append(y)
                yaw_num.append(yaw)
                temp = case.vehicle.create_polygon(x, y, yaw)
                veh = [[temp[0, 0], temp[0, 1]], [temp[1, 0], temp[1, 1]], [temp[2, 0], temp[2, 1]],
                       [temp[3, 0], temp[3, 1]]]
                vehObsAreas = []
                for obs_i in range(len(obsList)):
                    vehObsArea = VehObsArea(veh, obsList[obs_i])
                    for area_i in range(len(vehObsArea)):
                        vehObsAreas.append(vehObsArea[area_i])
                # obsVehArea = ObsVehArea(closestVec, veh)
                obsPower = 1.56
                valueArea = 0
                # valueArea += ca.power(1 / (obsVehArea + 0.0000001), obsPower)
                for area_i in range(len(vehObsAreas)):
                    valueArea += np.power(1 / (vehObsAreas[area_i] + 0.0000001), obsPower)
                power_num.append(valueArea)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X, Y = np.array(x_num), np.array(y_num)
    Z = np.array(yaw_num)
    power_num = np.array(power_num)
    color_map = cm.ScalarMappable()
    color_map.set_array(power_num)


    ax.scatter(X, Y, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Yaw/rad')
    plt.colorbar(color_map, ax=[ax], location='left')
    plt.savefig("OPF.svg")
    plt.show()

def plot_OPF_onepiece(case):
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    yaw_min = -np.pi
    yaw_max = np.pi
    obsList = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obsList.append(obs)
    power_num = []
    x_num = []
    y_num = []
    yaw_num = []
    yaw = case.thetaf
    for x in tqdm(np.arange(x_min, x_max, 0.5)):
        for y in np.arange(y_min, y_max, 0.5):
            x_num.append(x)
            y_num.append(y)
            yaw_num.append(yaw)
            temp = case.vehicle.create_polygon(x, y, yaw)
            veh = [[temp[0, 0], temp[0, 1]], [temp[1, 0], temp[1, 1]], [temp[2, 0], temp[2, 1]],
                   [temp[3, 0], temp[3, 1]]]
            vehObsAreas = []
            for obs_i in range(len(obsList)):
                vehObsArea = VehObsArea(veh, obsList[obs_i])
                for area_i in range(len(vehObsArea)):
                    vehObsAreas.append(vehObsArea[area_i])
            # obsVehArea = ObsVehArea(closestVec, veh)
            obsPower = 1.56
            valueArea = 0
            # valueArea += ca.power(1 / (obsVehArea + 0.0000001), obsPower)
            for area_i in range(len(vehObsAreas)):
                valueArea += np.power(1 / (vehObsAreas[area_i] + 0.0000001), obsPower)
            power_num.append(valueArea)
    fig = plt.figure()
    ax = plt.axes()
    plt.xlim(case.xmin, case.xmax)
    plt.ylim(case.ymin, case.ymax)

    X, Y = np.array(x_num), np.array(y_num)
    power_num = np.array(power_num)
    color_map = cm.ScalarMappable()
    color_map.set_array(power_num)


    ax.scatter(X, Y, alpha=0.3, c=np.array(power_num), s=10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    plt.colorbar(color_map, ax=[ax], location='right')
    plt.savefig("OPF_onepiece.svg")
    plt.show()


class PlotOPF(threading.Thread):
    def __init__(self, name, queue, res, obs, pbar):
        threading.Thread.__init__(self)
        self.queue = queue
        self.res = res
        self.obsList = obs
        self.pbar = pbar
        self.start()

    def run(self):
        while True:
            if self.queue.empty():
                break
            self.pbar.update(1)
            tmp = self.queue.get()
            x = tmp[0]
            y = tmp[1]
            yaw = tmp[2]
            temp = case.vehicle.create_polygon(x, y, yaw)
            veh = [[temp[0, 0], temp[0, 1]], [temp[1, 0], temp[1, 1]], [temp[2, 0], temp[2, 1]],
                   [temp[3, 0], temp[3, 1]]]
            vehObsAreas = []
            for obs_i in range(len(self.obsList)):
                vehObsArea = VehObsArea(veh, self.obsList[obs_i])
                for area_i in range(len(vehObsArea)):
                    vehObsAreas.append(vehObsArea[area_i])
            # obsVehArea = ObsVehArea(closestVec, veh)
            obsPower = 1.56
            valueArea = 0
            # valueArea += ca.power(1 / (obsVehArea + 0.0000001), obsPower)
            for area_i in range(len(vehObsAreas)):
                valueArea += np.power(1 / (vehObsAreas[area_i] + 0.0000001), obsPower)
            self.res.append([x, y, yaw, valueArea])
            self.queue.task_done()


def ThreadPlotPiece(case, num=8):
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    yaw_min = -np.pi
    yaw_max = np.pi
    obsList = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obsList.append(obs)
    xyzQueue = queue.Queue()
    res = []
    yaw = case.thetaf
    for x in np.arange(x_min, x_max, 0.25):
        for y in np.arange(y_min, y_max, 0.25):
            # for yaw in np.arange(yaw_min, yaw_max, np.pi / 30):
            xyzQueue.put([x, y, yaw])
    pbar = tqdm(total=xyzQueue.qsize())

    for i in range(num):
        threadNum = 'Thread' + str(i)
        PlotOPF(threadNum, xyzQueue, res, obsList, pbar)

    xyzQueue.join()
    pbar.close()

    x_num = [tmp[0] for tmp in res]
    y_num = [tmp[1] for tmp in res]
    yaw_num = [tmp[2] for tmp in res]
    power_num = [tmp[3] for tmp in res]

    X, Y = np.array(x_num), np.array(y_num)
    Z = np.array(yaw_num)
    power_num = np.array(power_num)
    color_map = cm.ScalarMappable()
    color_map.set_array(power_num)
    fig = plt.figure()
    ax = plt.axes()
#    ax.scatter(X, Y, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.scatter(X, Y, alpha=0.3, c=np.array(power_num), s=10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    plt.colorbar(color_map, ax=[ax], location='right')
    plt.savefig("OPF_1207_piece.svg")
    plt.show()


def ThreadPlotPiece_xyaw(case, num=8):
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    yaw_min = -np.pi
    yaw_max = np.pi
    obsList = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obsList.append(obs)
    xyzQueue = queue.Queue()
    res = []
    y = case.yf
    for x in np.arange(x_min, x_max, 0.25):
        for yaw in np.arange(yaw_min, yaw_max, np.pi / 30):
            xyzQueue.put([x, y, yaw])
    pbar = tqdm(total=xyzQueue.qsize())

    for i in range(num):
        threadNum = 'Thread' + str(i)
        PlotOPF(threadNum, xyzQueue, res, obsList, pbar)

    xyzQueue.join()
    pbar.close()

    x_num = [tmp[0] for tmp in res]
    y_num = [tmp[1] for tmp in res]
    yaw_num = [tmp[2] for tmp in res]
    power_num = [tmp[3] for tmp in res]

    X, Y = np.array(x_num), np.array(y_num)
    Z = np.array(yaw_num)
    power_num = np.array(power_num)
    color_map = cm.ScalarMappable()
    color_map.set_array(power_num)
    fig = plt.figure()
    ax = plt.axes()
#    ax.scatter(X, Y, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.scatter(X, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Yaw/rad')
    plt.colorbar(color_map, ax=[ax], location='right')
    plt.savefig("OPF_1207_piece_xyaw.svg")
    plt.show()


def ThreadPlotPiece_yyaw(case, num=8):
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    yaw_min = -np.pi
    yaw_max = np.pi
    obsList = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obsList.append(obs)
    xyzQueue = queue.Queue()
    res = []
    x = case.xf
    for y in np.arange(y_min, y_max, 0.25):
        for yaw in np.arange(yaw_min, yaw_max, np.pi / 30):
            xyzQueue.put([x, y, yaw])
    pbar = tqdm(total=xyzQueue.qsize())

    for i in range(num):
        threadNum = 'Thread' + str(i)
        PlotOPF(threadNum, xyzQueue, res, obsList, pbar)

    xyzQueue.join()
    pbar.close()

    x_num = [tmp[0] for tmp in res]
    y_num = [tmp[1] for tmp in res]
    yaw_num = [tmp[2] for tmp in res]
    power_num = [tmp[3] for tmp in res]

    X, Y = np.array(x_num), np.array(y_num)
    Z = np.array(yaw_num)
    power_num = np.array(power_num)
    color_map = cm.ScalarMappable()
    color_map.set_array(power_num)
    fig = plt.figure()
    ax = plt.axes()
#    ax.scatter(X, Y, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.scatter(Y, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.set_xlabel('Y/m')
    ax.set_ylabel('Yaw/rad')
    plt.colorbar(color_map, ax=[ax], location='right')
    plt.savefig("OPF_1207_piece_yyaw.svg")
    plt.show()


def ThreadPlot(case, num=16):
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    yaw_min = -np.pi
    yaw_max = np.pi
    obsList = []
    for obs_i in range(len(case.obs)):
        obs = list(case.obs[obs_i])
        obsList.append(obs)
    xyzQueue = queue.Queue()
    res = []
    # yaw = case.thetaf
    for x in np.arange(x_min, x_max, 1):
        for y in np.arange(y_min, y_max, 1):
            for yaw in np.arange(yaw_min, yaw_max, np.pi / 5):
                xyzQueue.put([x, y, yaw])

    pbar = tqdm(total=xyzQueue.qsize())

    for i in range(num):
        threadNum = 'Thread' + str(i)
        PlotOPF(threadNum, xyzQueue, res, obsList, pbar)

    xyzQueue.join()
    pbar.close()

    x_num = [tmp[0] for tmp in res]
    y_num = [tmp[1] for tmp in res]
    yaw_num = [tmp[2] for tmp in res]
    power_num = [tmp[3] for tmp in res]

    X, Y = np.array(x_num), np.array(y_num)
    Z = np.array(yaw_num)
    power_num = np.array(power_num)
    color_map = cm.ScalarMappable()
    color_map.set_array(power_num)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, alpha=0.3, c=np.array(power_num), s=10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Yaw/rad')
    plt.colorbar(color_map, ax=[ax], location='left')
    plt.savefig("OPF_1203_rough.svg")
    plt.show()


if __name__ == '__main__':
    case = Case.read('BenchmarkCases/Case%d.csv' % 7)
    ThreadPlotPiece(case)
