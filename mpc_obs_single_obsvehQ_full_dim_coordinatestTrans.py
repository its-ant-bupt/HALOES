#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
import csv
from coordinatesTrans import coTrans
from case import Case
from Vehicle import Vehicle, Path, OBCAPath
from Show import show
from quadraticOBCA import quadraticPath
from pyobca.search import *
from saveCsv import saveCsv


def trianglePoly(x1, y1, x2, y2, x3, y3):
    # edgeLen1 = ca.sqrt(ca.power(x1 - x2, 2) + ca.power(y1 - y2, 2))
    # edgeLen2 = ca.sqrt(ca.power(x1 - x3, 2) + ca.power(y1 - y3, 2))
    # edgeLen3 = ca.sqrt(ca.power(x2 - x3, 2) + ca.power(y2 - y3, 2))
    # s = (edgeLen1 + edgeLen2 + edgeLen3) / 2
    # shapeArea = ca.sqrt(s * (s - edgeLen1) * (s - edgeLen2) * (s - edgeLen3))
    # shapeArea = (s * (s - edgeLen1) * (s - edgeLen2) * (s - edgeLen3))
    _x = np.array([x1, x2, x3])
    _y = np.array([y1, y2, y3])
    shapeArea = 0.5 * ca.fabs(np.dot(_x, np.roll(_y, 1)) - np.dot(_y, np.roll(_x, 1)))
    return shapeArea


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
    return 0.5 * ca.fabs(
        # 注意这里的np.dot表示一维向量相乘
        np.dot(_x, np.roll(_y, 1)) - np.dot(_y, np.roll(_x, 1)))  # np.roll 意即“滚动”，类似移位操作


def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t, st, u_end.T


def transCoordinateObs(EndVec, obsList):
    newObsList = []
    for i in range(len(obsList)):
        tvec = [obsList[i][0], obsList[i][1], 0]
        newObs = coTrans(EndVec, tvec)
        newObsList.append([newObs[0], newObs[1]])
    return newObsList


def findClosestVec(VehVec, obsLists):
    dis = np.inf
    closest = []
    for i in range(len(obsLists)):
        for j in range(len(obsLists[i])):
            if np.hypot((obsLists[i][j][0]-VehVec[0]), (obsLists[i][j][1]-VehVec[1])) < dis:
                dis = np.hypot((obsLists[i][j][0]-VehVec[0]), (obsLists[i][j][1]-VehVec[1]))
                closest = [obsLists[i][j][0], obsLists[i][j][1]]
    return closest


if __name__ == '__main__':
    case = Case.read('BenchmarkCases/Case%d.csv' % 7)
    wheelbase = 2.8
    T = 0.1  # sampling time [s]
    N = 20  # prediction horizon
    x_min = -10
    x_max = 10
    y_min = -10
    y_max = 10
    v_max = 2.5
    v_min = -2.5
    theta_min = -np.pi
    theta_max = np.pi
    delta_min = -0.75
    delta_max = 0.75
    a_max = 1
    a_min = -1  # 加速度最小值 m/s2
    omega_max = 0.5  # 方向盘最大转向速率 rad/s
    omega_min = -0.5

    # 车辆状态参数
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    v = ca.SX.sym('v')
    theta = ca.SX.sym('theta')  # 车体方向角度
    delta = ca.SX.sym('delta')  # 车辆的方向盘转角
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, v)
    states = ca.vertcat(states, theta)
    states = ca.vertcat(states, delta)
    n_states = states.size()[0]

    # 车辆控制参数
    a = ca.SX.sym('a')
    omega = ca.SX.sym('omega')  # 方向盘速率
    controls = ca.vertcat(a, omega)
    n_controls = controls.size()[0]

    # 控制器输入
    # rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
    rhs = ca.vertcat(rhs, a)
    rhs = ca.vertcat(rhs, v * ca.tan(delta) / wheelbase)
    rhs = ca.vertcat(rhs, omega)

    # function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N + 1)
    P = ca.SX.sym('P', n_states + n_states)

    # define
    X[:, 0] = P[:5]  # initial condiction

    # define the relationship within the horizon
    for i in range(N):
        f_value = f(X[:, i], U[:, i])
        X[:, i + 1] = X[:, i] + f_value * T

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    Sf = np.array([[50000.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 50000.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 100000.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])

    Q = np.array([[1000.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 5000.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 2.0]])

    R = np.array([[0.0, 0.0], [0.0, 0.00]])

    # cost function
    obj = 0  # cost
    obj = obj + (X[:, N] - P[5:]).T @ Sf @ (X[:, N] - P[5:])
    obj = obj + (X[:, N - 1] - P[5:]).T @ Q @ (X[:, N - 1] - P[5:])
    for i in range(N - 1):
        # new type to calculate the matrix multiplication
        obj = obj + (X[:, i] - P[5:]).T @ Q @ (X[:, i] - P[5:]) + (U[:, i + 1] - U[:, i]).T @ R @ (
                    U[:, i + 1] - U[:, i])

    endvec = [-16.31840796, -2.263681592, 1.061089133]
    startvec = [-11.29353234, 1.069651741, 1.015800599]
    obca_start = startvec
    # obs cost
    obs = [[-25.03567043, -15.8687107], [-17.71684521, -2.775399527], [-16.02169786, -3.722943432],
           [-23.34052308, -16.8162546]]

    obs1 = [[-15.18501961, 1.754013251], [-7.866194392, 14.84732442], [-6.171047039, 13.89978052],
            [-13.48987226, 0.806469346]]

    # obs2 = [[-14.10682142, 3.958046312], [-18.59643337, -3.839700761], [-18.75396361, -3.760935639],
    #         [-13.16163996, 5.809026678]]
    obs2 = [[-14.10682142, 3.958046312], [-18.59643337, -3.839700761], [-22, -2.5],
            [-17.5, 7.5]]
    obstacles = [obs, obs1, obs2]
    closestVec = findClosestVec([startvec[0], startvec[1]], [obs, obs1, obs2])
    # print(closestVec)
    closestVec = coTrans(endvec, [closestVec[0], closestVec[1], 0])
    obs = transCoordinateObs(endvec, obs)
    obs1 = transCoordinateObs(endvec, obs1)
    obs2 = transCoordinateObs(endvec, obs2)

    for i in range(N + 1):
        temp = case.vehicle.create_polygon(X[0, i], X[1, i], X[3, i])
        veh = [[temp[0, 0], temp[0, 1]], [temp[1, 0], temp[1, 1]], [temp[2, 0], temp[2, 1]],
               [temp[3, 0], temp[3, 1]]]
        vehObsArea = VehObsArea(veh, obs)
        vehObsArea1 = VehObsArea(veh, obs1)
        vehObsArea2 = VehObsArea(veh, obs2)
        obsVehAeea = ObsVehArea(closestVec, veh)

        power = 1.56
        valueArea = ca.power(1 / (vehObsArea[0] + 0.0000001), power) + ca.power(1 / (vehObsArea[1] + 0.0000001), power) + \
                    ca.power(1 / (vehObsArea[2] + 0.0000001), power) + ca.power(1 / (vehObsArea[3] + 0.0000001), power) + \
                    ca.power(1 / (vehObsArea1[0] + 0.0000001), power) + ca.power(1 / (vehObsArea1[1] + 0.0000001), power) + \
                    ca.power(1 / (vehObsArea1[2] + 0.0000001), power) + ca.power(1 / (vehObsArea1[3] + 0.0000001), power) + \
                    ca.power(1 / (vehObsArea2[0] + 0.0000001), power) + ca.power(1 / (vehObsArea2[1] + 0.0000001), power) + \
                    ca.power(1 / (vehObsArea2[2] + 0.0000001), power) + ca.power(1 / (vehObsArea2[3] + 0.0000001), power) + \
                    ca.power(1 / (obsVehAeea + 0.0000001), 1*power)
        obj = obj + 1 * valueArea

    # constrains
    g = []  # equal constrains
    for i in range(N + 1):
        g.append(X[0, i])
        g.append(X[1, i])
        g.append(X[2, i])
        g.append(X[3, i])
        g.append(X[4, i])

    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P,
                'g': ca.vcat(g)}  # here also can use ca.vcat(g) or ca.vertcat(*g)
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6, }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # Simulation
    lbg = []
    ubg = []
    lbx = []
    ubx = []
    for _ in range(N + 1):
        lbg.append(x_min)  # x 最小值
        lbg.append(y_min)  # y 最小值
        lbg.append(v_min)  # v 最小值
        lbg.append(theta_min)  # theta 最小值
        lbg.append(-0.75)  # omega 最小值
        ubg.append(x_max)
        ubg.append(y_max)
        ubg.append(v_max)
        ubg.append(theta_max)
        ubg.append(0.75)

    for _ in range(N):
        lbx.append(-a_max)
        ubx.append(a_max)
        lbx.append(-omega_max)
        ubx.append(omega_max)

    t0 = 0.0

    startvec = coTrans(endvec, startvec)
    x0 = np.array([startvec[0], startvec[1], 0, startvec[2], 0]).reshape(-1, 1)
    # x0 = np.array([-16.1919692287797, -2.33749618274268, 1.0423756695225]).reshape(-1, 1)
    # x0 = np.array([-11.29353234, 1.069651741, 1.015800599]).reshape(-1, 1)  # initial state

    # xs = np.array([-16.31840796, -2.263681592, 1.061089133]).reshape(-1, 1)  # final state
    xs = np.array([0, 0, 0, 0, 0]).reshape(-1, 1)  # final state

    u0 = np.array([0.0, 0.0] * N).reshape(-1, 2)  # np.ones((N, 2)) # controls
    x_c = []  # contains for the history of the state
    u_c = []
    t_c = []  # for the time
    xx = []
    uu = []
    sim_time = 40.0

    # start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    c_p = np.concatenate((x0, xs))
    init_control = ca.reshape(u0, -1, 1)
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    lam_x_ = res['lam_x']
    last_loss = 0
    last_delta_loss = 0
    break_time = 0
    # inital test
    final_path = Path([], [], [])
    initQuadraticPath = []
    while (np.linalg.norm([x0[0] - xs[0], x0[1] - xs[1], x0[3] - xs[3]]) > 1e-2 and mpciter - sim_time / T < 0.0):
        # set parameter
        dis_loss = np.linalg.norm([x0[0] - xs[0], x0[1] - xs[1]])
        delta_loss = np.linalg.norm([x0[3] - xs[3]])
        print("Dis Loss: %s" % dis_loss)
        print("Delta Loss: %s" % delta_loss)
        print("Time: %s" % mpciter)
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx, lam_x0=lam_x_)
        lam_x_ = res['lam_x']
        # res = solver(x0=init_control, p=c_p,)
        # print(res['g'])
        index_t.append(time.time() - t_)
        u_sol = ca.reshape(res['x'], n_controls, N)  # one can only have this shape of the output
        ff_value = ff(u_sol, c_p)  # [n_states, N+1]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)

        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        uu.append(u_sol.full())
        mpciter = mpciter + 1
        if abs(dis_loss - last_loss) < 0.0000001 and abs(delta_loss - last_delta_loss) < 0.00000001:
            break_time += 1
        if break_time >= 20:
            break
        last_loss = dis_loss
        last_delta_loss = delta_loss

    t_v = np.array(index_t)

    # with open('coTrans_threeStateObsandVehQ_single_result%s.csv' % N, 'w', encoding='utf-8', newline='') as fp:
    #     # 写
    #     writer = csv.writer(fp)
    # 原点在新坐标系下的位置
    OriVec = coTrans(endvec, [0, 0, 0])
    # initQuadraticPath.append(OBCAPath(obca_start[0], obca_start[1], obca_start[2]))
    for i in range(len(xx)):
        # todo 将坐标反变换回原点
        _tvec = [xx[i][0][0], xx[i][1][0], xx[i][3][0]]
        resVec = coTrans(OriVec, _tvec)
        # writer.writerow([resVec[0], resVec[1], resVec[2], uu[i][0][0]])
        final_path.x.append(resVec[0])
        final_path.y.append(resVec[1])
        final_path.yaw.append(resVec[2])
        initQuadraticPath.append(OBCAPath(resVec[0], resVec[1], resVec[2]))

    initQuadraticPath.append(OBCAPath(endvec[0], endvec[1], endvec[2]))

    print(t_v.mean())
    print((time.time() - start_time) / (mpciter))

    # show(final_path, case, 7)
    # vehcfg = Vehicle()
    # path_x, path_y, path_yaw, path_steer = quadraticPath(initQuadraticPath, obstacles, vehcfg, case.xmax, case.ymax,
    #                                                      case.xmin, case.ymin)
    # obcaPath = Path(path_x, path_y, path_yaw)
    # show(obcaPath, case, 17)

    show(final_path, case, 7, "5-state-init")
    vehcfg = Vehicle()
    cfg = VehicleConfig()
    cfg.T = 0.2
    gap = 1
    sampleT = 0.2
    path_x, path_y, path_v, path_yaw, path_steer, path_a, path_steer_rate = quadraticPath(
                                                            initialQuadraticPath=initQuadraticPath, obstacles=obstacles,
                                                            vehicle=vehcfg, max_x=case.xmax, max_y=case.ymax,
                                                            min_x=case.xmin, min_y=case.ymin,
                                                            gap=gap, cfg=cfg, sampleT=sampleT)
    obcaPath = Path(path_x, path_y, path_yaw)
    show(obcaPath, case, 7, "5-state-obca")
    obcaPath_5gap = Path(path_x[::5], path_y[::5], path_yaw[::5])
    show(obcaPath_5gap, case, 7, "5-state-obca-5gap")
    path_t = [sampleT * k for k in range(len(path_x))]
    saveCsv(path_t=path_t, path_x=path_x, path_y=path_y, path_v=path_v, path_yaw=path_yaw, path_a=path_a,
            path_steer=path_steer, path_steer_rate=path_steer_rate, init_x=final_path.x, init_y=final_path.y,
            sampleT=sampleT, exp_name="5-state", path_num=7)


