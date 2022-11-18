#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
# from shapely.geometry import Polygon  # 多边形
import time
from draw import Draw_MPC_point_stabilization_v1
from draw import Draw_MPC_Obstacle
import csv


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
                                        obsList[j+1][0], obsList[j+1][1])

        exArea[v_i] = exArea[v_i] - obsArea
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


class Vehicle:
    def __init__(self):
        self.lw = 2.8  # wheelbase
        self.lf = 0.96  # front hang length
        self.lr = 0.929  # rear hang length
        self.lb = 1.942  # width

    def create_polygon(self, x, y, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)


        AX = x + (self.lf + self.lw) * cos_theta - (self.lb/2) * sin_theta
        BX = x + (self.lf + self.lw) * cos_theta + (self.lb/2) * sin_theta
        CX = x - self.lr * cos_theta + (self.lb/2) * sin_theta
        DX = x - self.lr * cos_theta - (self.lb/2) * sin_theta
        AY = y + (self.lf + self.lw) * sin_theta + (self.lb/2) * cos_theta
        BY = y + (self.lf + self.lw) * sin_theta - (self.lb/2) * cos_theta
        CY = y - self.lr * sin_theta - (self.lb/2) * cos_theta
        DY = y - self.lr * sin_theta + (self.lb/2) * cos_theta
        points = np.array([[AX, AY], [BX, BY], [CX, CY], [DX, DY]])
        # points = np.array([
        #     [-self.lr, -self.lb / 2, 1],
        #     [self.lf + self.lw, -self.lb / 2, 1],
        #     [self.lf + self.lw, self.lb / 2, 1],
        #     [-self.lr, self.lb / 2, 1],
        #     [-self.lr, -self.lb / 2, 1],
        # ]).dot(np.array([
        #     [cos_theta, -sin_theta, x],
        #     [sin_theta, cos_theta, y],
        #     [0, 0, 1]
        # ]).transpose())
        # return points[:, 0:2]
        return points


class Case:
    def __init__(self):
        self.x0, self.y0, self.theta0 = 0, 0, 0
        self.xf, self.yf, self.thetaf = 0, 0, 0
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.obs_num = 0
        self.obs = np.array([])
        self.vehicle = Vehicle()

    @staticmethod
    def read(file):
        case = Case()
        with open(file, 'r') as f:
            reader = csv.reader(f)
            tmp = list(reader)
            v = [float(i) for i in tmp[0]]
            case.x0, case.y0, case.theta0 = v[0:3]
            case.xf, case.yf, case.thetaf = v[3:6]
            case.xmin = min(case.x0, case.xf) - 10
            case.xmax = max(case.x0, case.xf) + 10
            case.ymin = min(case.y0, case.yf) - 10
            case.ymax = max(case.y0, case.yf) + 10

            case.obs_num = int(v[6])  # 获取障碍物数目
            num_vertexes = np.array(v[7:7 + case.obs_num], dtype=np.int)  # 获取每个障碍物的边数
            # 计算每个障碍物顶点坐标的开始位置
            vertex_start = 7 + case.obs_num + (np.cumsum(num_vertexes, dtype=np.int) - num_vertexes) * 2
            case.obs = []
            for vs, nv in zip(vertex_start, num_vertexes):
                # 添加每个障碍物顶点的坐标
                case.obs.append(np.array(v[vs:vs + nv * 2]).reshape((nv, 2), order='A'))
        return case


def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end.T


if __name__ == '__main__':
    case = Case.read('BenchmarkCases/Case%d.csv' % 7)
    wheelbase = 2.8
    T = 0.05  # sampling time [s]
    N = 5  # prediction horizon
    x_min = case.xmin
    x_max = case.xmax
    y_min = case.ymin
    y_max = case.ymax
    v_max = 2
    v_min = -2
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
    theta = ca.SX.sym('theta')  # 车体方向角度
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.size()[0]

    # 车辆控制参数
    v = ca.SX.sym('v')
    delta = ca.SX.sym('delta')  # 方向盘速率
    controls = ca.vertcat(v, delta)
    n_controls = controls.size()[0]

    # 控制器输入
    ## rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
    rhs = ca.vertcat(rhs, v * ca.tan(delta) / wheelbase)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N + 1)
    P = ca.SX.sym('P', n_states + n_states)

    ### define
    X[:, 0] = P[:3]  # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[:, i], U[:, i])
        X[:, i + 1] = X[:, i] + f_value * T

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])



    Sf = np.array([[30000.0, 0.0, 0.0],
                   [0.0, 30000.0, 0.0],
                   [0.0, 0.0, 10000.0]])

    Q = np.array([[1000.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0],
                  [0.0, 0.0, 10.0]])
    # R = np.array([[0.0, 0.0], [0.0, 2]])
    R = np.array([[0.0, 0.0], [0.0, 0.0]])

    # cost function
    obj = 0  # cost
    obj = obj + (X[:, N] - P[3:]).T @ Sf @ (X[:, N] - P[3:])
    obj = obj + (X[:, N-1] - P[3:]).T @ Q @ (X[:, N-1] - P[3:])
    for i in range(N-1):
        # new type to calculate the matrix multiplication
        obj = obj + (X[:, i] - P[3:]).T @ Q @ (X[:, i] - P[3:]) + (U[:, i+1]-U[:, i]).T @ R @ (U[:, i+1]-U[:, i])

    #### obs cost
    obs = [[-25.03567043, -15.8687107], [-17.71684521, -2.775399527], [-16.02169786, -3.722943432],
           [-23.34052308, -16.8162546]]
    obs1 = [[-15.18501961, 1.754013251], [-7.866194392, 14.84732442], [-6.171047039, 13.89978052],
            [-13.48987226, 0.806469346]]
    obs2 = [[-14.10682142, 3.958046312], [-18.59643337, -3.839700761], [-18.75396361, -3.760935639],
            [-13.16163996, 5.809026678]]

    for i in range(N + 1):
        temp = case.vehicle.create_polygon(X[0, i], X[1, i], X[2, i])
        veh = [[temp[0, 0], temp[0, 1]], [temp[1, 0], temp[1, 1]], [temp[2, 0], temp[2, 1]],
               [temp[3, 0], temp[3, 1]]]
        vehObsArea = VehObsArea(veh, obs)
        vehObsArea1 = VehObsArea(veh, obs1)
        vehObsArea2 = VehObsArea(veh, obs2)

        valueArea = ca.power(1/(vehObsArea[0]+0.0000001),2) + ca.power(1/(vehObsArea[1]+0.0000001),2) +\
                    ca.power(1/(vehObsArea[2]+0.0000001),2) + ca.power(1/(vehObsArea[3]+0.0000001),2) +\
                    ca.power(1/(vehObsArea1[0]+0.0000001),2) + ca.power(1/(vehObsArea1[1]+0.0000001),2) + \
                    ca.power(1/(vehObsArea1[2]+0.0000001),2) + ca.power(1/(vehObsArea1[3]+0.0000001),2) + \
                    ca.power(1/(vehObsArea2[0]+0.0000001),2) + ca.power(1/(vehObsArea2[1]+0.0000001),2) + \
                    ca.power(1/(vehObsArea2[2]+0.0000001),2) + ca.power(1/(vehObsArea2[3]+0.0000001),2)
        obj = obj + 1*valueArea


    #### constrains
    g = []  # equal constrains
    for i in range(N + 1):
        g.append(X[0, i])
        g.append(X[1, i])
        g.append(X[2, i])




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
        lbg.append(theta_min)  # theta 最小值
        ubg.append(x_max)
        ubg.append(y_max)
        ubg.append(theta_max)

    for _ in range(N):
        lbx.append(v_min)  # v 最小值
        lbx.append(-0.75)  # omega 最小值
        ubx.append(v_max)
        ubx.append(0.75)

    t0 = 0.0
    
    x0 = np.array([-11.29353234+0.75, 1.069651741+0.75, 1.015800599]).reshape(-1, 1)
    startX = x0

    xs = np.array([-16.31840796, -2.263681592, 1.061089133]).reshape(-1, 1)  # final state

    u0 = np.array([0.0, 0.0] * N).reshape(-1, 2)  # np.ones((N, 2)) # controls
    x_c = []  # contains for the history of the state
    u_c = []
    t_c = []  # for the time
    xx = []
    uu = []
    sim_time = 40.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    c_p = np.concatenate((x0, xs))
    init_control = ca.reshape(u0, -1, 1)
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    lam_x_ = res['lam_x']
    ### inital test
    while (np.linalg.norm([x0[0] - xs[0], x0[1] - xs[1], x0[2] - xs[2]]) > 1e-2 and mpciter - sim_time / T < 0.0):
        ## set parameter
        print("Dis Loss: %s" % (np.linalg.norm([x0[0] - xs[0], x0[1] - xs[1]])))
        print("Delta Loss: %s" % (np.linalg.norm([x0[2] - xs[2]])))
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
    t_v = np.array(index_t)
    with open('threeStateObsQ_single_result%s.csv'%N, 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        writer.writerow([startX[0][0], startX[1][0], startX[2][0], 0, 0])
        for i in range(len(xx)):
            writer.writerow([xx[i][0][0], xx[i][1][0], xx[i][2][0], uu[i][0][0], uu[i][1][0]])


    print(t_v.mean())
    print((time.time() - start_time) / (mpciter))
    # draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0.full(), target_state=xs, robot_states=xx, export_fig=True)

    # draw_result = Draw_MPC_Obstacle(rob_diam=0.3, init_state=x0.full(), target_state=xs, robot_states=xx, obstacle=np.array([obs_x, obs_y, obs_diam/2.]), export_fig=True)
