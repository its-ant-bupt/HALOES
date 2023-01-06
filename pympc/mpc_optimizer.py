import casadi as ca
from casadi import *
from coordinatesTrans import coTrans
import time
from Vehicle import *


def transCoordinateObs(EndVec, obsList):
    newObsList = []
    for i in range(len(obsList)):
        tvec = [obsList[i][0], obsList[i][1], 0]
        newObs = coTrans(EndVec, tvec)
        newObsList.append([newObs[0], newObs[1]])
    return newObsList


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


def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t, st, u_end.T


def findClosestVec(VehVec, obsLists):
    dis = np.inf
    closest = []
    for i in range(len(obsLists)):
        for j in range(len(obsLists[i])):
            if np.hypot((obsLists[i][j][0]-VehVec[0]), (obsLists[i][j][1]-VehVec[1])) < dis:
                dis = np.hypot((obsLists[i][j][0]-VehVec[0]), (obsLists[i][j][1]-VehVec[1]))
                closest = [obsLists[i][j][0], obsLists[i][j][1]]
    return closest


class MPCOptimizer:
    def __init__(self, case, vehicle, args):
        self.case = case
        self.vehicle = vehicle
        self.args = args
        self.wheelbase = self.vehicle.lw
        if self.args.trans:
            self.x_min = -10
            self.x_max = 10
            self.y_min = -10
            self.y_max = 10
        else:
            self.x_min = self.case.xmin
            self.x_max = self.case.xmax
            self.y_min = self.case.ymin
            self.y_max = self.case.ymax

        self.v_min = self.vehicle.MIN_V
        self.v_max = self.vehicle.MAX_V
        self.theta_min = self.vehicle.MIN_THETA
        self.theta_max = self.vehicle.MAX_THETA
        self.delta_min = -self.vehicle.MAX_STEER
        self.delta_max = self.vehicle.MAX_STEER
        self.a_max = self.vehicle.MAX_A
        self.a_min = self.vehicle.MIN_A
        self.omega_max = self.vehicle.MAX_OMEGA  # 方向盘最大转向速率 rad/s
        self.omega_min = self.vehicle.MIN_OMEGA

        self.T = self.args.sample_time
        self.N = self.args.pre_length

        self.startVec = [case.x0, case.y0, case.theta0]
        self.endVec = [case.xf, case.yf, case.thetaf]

        self.oriObsList = []
        self.obsList = []

        # 优化器设置
        self.obj = 0  # 目标函数
        self.g = []  # 约束条件
        self.lbg = []  # 约束条件的下界
        self.ubg = []  # 约束条件的上界
        self.lbx = []  # 变量的下界
        self.ubx = []  # 变量的上界
        self.states = None
        self.n_states = 0
        self.controls = None
        self.n_controls = 0
        self.rhs = None
        self.f = None
        self.U = None
        self.X = None
        self.P = None
        self.ff = None

    def initialize(self):
        for obs_i in range(len(self.case.obs)):
            obs = list(self.case.obs[obs_i])
            if self.args.trans:
                self.obsList.append(transCoordinateObs(self.endVec, obs))
            else:
                self.obsList.append(obs)
            self.oriObsList.append(obs)

    def build_model(self):
        # 车辆状态参数
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')  # 车体方向角度
        delta = ca.SX.sym('delta')  # 车辆的方向盘转角
        self.states = ca.vertcat(ca.vertcat(x, y, v, theta), delta)
        self.n_states = self.states.size()[0]

        # 车辆控制参数
        a = ca.SX.sym('a')
        omega = ca.SX.sym('omega')  # 方向盘速率
        self.controls = ca.vertcat(a, omega)
        self.n_controls = self.controls.size()[0]

        # 控制器输入
        # rhs
        self.rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
        self.rhs = ca.vertcat(self.rhs, a)
        self.rhs = ca.vertcat(self.rhs, v * ca.tan(delta) / self.wheelbase)
        self.rhs = ca.vertcat(self.rhs, omega)

        # function
        self.f = ca.Function('f', [self.states, self.controls], [self.rhs], ['input_state', 'control_input'], ['rhs'])

        # for MPC
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        # define
        self.X[:, 0] = self.P[:5]  # initial condiction

        # define the relationship within the horizon
        for i in range(self.N):
            f_value = self.f(self.X[:, i], self.U[:, i])
            self.X[:, i + 1] = self.X[:, i] + f_value * self.T

        self.ff = ca.Function('ff', [self.U, self.P], [self.X], ['input_U', 'target_state'], ['horizon_states'])

    def generate_object(self, disCostFinal, deltaCostFinal, disCost, deltaCost, aCost, steerCost, obsPower):

        Sf = np.array([[disCostFinal, 0.0, 0.0, 0.0, 0.0],
                       [0.0, disCostFinal, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, deltaCostFinal, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        Q = np.array([[disCost, 0.0, 0.0, 0.0, 0.0],
                      [0.0, disCost, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, deltaCost, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 2.0]])
        R = np.array([[aCost, 0.0], [0.0, steerCost]])

        self.obj = self.obj + (self.X[:, self.N] - self.P[5:]).T @ Sf @ (self.X[:, self.N] - self.P[5:])
        self.obj = self.obj + (self.X[:, self.N - 1] - self.P[5:]).T @ Q @ (self.X[:, self.N - 1] - self.P[5:])
        for i in range(self.N - 1):
            # new type to calculate the matrix multiplication
            self.obj = self.obj + (self.X[:, i] - self.P[5:]).T @ Q @ (self.X[:, i] - self.P[5:]) + (self.U[:, i + 1] - self.U[:, i]).T @ R @ (
                    self.U[:, i + 1] - self.U[:, i])

        # 障碍物势场损失
        closestVec = findClosestVec([self.startVec[0], self.startVec[1]], self.oriObsList)
        closestVec = coTrans(self.endVec, [closestVec[0], closestVec[1], 0])
        for i in range(self.N + 1):
            temp = self.case.vehicle.create_polygon(self.X[0, i], self.X[1, i], self.X[3, i])
            veh = [[temp[0, 0], temp[0, 1]], [temp[1, 0], temp[1, 1]], [temp[2, 0], temp[2, 1]],
                   [temp[3, 0], temp[3, 1]]]
            vehObsAreas = []
            for obs_i in range(len(self.obsList)):
                vehObsArea = VehObsArea(veh, self.obsList[obs_i])
                for area_i in range(len(vehObsArea)):
                    vehObsAreas.append(vehObsArea[area_i])
            obsVehArea = ObsVehArea(closestVec, veh)

            valueArea = 0
            # valueArea += ca.power(1 / (obsVehArea + 0.0000001), obsPower)
            for area_i in range(len(vehObsAreas)):
                valueArea += ca.power(1 / (vehObsAreas[area_i] + 0.0000001), obsPower)

            self.obj = self.obj + 1 * valueArea

    def generate_constrain(self):
        for i in range(self.N + 1):
            self.g.append(self.X[0, i])
            self.g.append(self.X[1, i])
            self.g.append(self.X[2, i])
            self.g.append(self.X[3, i])
            self.g.append(self.X[4, i])

        for _ in range(self.N + 1):
            self.lbg.append(self.x_min)  # x 最小值
            self.lbg.append(self.y_min)  # y 最小值
            self.lbg.append(self.v_min)  # v 最小值
            self.lbg.append(self.theta_min)  # theta 最小值
            self.lbg.append(self.delta_min)  # delta 最小值
            self.ubg.append(self.x_max)
            self.ubg.append(self.y_max)
            self.ubg.append(self.v_max)
            self.ubg.append(self.theta_max)
            self.ubg.append(self.delta_max)

        for _ in range(self.N):
            self.lbx.append(self.a_min)
            self.ubx.append(self.a_max)
            self.lbx.append(self.omega_min)
            self.ubx.append(self.omega_max)

    def solve(self):
        nlp_prob = {'f': self.obj, 'x': ca.reshape(self.U, -1, 1), 'p': self.P,
                    'g': ca.vcat(self.g)}  # here also can use ca.vcat(g) or ca.vertcat(*g)
        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6, }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        if self.args.trans:
            StartVec = coTrans(self.endVec, self.startVec)
            x0 = np.array([StartVec[0], StartVec[1], 0, StartVec[2], 0]).reshape(-1, 1)
            xs = np.array([0, 0, 0, 0, 0]).reshape(-1, 1)  # final state
        else:
            x0 = np.array([self.startVec[0], self.startVec[1], 0, self.startVec[2], 0]).reshape(-1, 1)
            xs = np.array([self.endVec[0], self.endVec[1], 0, self.endVec[2], 0]).reshape(-1, 1)

        u0 = np.array([0.0, 0.0] * self.N).reshape(-1, 2)  # controls
        x_c = []  # contains for the history of the state
        u_c = []
        t_c = []  # for the time
        xx = []
        uu = []
        sim_time = 40.0

        # start MPC
        mpciter = 0
        t0 = 0.0
        start_time = time.time()
        index_t = []
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        res = solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
        lam_x_ = res['lam_x']
        last_loss = 0
        last_delta_loss = 0
        break_time = 0
        # initial test
        final_path = Path([], [], [])
        initQuadraticPath = []

        while (np.linalg.norm([x0[0] - xs[0], x0[1] - xs[1], x0[3] - xs[3]]) > 1e-2 and mpciter - sim_time / self.T < 0.0):
            # set parameter
            dis_loss = np.linalg.norm([x0[0] - xs[0], x0[1] - xs[1]])
            delta_loss = np.linalg.norm([x0[3] - xs[3]])
            print("Dis Loss: %s" % dis_loss)
            print("Delta Loss: %s" % delta_loss)
            print("Time: %s" % mpciter)
            c_p = np.concatenate((x0, xs))
            init_control = ca.reshape(u0, -1, 1)
            t_ = time.time()
            res = solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx, lam_x0=lam_x_)
            lam_x_ = res['lam_x']
            index_t.append(time.time() - t_)
            u_sol = ca.reshape(res['x'], self.n_controls, self.N)  # one can only have this shape of the output
            ff_value = self.ff(u_sol, c_p)  # [n_states, N+1]
            x_c.append(ff_value)
            u_c.append(u_sol[:, 0])
            t_c.append(t0)
            t0, x0, u0 = shift_movement(self.T, t0, x0, u_sol, self.f)

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
        print(t_v.mean())
        print((time.time() - start_time) / (mpciter))

        # 原点在新坐标系下的位置
        if self.args.trans:
            OriVec = coTrans(self.endVec, [0, 0, 0])
        for i in range(len(xx)):
            # 将坐标反变换回原点
            _tvec = [xx[i][0][0], xx[i][1][0], xx[i][3][0]]
            if self.args.trans:
                resVec = coTrans(OriVec, _tvec)
            else:
                resVec = _tvec
            # writer.writerow([resVec[0], resVec[1], resVec[2], uu[i][0][0]])
            final_path.x.append(resVec[0])
            final_path.y.append(resVec[1])
            final_path.yaw.append(resVec[2])
            initQuadraticPath.append(OBCAPath(resVec[0], resVec[1], resVec[2]))

        initQuadraticPath.append(OBCAPath(self.endVec[0], self.endVec[1], self.endVec[2]))

        return final_path, initQuadraticPath






