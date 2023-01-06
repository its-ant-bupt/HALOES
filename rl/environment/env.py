import math
import numpy as np
from case import Case
from Vehicle import Vehicle
import heapq


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
    return 0.5 * abs(
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
    shapeArea = 0.5 * abs(np.dot(_x, np.roll(_y, 1)) - np.dot(_y, np.roll(_x, 1)))
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


class Environment:
    def __init__(self, path_num):
        self.case = Case.read('../../../BenchmarkCases/Case%d.csv' % path_num)
        self.vehicle = Vehicle()
        self.deltaT = 0.1  # The interval time of simulation
        self.MaxT = 40
        self._max_episode_steps = self.MaxT / self.deltaT

        self.totalT = 0

        # ==================== Vehicle Config ==========================
        self.v_min = self.vehicle.MIN_V  # velocity m/s
        self.v_max = self.vehicle.MAX_V
        self.theta_min = self.vehicle.MIN_THETA  # yaw rad
        self.theta_max = self.vehicle.MAX_THETA
        self.delta_min = -self.vehicle.MAX_STEER  # steer rad
        self.delta_max = self.vehicle.MAX_STEER
        self.a_max = self.vehicle.MAX_A  # acc m/s2
        self.a_min = self.vehicle.MIN_A
        self.omega_max = self.vehicle.MAX_OMEGA  # 方向盘最大转向速率 rad/s
        self.omega_min = self.vehicle.MIN_OMEGA

        self.x_pos = self.case.x0
        self.y_pos = self.case.y0
        self.yaw = self.case.theta0
        self.v_current = 0
        self.steer = 0

        self.x_goal = self.case.xf
        self.y_goal = self.case.yf
        self.yaw_goal = self.case.thetaf

        self.dis2goal = (self.x_goal - self.x_pos)**2 + (self.y_goal - self.y_pos)**2
        self.yaw_error = (self.yaw_goal - self.yaw)
        # ==============================================================

        # ==================== Obstacle Config =========================
        self.obstacles = []
        for obs_i in range(len(self.case.obs)):
            obs = list(self.case.obs[obs_i])
            self.obstacles.append(obs)
        # ==============================================================

        self.action_list = self.build_action()
        self.observation_space = 6 + 8 + 3 * 8

    def reset(self, path_num):
        '''
        Args:
            path_num: the number of parking case
        Returns:
            observation: the observation information of vehicle
        '''
        self.case = Case.read('../../../BenchmarkCases/Case%d.csv' % path_num)
        self.totalT = 0
        # ==================== Vehicle Config ==========================
        self.x_pos = self.case.x0
        self.y_pos = self.case.y0
        self.yaw = self.case.theta0
        self.v_current = 0
        self.steer = 0

        self.x_goal = self.case.xf
        self.y_goal = self.case.yf
        self.yaw_goal = self.case.thetaf

        self.dis2goal = (self.x_goal - self.x_pos) ** 2 + (self.y_goal - self.y_pos) ** 2
        self.yaw_error = (self.yaw_goal - self.yaw)
        # ==============================================================

        # ==================== Obstacle Config =========================
        self.obstacles = []
        for obs_i in range(len(self.case.obs)):
            obs = list(self.case.obs[obs_i])
            self.obstacles.append(obs)
        # ==============================================================

        return self.get_obs()

    def get_obs(self):
        '''
        Returns: observation [x, y, yaw, x_goal, y_goal, yaw_goal,
                              x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4  # 四个顶点的坐标
                              # 与车辆最近的三个障碍物的坐标
                              obs_x_1, obs_y_1, obs_x_2, obs_y_2, obs_x_3, obs_y_3, obs_x_4, obs_y_4  # i-th obstacle
                              ]
        '''
        observation = [self.x_pos, self.y_pos, self.yaw,
                       self.x_goal, self.y_goal, self.yaw_goal]
        polygon = self.vehicle.create_polygon(self.x_pos, self.y_pos, self.yaw)
        for i in range(4):
            observation += [float(polygon[i][0]), float(polygon[i][1])]
            
        dist = []
        for i in range(len(self.obstacles)):
            middle_x = np.mean([pos[0] for pos in self.obstacles[i]])
            middle_y = np.mean([pos[1] for pos in self.obstacles[i]])
            dis = math.sqrt((self.x_pos-float(middle_x))**2 + (self.y_pos - float(middle_y))**2)
            dist.append(dis)
        min_dis = heapq.nsmallest(3, dist)
        for mdis in min_dis:
            index = dist.index(mdis)
            for i in range(4):
                observation += [float(self.obstacles[index][i][0] - self.x_pos),
                                float(self.obstacles[index][i][1] - self.y_pos)]

        OBS = {
            "observation": np.array(observation),
            "currentgoal": np.array([self.x_pos, self.y_pos, self.yaw]),
            "actualgoal": np.array([self.x_goal, self.y_goal, self.yaw_goal])
        }

        return OBS

    def build_action(self):
        Amax = self.a_max
        SteerMax = self.omega_max
        interval = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        actionList = {}
        num = 0
        for i in range(len(interval)):
            for j in range(len(interval)):
                actionList[num] = [Amax*interval[i], SteerMax*interval[j]]
                num += 1
        return actionList

    def step(self, action):
        '''
        Args:
            action: [a, steering rate]
        Returns:
            next observation
            reward
            done?
        '''
        self.totalT += self.deltaT
        a = self.action_list[int(action)][0]
        steer_rate = self.action_list[int(action)][1]

        current_v = self.v_current
        self.x_pos = self.x_pos + current_v * self.deltaT * math.cos(self.yaw)
        self.y_pos = self.y_pos + current_v * self.deltaT * math.sin(self.yaw)
        self.v_current = self.v_current + a * self.deltaT
        self.yaw = self.mod2pi(current_v * self.deltaT * math.tan(self.steer) / self.vehicle.lw)
        self.steer = self.steer + steer_rate * self.deltaT

        reward = -1  # time loss
        if self.v_current < self.v_min:
            reward -= abs(self.v_current - self.v_min) * 10
            self.v_current = self.v_min
        if self.v_current > self.v_max:
            reward -= abs(self.v_current - self.v_max) * 10
            self.v_current = self.v_max

        observation = self.get_obs()

        done = 0

        if (abs(self.x_pos - self.x_goal) < 0.01 and
                abs(self.y_pos - self.y_goal) < 0.01 and
                abs(self.yaw - self.yaw_goal) < 0.01):
            done = 1
            reward += 100

        dis_loss, obstacle_loss = self.calculate_reward(observation["observation"], [self.x_goal, self.y_goal, self.yaw_goal], self.obstacles)

        reward = reward + dis_loss + obstacle_loss

        if self.totalT >= self.MaxT:
            done = 1

        return observation, reward, done

    def calculate_reward(self, obs, goal, obstacle):
        dis_loss = self.reward_goal(obs, goal)
        obstacle_loss = self.reward_obstacle(obs, obstacle)
        return dis_loss, obstacle_loss
    
    def reward_goal(self, obs, goal):
        dis = (goal[0] - obs[0])**2 + (goal[1] - obs[1])**2
        delta_dis = self.dis2goal - dis
        self.dis2goal = dis
        yawdelta = goal[2] - obs[2]
        delta_yaw = self.yaw_error - yawdelta
        self.yaw_error = yawdelta
        return 0.1 * delta_dis + delta_yaw

    def reward_obstacle(self, obs, obstacle):
        reward = 1
        veh = [[obs[6], obs[7]], [obs[8], obs[9]], [obs[10], obs[11]], [obs[12], obs[13]]]
        for obs_i in range(len(obstacle)):
            vehObsArea = VehObsArea(veh, obstacle[obs_i])
            for area_i in range(len(vehObsArea)):
                if vehObsArea[area_i] <= 0.001:
                    reward -= 10
        return reward

    def mod2pi(self, yaw):
        v = yaw % (2*math.pi)
        if v > math.pi:
            v -= 2*math.pi
        elif v < -math.pi:
            v += 2*math.pi
        return v


if __name__ == '__main__':
    env = Environment(7)
    obs = env.get_obs()
    reward_1, reward_2 = env.calculate_reward(obs, [env.x_goal, env.y_goal, env.yaw_goal], env.obstacles)
