import numpy as np
import math
import heapq

def interpData(path_t, path_x, path_y, path_v, path_a, path_yaw, path_steer, path_steer_rate):
    totalTime = int(path_t[-1])
    new_path_t = np.linspace(0, totalTime, int(totalTime/0.1)+1)
    new_path_x = np.interp(new_path_t, path_t, path_x)
    new_path_y = np.interp(new_path_t, path_t, path_y)
    new_path_a = np.interp(new_path_t[1:], path_t[1:], path_a)
    new_path_yaw = np.interp(new_path_t, path_t, path_yaw)
    new_path_steer = np.interp(new_path_t, path_t, path_steer)
    new_path_steer_rate = np.interp(new_path_t[1:], path_t[1:], path_steer_rate)
    new_path_v = np.interp(new_path_t, path_t, path_v)

    return new_path_t, new_path_x, new_path_y, new_path_v, new_path_a, new_path_yaw, new_path_steer, new_path_steer_rate


def generateObs(x, y, yaw, v, steer, env, args):
    observation = [x, y, yaw]
    if args.relative:
        observation = [0, 0, 0]
        observation += [env.x_goal-x, env.y_goal-y, env.yaw_goal-yaw]
    else:
        observation += [env.x_goal, env.y_goal, env.yaw_goal]
    polygon = env.vehicle.create_polygon(x, y, yaw)

    for i in range(4):
        if args.relative:
            observation += [float(polygon[i][0]) - x, float(polygon[i][1]) - y]
        else:
            observation += [float(polygon[i][0]), float(polygon[i][1])]

    dist = []
    for i in range(len(env.obstacles)):
        middle_x = np.mean([pos[0] for pos in env.obstacles[i]])
        middle_y = np.mean([pos[1] for pos in env.obstacles[i]])
        dis = math.sqrt((x - float(middle_x)) ** 2 + (y - float(middle_y)) ** 2)
        dist.append(dis)
    min_dis = heapq.nsmallest(3, dist)
    for mdis in min_dis:
        index = dist.index(mdis)
        for i in range(4):
            observation += [float(env.obstacles[index][i][0] - x),
                            float(env.obstacles[index][i][1] - y)]

    observation += [v, steer]

    return observation


def generateRealObs(x, y, yaw, v, steer, env):
    observation = [x, y, yaw]
    observation += [env.x_goal, env.y_goal, env.yaw_goal]
    polygon = env.vehicle.create_polygon(x, y, yaw)

    for i in range(4):
        observation += [float(polygon[i][0]), float(polygon[i][1])]

    dist = []
    for i in range(len(env.obstacles)):
        middle_x = np.mean([pos[0] for pos in env.obstacles[i]])
        middle_y = np.mean([pos[1] for pos in env.obstacles[i]])
        dis = math.sqrt((x - float(middle_x)) ** 2 + (y - float(middle_y)) ** 2)
        dist.append(dis)
    min_dis = heapq.nsmallest(3, dist)
    for mdis in min_dis:
        index = dist.index(mdis)
        for i in range(4):
            observation += [float(env.obstacles[index][i][0] - x),
                            float(env.obstacles[index][i][1] - y)]

    observation += [v, steer]

    return observation

