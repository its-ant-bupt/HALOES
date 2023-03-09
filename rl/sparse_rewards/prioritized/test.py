from typing import Dict, List
import os
import json

from tqdm import tqdm
import numpy as np
import gym

from rl.algorithms.PDDPG import Agent
from rl.environment.env import Environment
from rl.utils.savePath import show, saveCsv

class Path(object):
    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    # a::Array{Float64} # acc [m/s2]
    # steer::Array{Float64} # steer [rad]
    def __init__(self, x, y, yaw, v, a, steer_rate, steer):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a = a
        self.steer = steer
        self.steer_rate = steer_rate


def test(agent, path_num, env, args=None, testTime=20):
    OBS = env.reset(path_num)
    done = False
    if args.relative:
        total_x = [env.x_pos]
        total_y = [env.y_pos]
        total_yaw = [env.yaw]
    else:
        total_x = [OBS["observation"][0]]
        total_y = [OBS["observation"][1]]
        total_yaw = [OBS["observation"][2]]
    total_v = [0]
    total_steer_rate = [0]
    total_a = [0]
    total_steer = [0]
    score = 0
    # while not done and env.totalT:
    while env.totalT < testTime:
        state, curr_actgoal, curr_desgoal = OBS.values()
        assert len(curr_actgoal) == 3 and len(curr_desgoal) == 3, "Error!"
        obs = np.array(state)

        # Choose agent based action & make a transition
        action = agent.choose_action(obs, True)
        if args.alg == "IBDDPG":
            action = action[0]
        next_OBS, reward, done = env.step(action)
        if len(action) == 1:
            aValue, steerValue = env.action_list[int(action)][0], env.action_list[int(action)][1]
        else:
            aValue, steerValue = action[0] * agent.env.a_max, action[1] * agent.env.omega_max

        total_a.append(aValue)
        total_steer_rate.append(steerValue)

        next_state, next_actgoal, next_desgoal = next_OBS.values()
        assert len(next_actgoal) == 3 and len(next_desgoal) == 3, "Error!"
        next_obs = np.array(next_state)
        # total_x.append(next_obs[0])
        # total_y.append(next_obs[1])
        total_x.append(env.x_pos)
        total_y.append(env.y_pos)
        total_yaw.append(env.yaw)
        total_v.append(next_obs[-2])
        total_steer.append(next_obs[-1])

        OBS = next_OBS
        score += reward
        
    TotalPath = Path(total_x, total_y, total_yaw, total_v, total_a, total_steer_rate, total_steer)
    
    return score, TotalPath


if __name__ == '__main__':
    path_num = 7
    env = Environment(path_num)
    # Init. Datapath
    data_path = os.path.abspath('data/test')

    saveFigPath = os.path.join(data_path, "evaluate")

    if not os.path.exists(saveFigPath):
        os.mkdir(saveFigPath)

    # Init. Testing
    n_games = 10
    test_data = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games, training=False)
    agent.load_models(data_path)

    for i in tqdm(range(n_games), desc=f'Testing', total=n_games):
        score_history = [] * n_games

        for j in tqdm(range(n_games), desc=f'Testing', total=n_games):
            score, path = test(agent, path_num, env)
            show(path, env.case, path_num, os.path.join(saveFigPath, "Case-{}-{}-{}.jpg").format(path_num, i, j))
            path_t = [env.deltaT * k for k in range(len(path.x))]
            saveCsv(path_t=path_t, path_x=path.x, path_y=path.y, path_v=path.v, path_yaw=path.yaw, path_a=path.a,
                    path_steer=path.steer, path_steer_rate=path.steer_rate, init_x=env.case.x0, init_y=env.case.y0,
                    sampleT=env.deltaT, save_path=saveFigPath, i=i, j=j, case_num=path_num)

            score_history.append(score)

        print(f'Test Analysis:\n'
              f'Mean:{np.mean(score_history)}\n'
              f'Variance:{np.std(score_history)}')

        test_data.append({'Test Score': score_history})

    # Dump .json
    with open(os.path.join(data_path, 'testing_info.json'), 'w', encoding='utf8') as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
